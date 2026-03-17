from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import torch


def _add_project_root_to_sys_path() -> None:
	current_file = Path(__file__).resolve()
	for parent in current_file.parents:
		if (parent / "pyproject.toml").exists():
			project_root = str(parent)
			if project_root not in sys.path:
				sys.path.insert(0, project_root)
			return


_add_project_root_to_sys_path()


from infra.common.logging.logger import logger
from infra.engine.flows.common.runtime import build_flow_runtime


def _build_model(config_path: str, overrides: Iterable[str]) -> tuple[torch.nn.Module, object]:
	"""Build the RF-DETR model from an experiment config.

	This reuses the standard flow runtime so the exported model matches
	the training/eval configuration used by the framework.
	"""
	runtime = build_flow_runtime(
		overrides=list(overrides),
		config_path=config_path,
		build_loaders=False,
	)
	return runtime.built.model, runtime.wrapper


def _load_checkpoint(model: torch.nn.Module, wrapper: object, checkpoint: str) -> None:
	if not checkpoint:
		logger.warning("No checkpoint provided, exporting randomly initialized weights")
		return

	state = wrapper.load_checkpoint_state(checkpoint)
	wrapper.validate_checkpoint_class_compatibility(model, state)
	loaded, skipped, missing = wrapper.safe_load_state_dict(model, state)
	logger.info(
		"Loaded checkpoint tensors=%d, skipped_shape=%d, missing=%d",
		loaded,
		skipped,
		missing,
	)


def _infer_output_names(outputs) -> List[str]:
	if isinstance(outputs, tuple):
		if len(outputs) == 3:
			return ["dets", "labels", "masks"]
		if len(outputs) == 2:
			return ["dets", "labels"]
		return [f"output_{idx}" for idx in range(len(outputs))]
	return ["output"]


def _infer_input_channels(model: torch.nn.Module) -> int:
	for parameter in model.parameters():
		if parameter.dim() == 4:
			return int(parameter.shape[1])
	return 3


def _export_onnx(
	model: torch.nn.Module,
	output_file: str,
	input_size: int,
	check: bool,
	simplify: bool,
) -> None:
	device = torch.device("cpu")
	model = model.to(device)
	model.eval()

	if hasattr(model, "export") and callable(getattr(model, "export")):
		model.export()

	channels = _infer_input_channels(model)
	dummy = torch.rand(1, channels, input_size, input_size, device=device)

	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with torch.no_grad():
		sample_outputs = model(dummy)

	input_names = ["images"]
	output_names = _infer_output_names(sample_outputs)

	dynamic_axes = {"images": {0: "N"}}
	for name in output_names:
		dynamic_axes[name] = {0: "N"}

	torch.onnx.export(
		model,
		dummy,
		str(output_path),
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		opset_version=17,
		do_constant_folding=True,
		verbose=False,
	)

	logger.info("Exported ONNX model to %s", output_file)

	if check:
		import onnx

		onnx_model = onnx.load(output_file)
		onnx.checker.check_model(onnx_model)
		logger.info("ONNX model check passed")

	if simplify:
		import onnx
		import onnxsim

		onnx_model, ok = onnxsim.simplify(output_file)
		if not ok:
			raise RuntimeError("onnx-simplifier reported failure while simplifying model")
		onnx.save(onnx_model, output_file)
		logger.info("Simplified ONNX model saved to {}", output_file)


def main(args: argparse.Namespace) -> None:
	overrides: List[str] = list(args.update or [])
	config_path = str(Path(args.config).resolve())
	model, wrapper = _build_model(config_path=config_path, overrides=overrides)

	_load_checkpoint(model=model, wrapper=wrapper, checkpoint=args.resume)

	output_path = str(Path(args.output_file).resolve())
	_export_onnx(
		model=model,
		output_file=output_path,
		input_size=int(args.input_size),
		check=bool(args.check),
		simplify=bool(args.simplify),
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", "-c", type=str, required=True)
	parser.add_argument("--resume", "-r", type=str, required=True)
	parser.add_argument("--output_file", "-o", type=str, default="model.onnx")
	parser.add_argument("--input_size", "-s", type=int, default=640)
	parser.add_argument("--check", action="store_true", default=False)
	parser.add_argument("--simplify", action="store_true", default=False)
	parser.add_argument("--update", "-u", nargs="+", help="config overrides")

	cli_args = parser.parse_args()
	main(cli_args)

