from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from infra.core import to_canonical_predictions
from infra.engine.flows.common.runtime import build_flow_runtime


class _InferenceExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, postprocessor: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        results = to_canonical_predictions(outputs, self.postprocessor, orig_target_sizes)
        labels = torch.stack([result["labels"].to(torch.int64) for result in results], dim=0)
        boxes = torch.stack([result["boxes"].to(torch.float32) for result in results], dim=0)
        scores = torch.stack([result["scores"].to(torch.float32) for result in results], dim=0)
        return labels, boxes, scores


def _load_runtime(config_path: str):
    return build_flow_runtime(overrides=[], config_path=config_path, build_loaders=False)


def _load_checkpoint(runtime, checkpoint: str) -> None:
    state = runtime.wrapper.load_checkpoint_state(checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info(
        "Loaded checkpoint tensors={}, skipped_shape={}, missing={}",
        loaded,
        skipped,
        missing,
    )


def _export_onnx(runtime, output_file: str, input_size: int, check: bool, simplify: bool) -> None:
    model = runtime.built.model
    postprocessor = runtime.built.postprocessor
    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    wrapper = _InferenceExportWrapper(model=model, postprocessor=postprocessor).eval().to("cpu")

    dummy_images = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_sizes = torch.tensor([[input_size, input_size]], dtype=torch.int64)

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_sizes),
        str(output_path),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
            "labels": {0: "N"},
            "boxes": {0: "N"},
            "scores": {0: "N"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    logger.info("Exported ONNX model to {}", output_path)

    if check:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model check passed")

    if simplify:
        import onnx
        import onnxsim

        onnx_model, ok = onnxsim.simplify(str(output_path))
        if not ok:
            raise RuntimeError("onnx-simplifier reported failure")
        onnx.save(onnx_model, str(output_path))
        logger.info("Simplified ONNX model saved to {}", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str)
    parser.add_argument("--resume", "-r", required=True, type=str)
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default="infra/adapter/rtdetrv2_pytorch/models/rtdetrv2_r18vd_3ch.onnx",
    )
    parser.add_argument("--input-size", "-s", type=int, default=640)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=False)
    args = parser.parse_args()

    runtime = _load_runtime(config_path=str(Path(args.config).resolve()))
    _load_checkpoint(runtime, checkpoint=str(Path(args.resume).resolve()))
    _export_onnx(
        runtime,
        output_file=args.output_file,
        input_size=int(args.input_size),
        check=bool(args.check),
        simplify=bool(args.simplify),
    )


if __name__ == "__main__":
    main()
