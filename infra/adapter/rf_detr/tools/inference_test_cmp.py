from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import numpy as np
import onnxruntime as ort
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall
from rfdetr.util.coco_classes import COCO_CLASSES

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = SCRIPT_DIR / ".." / "models" / "rfdetr_small_3ch.onnx"
SCORE_THRESHOLD = 0.5
NUM_SELECT = 300
MEANS = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STDS = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SOURCE_OUTPUT = SCRIPT_DIR / "annotated_image_source.jpg"
ONNX_OUTPUT = SCRIPT_DIR / "annotated_image_onnx.jpg"
REPORT_OUTPUT = SCRIPT_DIR / "comparison_report.json"


def _class_name(class_id: int) -> str:
	idx = int(class_id)
	if idx in COCO_CLASSES:
		return str(COCO_CLASSES[idx])
	return f"class_{idx}"


def _build_labels(class_ids: np.ndarray, scores: np.ndarray) -> list[str]:
	return [
		f"{_class_name(int(class_id))}: {float(score):.2f}"
		for class_id, score in zip(class_ids, scores)
	]


def _prepare_input(image: Image.Image, input_height: int, input_width: int) -> np.ndarray:
	resized = image.resize((input_width, input_height), Image.BILINEAR)
	image_array = np.asarray(resized, dtype=np.float32) / 255.0
	image_array = (image_array - MEANS) / STDS
	image_array = np.transpose(image_array, (2, 0, 1))[None, ...]
	return image_array


def _decode_outputs(
	outputs: list[np.ndarray],
	original_width: int,
	original_height: int,
	score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	out_bbox = outputs[0]
	out_logits = outputs[1] if len(outputs) > 1 else None

	if out_bbox.ndim == 3:
		out_bbox = out_bbox[0]
	if out_logits is None:
		raise ValueError("ONNX outputs must include classification logits as second output")
	if out_logits.ndim == 3:
		out_logits = out_logits[0]

	prob = 1.0 / (1.0 + np.exp(-out_logits.astype(np.float32)))
	num_queries, num_classes = prob.shape
	flat_prob = prob.reshape(-1)
	num_select = min(NUM_SELECT, flat_prob.shape[0])

	topk_idx = np.argpartition(-flat_prob, num_select - 1)[:num_select]
	topk_scores = flat_prob[topk_idx]
	order = np.argsort(-topk_scores)
	topk_idx = topk_idx[order]
	scores = topk_scores[order]

	topk_boxes = (topk_idx // num_classes).astype(np.int64)
	class_ids = (topk_idx % num_classes).astype(np.int64)

	selected_cxcywh = out_bbox[topk_boxes].astype(np.float32)
	cx = selected_cxcywh[:, 0]
	cy = selected_cxcywh[:, 1]
	w = selected_cxcywh[:, 2]
	h = selected_cxcywh[:, 3]

	x1 = (cx - 0.5 * w) * float(original_width)
	y1 = (cy - 0.5 * h) * float(original_height)
	x2 = (cx + 0.5 * w) * float(original_width)
	y2 = (cy + 0.5 * h) * float(original_height)
	boxes = np.stack([x1, y1, x2, y2], axis=1)

	keep = scores >= float(score_threshold)
	return boxes[keep], scores[keep], class_ids[keep]


def _to_detections_payload(detections: sv.Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	boxes = np.asarray(detections.xyxy, dtype=np.float32)
	class_ids = detections.class_id
	if class_ids is None:
		class_ids = np.zeros((len(boxes),), dtype=np.int64)
	else:
		class_ids = np.asarray(class_ids, dtype=np.int64)
	conf = detections.confidence
	if conf is None:
		conf = np.ones((len(boxes),), dtype=np.float32)
	else:
		conf = np.asarray(conf, dtype=np.float32)
	return boxes, conf, class_ids


def _annotate_and_save(
	image: Image.Image,
	boxes: np.ndarray,
	scores: np.ndarray,
	class_ids: np.ndarray,
	output_path: Path,
) -> None:
	detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
	labels = _build_labels(class_ids, scores)
	annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
	annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
	if isinstance(annotated, Image.Image):
		annotated.save(str(output_path))
	else:
		Image.fromarray(annotated).save(str(output_path))


def _topk_payload(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, top_k: int = 10) -> list[dict[str, object]]:
	if len(scores) == 0:
		return []
	indices = np.argsort(-scores)[:top_k]
	payload: list[dict[str, object]] = []
	for idx in indices:
		box = boxes[int(idx)].tolist()
		payload.append(
			{
				"label": _class_name(int(class_ids[int(idx)])),
				"score": float(scores[int(idx)]),
				"bbox_xyxy": [float(v) for v in box],
			}
		)
	return payload


def main() -> None:
	if not ONNX_MODEL_PATH.exists():
		raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

	session = ort.InferenceSession(
		str(ONNX_MODEL_PATH),
		providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
	)
	input_info = session.get_inputs()[0]
	input_name = input_info.name

	input_shape = input_info.shape
	input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 640
	input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 640

	response = requests.get("https://media.roboflow.com/dog.jpg", timeout=30)
	response.raise_for_status()
	image = Image.open(BytesIO(response.content)).convert("RGB")

	source_model = RFDETRSmall()
	source_detections = source_model.predict(image, threshold=SCORE_THRESHOLD)
	source_boxes, source_scores, source_class_ids = _to_detections_payload(source_detections)
	_annotate_and_save(
		image=image,
		boxes=source_boxes,
		scores=source_scores,
		class_ids=source_class_ids,
		output_path=SOURCE_OUTPUT,
	)

	model_input = _prepare_input(image, input_height=input_height, input_width=input_width)
	outputs = session.run(None, {input_name: model_input})

	onnx_boxes, onnx_scores, onnx_class_ids = _decode_outputs(
		outputs=outputs,
		original_width=image.width,
		original_height=image.height,
		score_threshold=SCORE_THRESHOLD,
	)
	_annotate_and_save(
		image=image,
		boxes=onnx_boxes,
		scores=onnx_scores,
		class_ids=onnx_class_ids,
		output_path=ONNX_OUTPUT,
	)

	report = {
		"threshold": SCORE_THRESHOLD,
		"source_model": {
			"name": "RFDETRSmall",
			"num_detections": int(len(source_scores)),
			"top_detections": _topk_payload(source_boxes, source_scores, source_class_ids),
		},
		"onnx_model": {
			"path": str(ONNX_MODEL_PATH),
			"num_detections": int(len(onnx_scores)),
			"top_detections": _topk_payload(onnx_boxes, onnx_scores, onnx_class_ids),
		},
		"outputs": {
			"source_image": str(SOURCE_OUTPUT),
			"onnx_image": str(ONNX_OUTPUT),
		},
	}
	REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(f"Saved {SOURCE_OUTPUT}, {ONNX_OUTPUT}, {REPORT_OUTPUT}")


if __name__ == "__main__":
	main()
