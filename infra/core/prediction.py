from __future__ import annotations


def to_result_list(outputs, postprocessor, orig_sizes):
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs

    if isinstance(outputs, dict):
        processed = postprocessor(outputs, orig_sizes)
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        labels, boxes, scores = outputs
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]
    else:
        processed = postprocessor(outputs, orig_sizes)

    if isinstance(processed, list):
        return processed

    if isinstance(processed, (tuple, list)) and len(processed) == 3:
        labels, boxes, scores = processed
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    if isinstance(processed, dict) and {"labels", "boxes", "scores"}.issubset(set(processed.keys())):
        labels = processed["labels"]
        boxes = processed["boxes"]
        scores = processed["scores"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    raise TypeError(f"Unsupported output format: {type(processed)}")
