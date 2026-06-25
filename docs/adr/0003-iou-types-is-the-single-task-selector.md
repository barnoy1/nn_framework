# iou_types is the single task selector

The task (detection vs instance segmentation) is selected solely by `data.iou_types`, a subset of {`bbox`, `segm`}; instance segmentation is "detection plus the `segm` IoU type" rather than a distinct first-class task. `"segm" in iou_types` is the single switch gating mask loading, retention, prediction, and metrics. The `data.task` string field is dead (read nowhere) and `data.evaluator.iou_types` duplicates the selector; both are removed or derived from `iou_types` so there is exactly one source of truth.
