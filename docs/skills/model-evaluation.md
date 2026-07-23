# Model Evaluation

The model-evaluation skill covers how to measure CV/ML model quality correctly — task metrics, frozen eval sets, and failure analysis — the discipline that generic unit testing does not address.

**Skill directory:** `skills/model-evaluation/`

## Purpose

Unit tests check that code runs; evaluation checks whether the *model* is good enough to ship on data it has never seen, and *where* it fails. Detection evaluation is built on **`supervision`** (`import supervision as sv`), the detection-native library that speaks the same `sv.Detections` object as RF-DETR, YOLO, Ultralytics, and Transformers outputs, with metrics aligned to `pycocotools`. It is the implementation counterpart to GSD's `eval-planner`.

## When to Use

- Detection metrics: mAP@[.50:.95], mAP@.50, per-class AP, per-size buckets
- Precision / Recall / F1 and confusion matrices
- Selecting a deployment confidence threshold on validation
- Per-slice / failure analysis to find where the model breaks
- Classification metrics (per-class F1, calibration/ECE) via torchmetrics
- Wiring evaluation into CI as a regression gate

## Key Patterns

### Detection mAP with supervision

```python
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget

dataset = sv.DetectionDataset.from_coco(
    images_directory_path="data/test/images",
    annotations_path="data/test/_annotations.coco.json",
)

predictions, targets = [], []
for _path, image, target in dataset:
    predictions.append(predict(image))   # -> sv.Detections
    targets.append(target)

result = MeanAveragePrecision(metric_target=MetricTarget.BOXES).update(
    predictions, targets
).compute()
print(result.map50_95, result.map50)
```

### Confusion matrix (at a real deployment threshold)

```python
cm = sv.ConfusionMatrix.from_detections(
    predictions=predictions, targets=targets, classes=dataset.classes,
    conf_threshold=0.30, iou_threshold=0.50,
)
cm.plot(normalize=True, save_path="artifacts/confusion_matrix.png")
```

### Eval-as-CI gate

Compare against a committed baseline and fail the build when quality drops outside the bootstrap confidence interval.

## Gotchas

- The top-level `sv.MeanAveragePrecision` is **deprecated** (removed in 0.31.0) and inconsistent with pycocotools — import from `supervision.metrics`.
- `MetricTarget.MASKS` is silently ignored by mAP in released versions; use `pycocotools` for instance-segmentation mAP.
- Do not pre-filter predictions at a deploy threshold before mAP — it needs the full score-ranked list (filter at ~0.001).
- `-1` is an "absent" sentinel, not a score.
- `len(predictions) != len(targets)` raises — append `sv.Detections.empty()` for empty images.

## Anti-Patterns

- Reporting mean mAP only, hiding a collapsed class or small-object failure
- Selecting thresholds or checkpoints on the test set (leakage)
- Splitting by frame instead of by scene/video/match (near-duplicate leakage)
- Hand-rolling mAP/IoU instead of using supervision or pycocotools
- Shipping without a slice/failure analysis or a frozen regression set
