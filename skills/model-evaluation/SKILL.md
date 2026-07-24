---
name: model-evaluation
description: >
  Use this skill whenever measuring whether a CV/ML model is good enough to ship —
  computing detection mAP/IoU with supervision, confusion matrices, per-class and
  per-size breakdowns, precision-recall curves, calibration, deployment-threshold
  selection, per-slice failure analysis, or an eval-as-CI regression gate. Reach for it
  any time you would otherwise eyeball predictions or report a single accuracy number,
  even if the user only says "how good is the model?". Covers frozen test sets and
  leakage-free eval discipline that generic unit testing does not. Not for testing code
  correctness (see testing).
---

# Model Evaluation

Measuring model quality is a different discipline from unit testing. Tests check that code
runs; evaluation checks whether the *model* is good enough to ship on data it has never
seen, and *where* it fails.

For detection work, use **`supervision`** (`import supervision as sv`) — it is the
detection-native evaluation library, speaks the same `sv.Detections` object as RF-DETR /
YOLO / Ultralytics / Transformers outputs, and its metrics are aligned with
`pycocotools` (since supervision 0.26.0).

## Eval-set discipline (get this right first)

A metric is only meaningful on a **frozen, held-out test set** that the model — and your
tuning decisions — never touched.

- **Three splits.** Train (fit), validation (tune/early-stop/threshold-select), test
  (report once). Never select thresholds or checkpoints on the test set.
- **Freeze the test set.** Keep it immutable and content-addressed so numbers are
  comparable across runs. A metric that moves because the test set changed is not a metric.
- **Prevent leakage.** Split by the unit that must generalize — by scene/video/match, not
  by frame — so near-duplicate frames don't span train and test.
- **Report with uncertainty.** Bootstrap the test set for a confidence interval,
  especially under ~2k examples.

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Frozen, validated evaluation settings."""

    test_manifest: str = Field(..., description="Versioned path to the frozen test set")
    map_conf_threshold: float = 0.001  # keep near-zero for mAP (see gotchas)
    deploy_conf_threshold: float = 0.30  # chosen on val, used for confusion matrix
    iou_threshold: float = 0.50
    bootstrap_samples: int = 1000
```

## Detection metrics with supervision

Use the **`supervision.metrics`** module. Note: the top-level `sv.MeanAveragePrecision`
is **deprecated** (removed in 0.31.0) and produces numbers inconsistent with pycocotools —
always import from `supervision.metrics`.

```python
import numpy as np
import supervision as sv
from loguru import logger
from supervision.metrics import MeanAveragePrecision, MetricTarget

dataset = sv.DetectionDataset.from_coco(
    images_directory_path="data/test/images",
    annotations_path="data/test/_annotations.coco.json",
)

predictions: list[sv.Detections] = []
targets: list[sv.Detections] = []

# Iteration yields (image_path, image HWC uint8 BGR, annotations)
for _path, image, target in dataset:
    pred = predict(image)  # -> sv.Detections
    # Keep the FULL score-ranked list for mAP; do not filter at a deploy threshold.
    if pred.confidence is not None:
        pred = pred[pred.confidence >= 0.001]
    predictions.append(pred)  # sv.Detections.empty() if the model found nothing
    targets.append(target)

result = MeanAveragePrecision(metric_target=MetricTarget.BOXES).update(
    predictions, targets
).compute()

print(result)  # pycocotools-style 6-line summary
logger.info("mAP@50:95={:.4f} mAP@50={:.4f}", result.map50_95, result.map50)
```

Building `sv.Detections` — either from a model adapter or raw arrays:

```python
# From common model outputs
det = sv.Detections.from_ultralytics(model(image)[0])
det = sv.Detections.from_inference(model.infer(image)[0])
det = sv.Detections.from_transformers(outputs)

# Or directly (xyxy is ABSOLUTE pixels, not normalized)
det = sv.Detections(
    xyxy=np.array([[10, 10, 110, 110]], dtype=np.float32),
    confidence=np.array([0.92], dtype=np.float32),
    class_id=np.array([0], dtype=int),
)
```

### Per-class and per-size breakdowns

The mean hides a collapsed class and small-object failure. Always print both.
`-1` is a **sentinel for "absent"**, not a score — filter it before averaging.

```python
for class_id, ap_row in zip(result.matched_classes, result.ap_per_class):
    valid = ap_row[ap_row > -1]
    ap = float(valid.mean()) if valid.size else float("nan")
    logger.info("{:>14s} AP@50:95={:.4f} AP@50={:.4f}",
                dataset.classes[int(class_id)], ap, ap_row[0])

for name, bucket in (("small", result.small_objects),
                     ("medium", result.medium_objects),
                     ("large", result.large_objects)):
    if bucket is not None and bucket.map50_95 > -1:
        logger.info("{:>6}: mAP@50:95={:.4f}", name, bucket.map50_95)
```

### Precision / Recall / F1

```python
from supervision.metrics import AveragingMethod, F1Score, Precision, Recall

f1 = F1Score(averaging_method=AveragingMethod.MACRO).update(predictions, targets).compute()
p = Precision().update(predictions, targets).compute()
r = Recall().update(predictions, targets).compute()
logger.info("F1@50={:.4f} P@50={:.4f} R@50={:.4f}",
            f1.f1_50, p.precision_at_50, r.recall_at_50)
```

### Confusion matrix

Unlike mAP, the confusion matrix wants a **real deployment threshold**. The matrix is
`(C+1, C+1)` — the extra row/column is background (missed detections / false positives).

```python
cm = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=dataset.classes,
    conf_threshold=0.30,
    iou_threshold=0.50,
)
fig = cm.plot(normalize=True, save_path="artifacts/confusion_matrix.png")
```

## Operating-point selection (thresholds are a decision)

mAP integrates over all thresholds, but deployment runs at one. Sweep the threshold on the
**validation** set to hit a product requirement, then report the result on test.

```python
def pick_threshold(preds, tgts, min_precision: float = 0.90) -> float:
    """Lowest threshold meeting a precision floor -> best recall at that precision."""
    best_t, best_recall = 1.0, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        filtered = [p[p.confidence >= t] if p.confidence is not None else p for p in preds]
        pr = Precision().update(filtered, tgts).compute()
        rc = Recall().update(filtered, tgts).compute()
        if pr.precision_at_50 >= min_precision and rc.recall_at_50 > best_recall:
            best_t, best_recall = float(t), rc.recall_at_50
    logger.info("deploy threshold={:.2f} (recall={:.4f})", best_t, best_recall)
    return best_t
```

## Slice / failure analysis (the highest-value step)

Aggregate metrics hide the failures that matter. Score the same metric across meaningful
slices (camera, lighting, occlusion, jersey color) and surface the worst.

```python
from collections import defaultdict


def evaluate_by_slice(preds, tgts, slice_of) -> dict[str, float]:
    """slice_of(index) -> slice key for that sample."""
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for i, (p, t) in enumerate(zip(preds, tgts)):
        pb, tb = buckets[slice_of(i)]
        pb.append(p)
        tb.append(t)

    scores = {
        key: MeanAveragePrecision().update(pb, tb).compute().map50_95
        for key, (pb, tb) in buckets.items()
    }
    worst = min(scores, key=scores.get)
    logger.warning("worst slice: {} (mAP={:.3f})", worst, scores[worst])
    return scores
```

Curate a small **hard-case regression set** of known failures and track it separately — it
catches regressions the aggregate averages away.

## Classification and segmentation

`supervision` is detection-focused. For classification metrics use `torchmetrics`:

```python
from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score

f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average=None)  # per-class
ece = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15)  # are probs trustworthy?
```

For **instance-segmentation mAP**, do not rely on `MetricTarget.MASKS` — see gotchas.
Use `pycocotools` (`COCOeval(..., iouType="segm")`) or `torchmetrics` with
`iou_type="segm"`.

## Evaluation as a CI gate

Make "did quality regress?" mechanical. Compare against a committed baseline and fail
outside the bootstrap CI.

```python
import json
import sys


def gate(current_map: float, baseline_path: str, tolerance: float = 0.005) -> int:
    baseline = json.load(open(baseline_path))
    drop = baseline["map50_95"] - current_map
    if drop > tolerance:
        logger.error("mAP regressed by {:.4f} (> {:.4f})", drop, tolerance)
        return 1
    logger.info("eval gate passed (delta={:+.4f})", -drop)
    return 0


sys.exit(gate(result.map50_95, "eval/baseline.json"))
```

Log every eval run (metrics, chosen threshold, per-slice table, confusion matrix) to your
tracker via the `wandb` skill so results stay comparable over time. In headless CI set
`MPLBACKEND=Agg` — result `.plot()` calls `plt.show()`.

## Gotchas (supervision-specific, verified against 0.29.1)

- **`MetricTarget.MASKS` is silently ignored by `MeanAveragePrecision`.** In released
  versions it always scores boxes, so segmentation mAP looks identical to box mAP and can
  read 1.000 for a badly wrong mask. Use `pycocotools` for mask mAP.
  (`Precision`/`Recall`/`F1Score` *do* honor masks correctly.)
- **Don't pre-filter confidence before mAP.** mAP needs the full score-ranked list; filter
  at ~0.001. If `confidence is None` every prediction silently scores 0.0 and ranking is
  meaningless — always set confidence on predictions.
- **`-1` means "absent", not zero.** `map50_95` masks sentinels out, but `map50`, `map75`,
  and `ap_per_class` rows do not.
- **`len(predictions) != len(targets)` raises.** Append `sv.Detections.empty()` for images
  with no detections rather than skipping them.
- **`class_id` is used directly as the COCO category id** — predictions and targets must
  share one ID space. Use `class_mapping={...}` to remap (it applies to *both* sides).
- **`xyxy` is absolute pixels**, `x2 > x1`, `y2 > y1`. Never normalized coords.
- **Size buckets use box area** unless you pass `data["area"]`; released versions ignore
  COCO `iscrowd`, so numbers can drift from pycocotools on COCO val specifically.
- **`to_pandas()` needs the extra**: install `supervision[metrics]`.

## Anti-patterns

- Reporting mean mAP only, hiding a collapsed class or small-object failure.
- Selecting thresholds or checkpoints on the test set (leakage inflates every number).
- Splitting by frame instead of by scene/video/match (near-duplicate leakage).
- Using the deprecated top-level `sv.MeanAveragePrecision` — inconsistent with pycocotools.
- Hand-rolling mAP/IoU instead of using supervision or pycocotools.
- Shipping without a slice/failure analysis or a frozen regression set.
