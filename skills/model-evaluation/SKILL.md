---
name: model-evaluation
description: >
  Evaluate CV/ML models correctly — use when measuring detection, classification, or
  segmentation quality: computing mAP/IoU, precision-recall curves, confusion matrices,
  per-slice failure analysis, calibration, operating-point selection, and eval-as-CI
  regression gates. Covers frozen test sets and leakage-free evaluation discipline that
  generic unit testing does not. Pairs with GSD's eval-planner and with wandb/matplotlib.
---

# Model Evaluation

Measuring model quality is a different discipline from unit testing. Tests check that code
runs; evaluation checks whether the model is *good enough to ship* on data it has never
seen, and *where it fails*. This skill covers the metrics, the eval-set discipline, and
the reporting that turn "the loss went down" into a shippable decision.

## Eval-set discipline (get this right first)

A metric is only meaningful on a **frozen, held-out test set** that the model — and your
tuning decisions — never touched.

- **Three splits.** Train (fit), validation (tune/early-stop/threshold-select), test
  (report once). Never select thresholds or checkpoints on the test set.
- **Freeze the test set.** Version it (see the `dvc` skill) so numbers are comparable
  across runs. A metric that moves because the test set changed is not a metric.
- **Prevent leakage.** Split by the unit that must generalize — by scene/video/patient,
  not by frame — so near-duplicate frames don't span train and test. For detection,
  dedupe near-identical crops before splitting.
- **Report with uncertainty.** A single number hides noise; bootstrap the test set for a
  confidence interval, especially with < ~2k examples.

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Frozen, validated evaluation settings."""

    test_manifest: str = Field(..., description="Versioned path to the frozen test set")
    iou_threshold: float = 0.5
    score_threshold: float = 0.05  # keep low for mAP; raise only for deployment metrics
    per_class: bool = True
    bootstrap_samples: int = 1000
```

## Detection metrics (mAP / IoU)

Use `torchmetrics.detection.MeanAveragePrecision` (wraps pycocotools semantics) — do not
hand-roll mAP. Report **mAP@[.50:.95]** (COCO primary), **mAP@.50**, and **per-class AP**;
the mean hides a collapsed class.

```python
import torch
from loguru import logger
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(
    iou_type="bbox",
    class_metrics=True,          # per-class AP, not just the mean
    backend="pycocotools",
)

for images, targets in test_loader:
    preds = model(images)  # list[dict(boxes[N,4] xyxy, scores[N], labels[N])]
    metric.update(preds, targets)  # targets: list[dict(boxes, labels)]

result = metric.compute()
logger.info("mAP@[.50:.95]={:.4f}  mAP@.50={:.4f}", result["map"], result["map_50"])
for cls_id, ap in zip(result["classes"].tolist(), result["map_per_class"].tolist()):
    logger.info("class {}: AP={:.4f}", cls_id, ap)
```

Report metrics by object size (`map_small` / `map_medium` / `map_large`) — small-object AP
is where detectors usually fail and where a single mean is most misleading.

## Classification metrics

Accuracy alone is a trap under class imbalance. Report macro/weighted precision, recall,
F1, per-class recall, and a confusion matrix.

```python
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassCalibrationError,
)

f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average=None)      # per-class
cm = MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
ece = MulticlassCalibrationError(num_classes=NUM_CLASSES, n_bins=15)  # is P(class) trustworthy?

for logits, y in test_loader:
    probs = logits.softmax(dim=1)
    f1.update(probs, y); cm.update(probs, y); ece.update(probs, y)

logger.info("per-class F1: {}", f1.compute().tolist())
logger.info("ECE: {:.4f}", ece.compute().item())
```

## Segmentation

Report **mean IoU (Jaccard)** and **Dice**, per class. `torchmetrics.JaccardIndex` and
`Dice` cover both; watch the ignore-index for void/unlabeled pixels.

## Operating-point selection (thresholds are a decision, not a default)

mAP integrates over all thresholds, but deployment runs at one. Pick the score threshold
on the **validation** set to hit a product requirement (e.g. "precision ≥ 0.9"), then
report the resulting recall on test.

```python
from torchmetrics.classification import BinaryPrecisionRecallCurve

prc = BinaryPrecisionRecallCurve()
prc.update(val_scores, val_labels)
precision, recall, thresholds = prc.compute()

# lowest threshold that still meets the precision floor -> best recall at that precision
mask = precision[:-1] >= 0.90
chosen = thresholds[mask][recall[:-1][mask].argmax()] if mask.any() else thresholds[-1]
logger.info("deploy threshold={:.3f}", chosen.item())
```

Save the PR curve as an artifact (see the `matplotlib` skill) — reviewers read the curve,
not just the number.

## Slice / failure analysis (the highest-value step)

Aggregate metrics hide the failures that matter. Compute the same metric across meaningful
slices and surface the worst ones.

```python
from collections import defaultdict

def evaluate_by_slice(preds, targets, slice_fn) -> dict[str, float]:
    """slice_fn(target) -> slice key, e.g. lighting, camera, object-size bucket."""
    buckets: dict[str, MeanAveragePrecision] = defaultdict(
        lambda: MeanAveragePrecision(backend="pycocotools")
    )
    for p, t in zip(preds, targets):
        buckets[slice_fn(t)].update([p], [t])
    scores = {k: m.compute()["map"].item() for k, m in buckets.items()}
    worst = min(scores, key=scores.get)
    logger.warning("worst slice: {} (mAP={:.3f})", worst, scores[worst])
    return scores
```

Curate a small **hard-case / regression set** of known-failure examples and track it
separately — it catches regressions the aggregate metric averages away.

## Evaluation as a CI gate

Make "did quality regress?" a mechanical check, not a vibe. Compare against a committed
baseline and fail the build on a real drop (outside the bootstrap CI).

```python
import json, sys

def gate(current: dict, baseline_path: str, tolerance: float = 0.005) -> int:
    baseline = json.load(open(baseline_path))
    drop = baseline["map"] - current["map"]
    if drop > tolerance:
        logger.error("mAP regressed by {:.4f} (> {:.4f})", drop, tolerance)
        return 1
    logger.info("eval gate passed (ΔmAP={:+.4f})", -drop)
    return 0

sys.exit(gate(result, "eval/baseline.json"))
```

Log every eval run (metrics, chosen threshold, per-slice table, PR curve) to your tracker
via the `wandb` skill so results are comparable and reviewable over time.

## Anti-patterns

- Reporting accuracy/mean-mAP only, hiding a collapsed class or small-object failure.
- Selecting thresholds or checkpoints on the test set (leakage — inflates every number).
- Splitting by frame instead of by scene/video/entity (near-duplicate leakage).
- Hand-rolling mAP/IoU instead of using torchmetrics/pycocotools (subtle, wrong).
- Shipping without a slice/failure analysis or a frozen regression set.
- A single point estimate with no uncertainty on a small test set.
