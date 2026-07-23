# Model Evaluation

The model-evaluation skill covers how to measure CV/ML model quality correctly — task metrics, frozen eval sets, and failure analysis — the discipline that generic unit testing does not address.

**Skill directory:** `skills/model-evaluation/`

## Purpose

Unit tests check that code runs; evaluation checks whether the *model* is good enough to ship on data it has never seen, and *where* it fails. This skill teaches Claude Code to compute the right metrics (mAP, IoU, per-class F1, calibration), enforce leakage-free eval-set discipline, select deployment thresholds, run per-slice failure analysis, and gate CI on regressions. It is the implementation counterpart to GSD's `eval-planner`: GSD designs the eval strategy, this skill carries it out.

## When to Use

- Computing detection metrics: mAP@[.50:.95], mAP@.50, per-class AP, IoU by object size
- Computing classification metrics: per-class F1, confusion matrix, calibration/ECE
- Computing segmentation metrics: mean IoU, Dice
- Building precision-recall curves and selecting deployment thresholds
- Per-slice / failure analysis to find where the model breaks
- Wiring evaluation into CI as a regression gate against a committed baseline

## Key Patterns

### Detection mAP (torchmetrics / pycocotools)

```python
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, backend="pycocotools")
for images, targets in test_loader:
    metric.update(model(images), targets)
result = metric.compute()  # map, map_50, map_per_class, map_small/medium/large
```

### Operating-point selection

Select the score threshold on the **validation** set to meet a product requirement
(e.g. precision ≥ 0.9), then report the resulting recall on the frozen test set.

### Eval-as-CI gate

Compare against a committed baseline and fail the build when quality drops outside the
bootstrap confidence interval.

## Anti-Patterns

- Reporting accuracy or mean-mAP only, hiding a collapsed class or small-object failure.
- Selecting thresholds or checkpoints on the test set (leakage inflates every number).
- Splitting by frame instead of by scene/video/entity (near-duplicate leakage).
- Hand-rolling mAP/IoU instead of using torchmetrics/pycocotools.
- Shipping without a slice/failure analysis or a frozen regression set.
