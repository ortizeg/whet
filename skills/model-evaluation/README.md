# Model Evaluation Skill

## Purpose

This skill teaches Claude how to evaluate CV/ML models correctly — the discipline of
measuring whether a model is good enough to ship and understanding where it fails. It
covers what generic unit testing does not: task metrics, frozen eval sets, and failure
analysis.

Detection evaluation uses **`supervision`** (`import supervision as sv`) — the
detection-native library that speaks the same `sv.Detections` object as RF-DETR, YOLO,
Ultralytics, and Transformers outputs, with metrics aligned to `pycocotools`.

## When to Use

- Detection metrics: mAP@[.50:.95], mAP@.50, per-class AP, per-size buckets
- Precision / Recall / F1 and confusion matrices
- Selecting a deployment confidence threshold on validation
- Per-slice / failure analysis to find where the model breaks
- Classification metrics (per-class F1, calibration/ECE) via torchmetrics
- Wiring evaluation into CI as a regression gate against a committed baseline

## Why This Exists

The `testing` skill checks that code runs; this skill checks that the *model* is correct
on held-out data. It also pairs with GSD's `eval-planner`: GSD designs the eval strategy,
this skill carries it out.

## Key Patterns

- Frozen, versioned test set; split by scene/video/match to avoid leakage
- `supervision.metrics.MeanAveragePrecision` with `.update(preds, targets).compute()`
- `sv.ConfusionMatrix.from_detections` at a real deployment threshold
- Operating-point selection on validation, reported on test
- Per-slice metrics + a curated hard-case regression set
- Evaluation-as-CI-gate against a committed baseline

## Important Gotchas Encoded

- The top-level `sv.MeanAveragePrecision` is **deprecated** (removed in 0.31.0) and
  inconsistent with pycocotools — import from `supervision.metrics` instead.
- `MetricTarget.MASKS` is silently ignored by mAP in released versions — use
  `pycocotools` for instance-segmentation mAP.
- Do not pre-filter predictions at a deploy threshold before mAP (use ~0.001).
- `-1` is an "absent" sentinel, not a score.
