# Model Evaluation Skill

## Purpose

This skill teaches Claude how to evaluate CV/ML models correctly — the discipline of
measuring whether a model is good enough to ship and understanding where it fails. It
covers what generic unit testing does not: task metrics, frozen eval sets, and failure
analysis.

## When to Use

Use this skill whenever you are measuring model quality:

- Computing detection metrics (mAP@[.50:.95], mAP@.50, per-class AP, IoU by object size)
- Computing classification metrics (per-class F1, confusion matrix, calibration/ECE)
- Computing segmentation metrics (mean IoU, Dice)
- Building precision-recall curves and selecting deployment thresholds
- Running per-slice / failure analysis to find where the model breaks
- Wiring evaluation into CI as a regression gate against a baseline

## Why This Exists

The `testing` skill checks that code runs; this skill checks that the *model* is correct
on held-out data. It also pairs directly with GSD's `eval-planner` — GSD designs the eval
strategy, and this skill is the implementation guidance for carrying it out.

## Key Patterns

- Frozen, versioned test set with leakage-free, split-by-entity discipline
- torchmetrics / pycocotools for mAP, IoU, F1, calibration — never hand-rolled
- Operating-point selection on validation, reported on test
- Per-slice metrics + a curated hard-case regression set
- Evaluation-as-CI-gate comparing against a committed baseline
