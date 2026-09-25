# Q2 baseline review and optimization profile

## Current revision

2026-09-25 correction: final performance is reported on test, not valid.
The first locked attention checkpoint obtained test clean Macro-F1 0.598213
and missing-grid Macro-F1 0.560844. The later bias-calibrated 0.602589/0.563362
was produced after observing those test scores and is exploratory, not an
independent holdout confirmation. Do not claim calibration was locked before
the first test. Future train/valid exploration must disclose this test history.
See section 8 of the method document and section 7 of the implementation
document for updated experiments, corrected seed aggregation, shared
evaluation/export semantics, and the controlled embedding-unfreezing run.

The active method and implementation updates are
`docs/E题_Q2_优化方案_v3_实验中.md` and
`docs/E题_Q2_优化实现说明_v2_实验中.md`. The frozen `late_balanced`
profile below is historical exploration, not a selected final method.
Fine-tuning initially used stale frozen validation features; those scores are
invalid. Reevaluating the saved epoch-15 tuned checkpoint with its own BERT
weights gives missing Macro-F1 0.54049 and clean Macro-F1 0.58243. No test-set
claim follows from these validation results. The target of at least 0.60
missing-grid Macro-F1 remains unmet.

## What the baseline actually measured

`runs_baseline_20260924` contains nine complete runs: `full`, `late_clean`, and
`late_aug`, each with seeds 1111–1113. Their files contain validation results
only. The baseline suite stopped at `no_msd/seed_1111` epoch 5, so no baseline
test report exists and the value around 0.49 is not a test-set F1.

The selected validation score is the equal-weight mean Macro-F1 over the 72
non-stress local-missing scenarios. For the complete runs, `full` is 0.476–
0.498, `late_clean` is 0.498–0.514, and `late_aug` is 0.501–0.519. Clean
validation Macro-F1 is 0.508–0.543. The best clean validation accuracy among
these runs is 0.593. Neutral F1 is the weakest class in the inspected
confusion matrices. Thus the observed gap is a validation/protocol issue first,
not evidence that the model has an implausibly high test score.

The project documentation does not define 0.66 as an official target. A value
quoted elsewhere must be labeled by split, task (three-class or binary), and
metric (Accuracy, weighted-F1, or Macro-F1) before comparison. This Q2 profile
uses three-class Macro-F1 and reports clean and missing conditions separately.

## Optimization decision

The first optimization profile is `late_balanced`. It keeps the fixed aligned
input, frozen BERT, train-only normalization, 60 epochs, FP32 computation, and
the same validation grid. It uses the empirically stronger clean late-fusion
backbone and applies a mild square-root inverse class-prior weight to the
classification cross-entropy. The regression head remains unchanged. The
weights are computed from train labels only, so valid/test labels cannot affect
training.

`full`, `late_clean`, and `late_aug` remain unchanged as historical controls.
`late_balanced` is an exploratory profile and must be reported as a new
variant; it must not replace the fixed ten-variant ablation table silently.
The primary selection rule remains missing-grid Macro-F1, followed by MAE and
clean metrics. If the new profile does not improve both the missing-grid and
clean validation trade-off across seeds, retain `late_clean` as the simpler
deployment candidate.

## Implementation changes

- `Student` accepts `late_balanced`; its architecture is clean late fusion.
- The trainer applies no corruption to `late_balanced`, matching
  `late_clean`, and computes class weights from the train-only class prior.
- `task_loss` accepts an optional classification weight vector; the default is
  `None`, so all historical variants preserve their previous objective.
- State inference is vectorized, diagnostic attention output is suppressed
  during training, and RNG restoration uses CPU ByteTensors. Attention backend
  changes may alter floating-point results and random paths; training is not
  claimed to be bitwise equivalent.

The user has since authorized stopping the original suite. Run each new
profile in a separate named directory and retain its outputs. Do not use new
results to rewrite the completed historical table.
