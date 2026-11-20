# CC-MIA Logit / Geometry Diagnostics

Offline-only diagnostics for the completed CC-MIA causal-oracle experiment. No model training or optimizer updates are allowed.

Read LOGIT_GEOMETRY_PROTOCOL.md first, then follow CODEX_LOGIT_GEOMETRY_20261120.txt.

The implementation reuses the parent experiment's sealed Calibration/Evaluation reader and verifies all checkpoint/logit artifacts before producing prior-adjustment curves, Many/Few Pareto comparisons, radial/tangential decomposition of cumulative momentum-propagated source-conditioned parameter displacements, separate bias attribution, and target-row/fixed-margin projections.

Evaluation is TRAIN-derived development data, not an independent test set. Official CIFAR-100 TEST access is prohibited.
