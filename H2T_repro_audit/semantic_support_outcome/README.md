# Semantic head support vs observed H2T gain
Only existing H2T **final classifier** inference, not a new feature fusion method. Compares Few-class accuracy gains in pre-defined strata of availability of at least one Many class in the same CIFAR-100 superclass. A structured association screen cannot establish a causal transfer mechanism.

Run:
```
python H2T_repro_audit/semantic_support_outcome/evaluate.py --no-fusion predictions/no_fusion.npz --h2t predictions/h2t.npz --metadata metadata.json --output results/new_run
```
Prediction NPZ keys: ids, labels, logits. Metadata JSON keys: class_counts and coarse_labels, each 100 elements. Use original dataset sources. No checkpoints committed. TXT execution directive in CODEX_SEMANTIC_SUPPORT_OUTCOME.txt.

STOP if legitimate same-source checkpoints cannot be loaded, if user has no clean holdout logits, or if fine/coarse class mapping is unverifiable. Only historical fixed-BN H2T pair is targeted, not full official H2T or PI-H2T.
