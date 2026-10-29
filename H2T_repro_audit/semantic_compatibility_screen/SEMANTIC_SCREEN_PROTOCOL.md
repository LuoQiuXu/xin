# Semantic compatibility screen — evidence before module design

## Motivation and limitation
H2T random channel fusion and PI-H2T recipient-conditioned ratio may use donors with varied semantic relationships. Prior experiments **already** showed donor geometry varies substantially but safer donor selection did not consistently help final classification. Therefore neither donor affinity nor frozen margin is assumed to equal training utility.

## Only hypothesis
Within the SAME recipient, matched donors from the SAME official CIFAR-100 superclass vs DIFFERENT superclass yield different frozen-classifier mixed-feature label consistency, beyond a simple donor–recipient Euclidean distance explanation. A result is **associational**, not a causal proof and not PI-H2T final-stage performance.

## Protocol
- CIFAR-100-LT IF100, legal LT-TRAIN-only samples; favor a validation-selected Clean Stage I backbone (no official TEST checkpoint selection).
- Eligibility first: Few recipients (<20 training examples), Many donors (>100); exclude recipient fine class and same image; require both semantic groups per recipient. If availability is poor, stop and report rather than silently broadening.
- Official CIFAR100 coarse labels supply similarity proxy; **not** WordNet embedding cosine or hand-labeled subjective relations.
- Fixed cached eval features from same ResNet32 Stage-I; same frozen class weights/bias; no BN buffer update.
- Candidate pairs fixed before outcomes. For each recipient compare same-/different-coarse donors. Fixed donor mass 0.25, same for all; no donor selection for training.
- Match within recipient and discretized donor–recipient feature distance. Inspect residual distance gap and class/coarse coverage. Compare correctly-labelled margin and ref accuracy; report clean margin and clean ref correct rate. Avoid treating class weight geometry as ground truth.
- If available run native PI ratio as *separate frozen forward diagnostic* only after formula verification, not a new training method.
- Primary decision is whether a robust geometric signal even exists; requires adequate coverage and meaningful matched effect, not just a nonzero number.
- All results development-only; no official TEST access, no Stage-II updates or publication claims.

## Files
`screen.py`: runnable NumPy-only matched analysis, input specification in module docstring.
`CODEX_SEMANTIC_COMPATIBILITY_SCREEN.txt`: detailed Codex task to extract real features/checkpoints and run the analysis.

## Preregistered outcome interpretation
- Coverage failure -> feasibility NO-GO.
- No semantic signal once distance is approximately matched -> hypothesis NO-GO.
- Semantic signal exists -> geometric candidate; training utility remains unknown, next stage requires paired **unaltered** H2T Stage-II training with donor distribution stratification, not new fusion module.
- Severe residual distance mismatch, only few covered recipients, or reference clean wrong for most Few -> inconclusive.
