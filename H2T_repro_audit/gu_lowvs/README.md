# GU-LowVS V1 | First-order training-utility guided enhancement

**This is a method-design iteration, not another VS statistics probe.**

Previous clean, paired CIFAR100-LT Stage-II controls: fixed Low-VS gave +0.278pp Overall and +0.948pp Tail against Uniform, at a Head cost. RAVS-V1 candidate-pool/hardness controller did not improve over fixed Low-VS.

**Key distinction:** Keep the exact Bottom48-to-random16 *Low-VS channel identities*. Change what happens **after** the mask is selected. An example's pre-fusion confidence is not a reliable measure of augmentation training utility.

## Method

On each selected complete 8x8 feature channel, define a fusion strength alpha in {0, 0.5, 1}: mixed = receiver + alpha * (LowVS_full_swap - receiver). Other channels remain the receiver.

A fixed Clean Stage-I ResNet32 and BN produce features. Only the Stage-II 100-class linear classifier is trained (CE, original BS recipients/IS donors; 10 epochs, batch128). The method has zero new learned parameters.

Using original **TRAIN-only** images under eval transform, compute a small frozen [10847,64] clean GAP feature bank once. At the beginning of each Stage-II epoch, evaluate the analytic gradient G_ref of an equally weighted class-wise clean TRAIN CE at the current classifier W,b:

L_ref = (1/100) sum_c [(1/n_c) sum_{i:y_i=c} CE(W f_i+b,c)].

For each current augmented receiver/donor fusion, compute exact per-sample linear classifier CE gradient at alpha0, alpha.5 and alpha1, WITHOUT auto-grad through the gate. Score each action by the Euclidean dot product <G_ref, g_alpha>, and choose the argmax. The one-step first-order clean-reference-risk approximation is

L_ref(theta - eta*g_alpha) ~= L_ref(theta) - eta * <G_ref,g_alpha>.

The score does NOT include momentum or weight decay, cannot identify unseen-data utility, and might overfit clean TRAIN risk (reference samples also participate in stage-II training). It is deliberately a falsifiable train-only proxy.

## Minimal matched experiment (Stage-II seeds124/125)

| Arm | Role |
|---|---|
| LowVS-Full | Same run-engine, alpha=1 baseline |
| LowVS-Half | Simpler constant alpha=.5 intensity control |
| GU-LowVS | Per-sample action chosen by train-only first-order score |
| GU-Shuffled | Same computed alpha histogram within each minibatch, permuted across receivers (control for utility-based assignment); independently trained classifier paths may create slightly different cross-arm histograms |

8 new classifier runs only; Stage I remains the clean epoch184 seed42 checkpoint. Each run inherits the historical donor, augmentation and Gumbel plan. New paired baselines are retrained in this same environment; each later arm compares all 850 batches to LowVS-Full for matching samples, images, frozen features, VS scores and RNG. The reference bank is built from **original TRAIN only**, never from the 1,980-image development holdout or official TEST.

Main evaluation: fixed epoch10 using 99 observed classes from the original TRAIN holdout (20/class, class0 absent) while retaining 100 logits. Best epoch is a sensitivity check. Two seeds under one frozen Stage I and reuse of the development holdout do NOT establish independent generalization.

## Run

From the user's existing local repository root:
1. python H2T_repro_audit/gu_lowvs/test_utility_policy.py
2. python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py preflight
3. python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py prepare
4. Run the 8 ordered training commands in CODEX_GU_LOWVS_V1.txt (LowVS-Full first for each seed).
5. python H2T_repro_audit/gu_lowvs/run_gu_lowvs.py report

All new outputs in H2T_repro_audit/gu_lowvs/results/. The user's Codex instructions forbid all Git/version control activities, which are to be handled exclusively by the user.

**Research decision:** improvement over LowVS-Full alone is insufficient; GU must additionally outperform GU-Shuffled, not simply Half, and must not collapse to alpha0 (no fusion) or alpha1 (fixed Low-VS). If these controls fail, stop V1, not sweep alpha values or reference-gradient temperatures. Any method paper would require comparison to published alternatives, independent Stage I seeds and reserved testing after method selection.
