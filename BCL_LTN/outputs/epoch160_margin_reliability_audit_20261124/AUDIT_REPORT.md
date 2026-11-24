# Epoch-160 Margin Reliability Audit

## Scope and leakage controls
- Frozen BCL-LTN epoch-160-pre-decay checkpoint; inference only, no optimizer or BN updates.
- Fixed CIFAR-100-LT IF=100 TRAIN subset only; official TEST is not instantiated or read.
- The model receives images only. Observed labels are used after inference to calculate margins.
- Clean labels and the corruption mask are used only after prediction, for offline metric evaluation.
- Group membership is determined by the OBSERVED LABEL's class-frequency group, not the original source class.

## Score definitions
- raw_margin: z[y_obs] - max(z[k], k != y_obs).
- lc_plus_log_prior_margin: the exact training-loss adjustment z + log(pi), then the observed-label margin.
- inverse_prior_margin: z - log(pi), an alternative scoring hypothesis, not a training change.
- class_normalized_raw_margin: (raw_margin - observed-class mean) / (observed-class standard deviation + 1e-8).
- pi is computed from the observed noisy-label class counts used by LC/BCL.
- Class normalization uses all scores in each observed class and never reads clean/noisy status.

## Observed-label group composition

| Group | Samples | Clean | Noisy | Clean prevalence |
|---|---:|---:|---:|---:|
| Overall | 10847 | 6468 | 4379 | 59.63% |
| Many | 6752 | 5237 | 1515 | 77.56% |
| Medium | 2559 | 1037 | 1522 | 40.52% |
| Few | 1536 | 194 | 1342 | 12.63% |

## Pooled ranking metrics
Positive class = unflipped/clean observed label; negative class = flipped/noisy observed label.
ROC-AUC is rank-based. average_precision is step-wise AP; pr_auc_trapezoid is trapezoidal PR area.
Confidence intervals are 95% stratified bootstrap intervals preserving the clean/noisy counts.

| Group | Score | ROC-AUC [95% CI] | AP [95% CI] | PR-AUC trapezoid |
|---|---|---:|---:|---:|
| Overall | raw_margin | 0.8738 [0.8677835658127147, 0.8804611788455132] | 0.9190 [0.9146030211209846, 0.9240062472179028] | 0.9190 |
| Overall | lc_plus_log_prior_margin | 0.9098 [0.9052465628033272, 0.9155940260220429] | 0.9426 [0.9394985599080763, 0.9467630405821915] | 0.9425 |
| Overall | inverse_prior_margin | 0.7832 [0.7746088345342497, 0.7912837417451566] | 0.8527 [0.8456529402787486, 0.8586847311613945] | 0.8527 |
| Overall | class_normalized_raw_margin | 0.7805 [0.7721083757258846, 0.7885763487129993] | 0.8120 [0.8029520029370472, 0.8202363868438052] | 0.8120 |
| Many | raw_margin | 0.9011 [0.8932424497939578, 0.9082445514935301] | 0.9671 [0.9632669518330175, 0.9705389580314034] | 0.9671 |
| Many | lc_plus_log_prior_margin | 0.9106 [0.9009026059184112, 0.9181266010885984] | 0.9710 [0.9671302758970342, 0.9739855689115465] | 0.9710 |
| Many | inverse_prior_margin | 0.8802 [0.869793173226049, 0.8890124721847782] | 0.9586 [0.9539744217820118, 0.9622920774595932] | 0.9586 |
| Many | class_normalized_raw_margin | 0.8981 [0.8896769337999295, 0.9069838657533884] | 0.9624 [0.956485131553503, 0.9674729361668509] | 0.9623 |
| Medium | raw_margin | 0.8581 [0.8423946375689502, 0.8731166295173204] | 0.8353 [0.817820177159696, 0.8522737862275964] | 0.8353 |
| Medium | lc_plus_log_prior_margin | 0.8622 [0.8469005692149978, 0.8769475845744256] | 0.8382 [0.8206654107604672, 0.8553608058817547] | 0.8381 |
| Medium | inverse_prior_margin | 0.8404 [0.8242178520877341, 0.8570740834840215] | 0.8136 [0.7951298292391545, 0.8326491284383761] | 0.8135 |
| Medium | class_normalized_raw_margin | 0.8488 [0.8347519568349517, 0.8631858267746468] | 0.8055 [0.7866760771579633, 0.8238787375564345] | 0.8053 |
| Few | raw_margin | 0.6767 [0.63109578333615, 0.7167389417241538] | 0.2612 [0.21934182387725093, 0.3170224593921766] | 0.2576 |
| Few | lc_plus_log_prior_margin | 0.6822 [0.6425716732988155, 0.7189956327684484] | 0.2826 [0.23239726526844504, 0.34788887576993005] | 0.2783 |
| Few | inverse_prior_margin | 0.6455 [0.6038900817367524, 0.6858770760674174] | 0.2311 [0.19035629666371487, 0.28633319381761885] | 0.2276 |
| Few | class_normalized_raw_margin | 0.6830 [0.6439932321354495, 0.7213927128305191] | 0.2397 [0.203371410922396, 0.2943870995065856] | 0.2368 |

## Macro per-observed-class ranking metrics
Each observed class is evaluated separately, then metrics are macro-averaged within its frequency group.

| Group | Score | Macro ROC-AUC | Macro AP | Macro PR-AUC trapezoid | Classes evaluated |
|---|---|---:|---:|---:|---:|
| Overall | raw_margin | 0.8156 | 0.7097 | 0.6956 | 100 |
| Overall | lc_plus_log_prior_margin | 0.8142 | 0.7077 | 0.6934 | 100 |
| Overall | inverse_prior_margin | 0.8041 | 0.6976 | 0.6847 | 100 |
| Overall | class_normalized_raw_margin | 0.8156 | 0.7097 | 0.6956 | 100 |
| Many | raw_margin | 0.9046 | 0.9560 | 0.9555 | 35 |
| Many | lc_plus_log_prior_margin | 0.9040 | 0.9553 | 0.9549 | 35 |
| Many | inverse_prior_margin | 0.9050 | 0.9555 | 0.9550 | 35 |
| Many | class_normalized_raw_margin | 0.9046 | 0.9560 | 0.9555 | 35 |
| Medium | raw_margin | 0.8523 | 0.7986 | 0.7917 | 35 |
| Medium | lc_plus_log_prior_margin | 0.8498 | 0.7883 | 0.7810 | 35 |
| Medium | inverse_prior_margin | 0.8444 | 0.7946 | 0.7892 | 35 |
| Medium | class_normalized_raw_margin | 0.8523 | 0.7986 | 0.7917 | 35 |
| Few | raw_margin | 0.6691 | 0.3188 | 0.2801 | 30 |
| Few | lc_plus_log_prior_margin | 0.6678 | 0.3247 | 0.2860 | 30 |
| Few | inverse_prior_margin | 0.6396 | 0.2834 | 0.2473 | 30 |
| Few | class_normalized_raw_margin | 0.6691 | 0.3188 | 0.2801 | 30 |

## Threshold operating points
Thresholds use score > threshold at zero and at quantiles of the unlabeled score distribution; no clean/noisy labels select a threshold.

| Group | Score | Rule | Threshold | Precision(clean) | Recall(clean) | Noisy acceptance |
|---|---|---|---:|---:|---:|---:|
| Overall | raw_margin | zero | 0.00000 | 0.9657 | 0.5264 | 0.0276 |
| Overall | raw_margin | q10 | -4.09988 | 0.6491 | 0.9797 | 0.7821 |
| Overall | raw_margin | q25 | -2.71387 | 0.7310 | 0.9194 | 0.4997 |
| Overall | raw_margin | q50 | -1.19901 | 0.8835 | 0.7407 | 0.1443 |
| Overall | raw_margin | q75 | 0.59778 | 0.9786 | 0.4103 | 0.0132 |
| Overall | raw_margin | q90 | 2.07590 | 0.9935 | 0.1667 | 0.0016 |
| Overall | lc_plus_log_prior_margin | zero | 0.00000 | 0.9770 | 0.5716 | 0.0199 |
| Overall | lc_plus_log_prior_margin | q10 | -4.70259 | 0.6535 | 0.9862 | 0.7726 |
| Overall | lc_plus_log_prior_margin | q25 | -3.30369 | 0.7481 | 0.9409 | 0.4679 |
| Overall | lc_plus_log_prior_margin | q50 | -1.39099 | 0.9213 | 0.7724 | 0.0975 |
| Overall | lc_plus_log_prior_margin | q75 | 0.96761 | 0.9878 | 0.4142 | 0.0075 |
| Overall | lc_plus_log_prior_margin | q90 | 2.86580 | 0.9945 | 0.1668 | 0.0014 |
| Overall | inverse_prior_margin | zero | 0.00000 | 0.9291 | 0.3624 | 0.0409 |
| Overall | inverse_prior_margin | q10 | -4.17073 | 0.6399 | 0.9658 | 0.8027 |
| Overall | inverse_prior_margin | q25 | -2.78422 | 0.6934 | 0.8721 | 0.5695 |
| Overall | inverse_prior_margin | q50 | -1.45407 | 0.7962 | 0.6676 | 0.2523 |
| Overall | inverse_prior_margin | q75 | -0.11234 | 0.9192 | 0.3854 | 0.0500 |
| Overall | inverse_prior_margin | q90 | 1.18329 | 0.9806 | 0.1645 | 0.0048 |
| Overall | class_normalized_raw_margin | zero | 0.00000 | 0.7881 | 0.7219 | 0.2866 |
| Overall | class_normalized_raw_margin | q10 | -1.32333 | 0.6482 | 0.9784 | 0.7842 |
| Overall | class_normalized_raw_margin | q25 | -0.61574 | 0.7142 | 0.8983 | 0.5309 |
| Overall | class_normalized_raw_margin | q50 | 0.11715 | 0.8018 | 0.6722 | 0.2455 |
| Overall | class_normalized_raw_margin | q75 | 0.70783 | 0.8462 | 0.3548 | 0.0952 |
| Overall | class_normalized_raw_margin | q90 | 1.15299 | 0.8636 | 0.1449 | 0.0338 |
| Many | raw_margin | zero | 0.00000 | 0.9813 | 0.5601 | 0.0370 |
| Many | raw_margin | q10 | -3.94798 | 0.8413 | 0.9761 | 0.6363 |
| Many | raw_margin | q25 | -2.27346 | 0.9161 | 0.8858 | 0.2805 |
| Many | raw_margin | q50 | -0.37710 | 0.9766 | 0.6296 | 0.0521 |
| Many | raw_margin | q75 | 1.28524 | 0.9917 | 0.3196 | 0.0092 |
| Many | raw_margin | q90 | 2.57076 | 0.9956 | 0.1285 | 0.0020 |
| Many | lc_plus_log_prior_margin | zero | 0.00000 | 0.9796 | 0.6425 | 0.0462 |
| Many | lc_plus_log_prior_margin | q10 | -3.92643 | 0.8430 | 0.9780 | 0.6297 |
| Many | lc_plus_log_prior_margin | q25 | -2.08442 | 0.9202 | 0.8898 | 0.2667 |
| Many | lc_plus_log_prior_margin | q50 | 0.07131 | 0.9799 | 0.6317 | 0.0449 |
| Many | lc_plus_log_prior_margin | q75 | 1.99397 | 0.9947 | 0.3206 | 0.0059 |
| Many | lc_plus_log_prior_margin | q90 | 3.41175 | 0.9970 | 0.1287 | 0.0013 |
| Many | inverse_prior_margin | zero | 0.00000 | 0.9820 | 0.3538 | 0.0224 |
| Many | inverse_prior_margin | q10 | -4.53035 | 0.8382 | 0.9725 | 0.6488 |
| Many | inverse_prior_margin | q25 | -2.97668 | 0.9080 | 0.8780 | 0.3076 |
| Many | inverse_prior_margin | q50 | -1.31395 | 0.9636 | 0.6212 | 0.0812 |
| Many | inverse_prior_margin | q75 | 0.19738 | 0.9858 | 0.3177 | 0.0158 |
| Many | inverse_prior_margin | q90 | 1.55121 | 0.9941 | 0.1283 | 0.0026 |
| Many | class_normalized_raw_margin | zero | 0.00000 | 0.9666 | 0.7021 | 0.0838 |
| Many | class_normalized_raw_margin | q10 | -1.35491 | 0.8435 | 0.9786 | 0.6277 |
| Many | class_normalized_raw_margin | q25 | -0.61754 | 0.9188 | 0.8885 | 0.2713 |
| Many | class_normalized_raw_margin | q50 | 0.15500 | 0.9730 | 0.6273 | 0.0601 |
| Many | class_normalized_raw_margin | q75 | 0.70884 | 0.9864 | 0.3179 | 0.0152 |
| Many | class_normalized_raw_margin | q90 | 1.11999 | 0.9852 | 0.1272 | 0.0066 |
| Medium | raw_margin | zero | 0.00000 | 0.9310 | 0.4291 | 0.0217 |
| Medium | raw_margin | q10 | -4.19828 | 0.4433 | 0.9846 | 0.8423 |
| Medium | raw_margin | q25 | -3.07495 | 0.5122 | 0.9479 | 0.6150 |
| Medium | raw_margin | q50 | -1.86372 | 0.6630 | 0.8177 | 0.2832 |
| Medium | raw_margin | q75 | -0.49454 | 0.8797 | 0.5429 | 0.0506 |
| Medium | raw_margin | q90 | 0.83887 | 0.9609 | 0.2372 | 0.0066 |
| Medium | lc_plus_log_prior_margin | zero | 0.00000 | 0.9611 | 0.3095 | 0.0085 |
| Medium | lc_plus_log_prior_margin | q10 | -5.18699 | 0.4446 | 0.9875 | 0.8403 |
| Medium | lc_plus_log_prior_margin | q25 | -3.88715 | 0.5128 | 0.9489 | 0.6143 |
| Medium | lc_plus_log_prior_margin | q50 | -2.62526 | 0.6685 | 0.8245 | 0.2786 |
| Medium | lc_plus_log_prior_margin | q75 | -1.12481 | 0.8844 | 0.5458 | 0.0486 |
| Medium | lc_plus_log_prior_margin | q90 | 0.32819 | 0.9609 | 0.2372 | 0.0066 |
| Medium | inverse_prior_margin | zero | 0.00000 | 0.8931 | 0.4349 | 0.0355 |
| Medium | inverse_prior_margin | q10 | -3.67568 | 0.4420 | 0.9817 | 0.8443 |
| Medium | inverse_prior_margin | q25 | -2.71295 | 0.5029 | 0.9306 | 0.6268 |
| Medium | inverse_prior_margin | q50 | -1.57333 | 0.6482 | 0.7994 | 0.2957 |
| Medium | inverse_prior_margin | q75 | -0.34075 | 0.8516 | 0.5256 | 0.0624 |
| Medium | inverse_prior_margin | q90 | 0.88552 | 0.9648 | 0.2382 | 0.0059 |
| Medium | class_normalized_raw_margin | zero | 0.00000 | 0.6664 | 0.8129 | 0.2773 |
| Medium | class_normalized_raw_margin | q10 | -1.21375 | 0.4433 | 0.9846 | 0.8423 |
| Medium | class_normalized_raw_margin | q25 | -0.64187 | 0.5133 | 0.9499 | 0.6137 |
| Medium | class_normalized_raw_margin | q50 | -0.01653 | 0.6622 | 0.8168 | 0.2838 |
| Medium | class_normalized_raw_margin | q75 | 0.71156 | 0.8438 | 0.5207 | 0.0657 |
| Medium | class_normalized_raw_margin | q90 | 1.27181 | 0.9102 | 0.2247 | 0.0151 |
| Few | raw_margin | zero | 0.00000 | 0.4576 | 0.1392 | 0.0238 |
| Few | raw_margin | q10 | -4.32806 | 0.1397 | 0.9948 | 0.8860 |
| Few | raw_margin | q25 | -3.21973 | 0.1519 | 0.9021 | 0.7280 |
| Few | raw_margin | q50 | -2.24048 | 0.1784 | 0.7062 | 0.4702 |
| Few | raw_margin | q75 | -1.36423 | 0.2240 | 0.4433 | 0.2221 |
| Few | raw_margin | q90 | -0.62842 | 0.3247 | 0.2577 | 0.0775 |
| Few | lc_plus_log_prior_margin | zero | 0.00000 | 0.7333 | 0.0567 | 0.0030 |
| Few | lc_plus_log_prior_margin | q10 | -5.68159 | 0.1397 | 0.9948 | 0.8860 |
| Few | lc_plus_log_prior_margin | q25 | -4.47326 | 0.1519 | 0.9021 | 0.7280 |
| Few | lc_plus_log_prior_margin | q50 | -3.37398 | 0.1771 | 0.7010 | 0.4709 |
| Few | lc_plus_log_prior_margin | q75 | -2.44522 | 0.2214 | 0.4381 | 0.2228 |
| Few | lc_plus_log_prior_margin | q90 | -1.59210 | 0.3247 | 0.2577 | 0.0775 |
| Few | inverse_prior_margin | zero | 0.00000 | 0.3053 | 0.2062 | 0.0678 |
| Few | inverse_prior_margin | q10 | -3.24164 | 0.1389 | 0.9897 | 0.8867 |
| Few | inverse_prior_margin | q25 | -2.43613 | 0.1415 | 0.8402 | 0.7370 |
| Few | inverse_prior_margin | q50 | -1.58147 | 0.1680 | 0.6649 | 0.4762 |
| Few | inverse_prior_margin | q75 | -0.74812 | 0.1953 | 0.3866 | 0.2303 |
| Few | inverse_prior_margin | q90 | -0.10853 | 0.2792 | 0.2216 | 0.0827 |
| Few | class_normalized_raw_margin | zero | 0.00000 | 0.1743 | 0.7680 | 0.5261 |
| Few | class_normalized_raw_margin | q10 | -1.33379 | 0.1382 | 0.9845 | 0.8875 |
| Few | class_normalized_raw_margin | q25 | -0.53037 | 0.1536 | 0.9124 | 0.7265 |
| Few | class_normalized_raw_margin | q50 | 0.12933 | 0.1862 | 0.7371 | 0.4657 |
| Few | class_normalized_raw_margin | q75 | 0.68439 | 0.2135 | 0.4227 | 0.2250 |
| Few | class_normalized_raw_margin | q90 | 1.12844 | 0.2857 | 0.2268 | 0.0820 |

## Interpretation guardrails
- AUC measures ranking, not whether a usable fixed threshold exists.
- The class-normalized score is a positive affine transform within each observed class, so within-class ROC-AUC/AP must match raw margin. Pooled group metrics may differ because classes are rescaled relative to one another.
- z + log(pi) is included only as a diagnostic of the training convention; z - log(pi) is a competing hypothesis. Neither is assumed superior.
- Do not use clean labels, flip masks, AUC, or PR-AUC to choose a threshold for a later training run.
- This audit does not change BCL, does not train a noise filter, and does not establish real-world label-noise robustness.

## Provenance
- Checkpoint SHA256: 10f0cb41db390f04a9d401433cc121396ab67fad7eb11dee0095c40350d75428
- Checkpoint model-state SHA256: ef1a96e6d018a0dc2f6888b8816643935ab100856b6ad41655371a4ec5498102
- Noise-manifest SHA256: 0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8
- Fixed indices SHA256: be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7
- Inference seconds: 1.97
- Device: cuda:0
- TEST file opens: 0 (denied: 0)
