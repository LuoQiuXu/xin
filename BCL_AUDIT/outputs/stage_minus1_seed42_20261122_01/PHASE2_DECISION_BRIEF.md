# Phase 2 H1-H3 Decision Brief

## Scope and evidential boundary

- Dataset/protocol: CIFAR-100-LT IF=100, ResNet-32, paired LC-only vs BCL, seed 42.
- This is one paired seed. The bootstrap intervals quantify class or class-endpoint sampling uncertainty conditional on seed 42. There is no independent replication or causal identification.
- DINOv2 topology is TRAIN-only and fixed before final TEST analysis. TEST is read only from saved epoch-200 prediction/logit archives. No TEST forward or autograd is used.
- No new training, Oracle, extra seed, or method is executed by this analysis.

## H1 — Class-level gain heterogeneity

Few classes: 30; positive/zero/negative BCL-minus-LC gains: 16/2/12.
Mean/median gain: 2.167/1.500 pp; IQR: -2.000 to 7.000 pp; range: -8.000 to 16.000 pp.
Mixed positive and negative Few-class gains: **True**.

| Group | Classes | Mean gain (pp) | Median gain (pp) | SD (pp) | Q1–Q3 (pp) | Min–max (pp) | Positive/zero/negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| many | 35 | 3.971 | 4.000 | 5.028 | 1.000–7.000 | -7.000–19.000 | 28/3/4 |
| medium | 35 | 2.543 | 2.000 | 4.650 | -0.500–6.000 | -5.000–12.000 | 21/5/9 |
| few | 30 | 2.167 | 1.500 | 6.320 | -2.000–7.000 | -8.000–16.000 | 16/2/12 |

The [class-level gain table](phase2_h1_class_gain.csv) contains all 100 observations.

### Few-class top 10 gains

| Class | ID | Gain (pp) |
|---|---:|---:|
| television | 87 | 16.000 |
| tiger | 88 | 13.000 |
| tulip | 92 | 10.000 |
| wolf | 97 | 10.000 |
| trout | 91 | 9.000 |
| sunflower | 82 | 9.000 |
| sweet_pepper | 83 | 8.000 |
| snake | 78 | 7.000 |
| train | 90 | 7.000 |
| wardrobe | 94 | 7.000 |

### Few-class bottom 10 gains

| Class | ID | Gain (pp) |
|---|---:|---:|
| skunk | 75 | -8.000 |
| woman | 98 | -8.000 |
| willow_tree | 96 | -7.000 |
| tank | 85 | -5.000 |
| table | 84 | -5.000 |
| tractor | 89 | -5.000 |
| sea | 71 | -2.000 |
| shrew | 74 | -2.000 |
| skyscraper | 76 | -2.000 |
| seal | 72 | -2.000 |

## H2 — Topology exposure and deconfounded associations

Few-class exposure is reported separately as (1) any Many-class neighbor in the symmetric-union DINO Top-5 graph, (2) any same-superclass Many class, and (3) maximum DINO cosine similarity to the 35 Many classes.

Class-level adjusted associations control LC-only TEST accuracy and log training count for gain. Tail-to-Head confusion change also controls LC-only baseline Tail-to-Head confusion. Delta accuracy contains the LC-only baseline with a negative sign, which can couple it to that control. Read the unadjusted Spearman alongside the adjusted estimates. Intervals use the 2,000-replicate Few-class bootstrap.

| Exposure | Outcome | Adjusted standardized beta [95% class-bootstrap interval] | Adjusted partial Spearman [95% class-bootstrap interval] | Unadjusted Spearman [95% class-bootstrap interval] | Valid replicates (beta / partial ρ / unadjusted ρ) |
|---|---|---:|---:|---:|---:|
| max_dino_similarity_to_many | delta_accuracy_pp | -0.313 [-0.581, 0.033] | -0.321 [-0.585, 0.069] | -0.276 [-0.569, 0.103] | 2000 / 2000 / 2000 of 2000 |
| max_dino_similarity_to_many | delta_tail_to_head_mean_confusion_pp | 0.302 [-0.133, 0.694] | 0.287 [-0.144, 0.608] | 0.195 [-0.209, 0.530] | 2000 / 2000 / 2000 of 2000 |
| has_dino_many_neighbor | delta_accuracy_pp | 0.005 [-0.410, 0.381] | 0.016 [-0.380, 0.444] | -0.008 [-0.426, 0.406] | 1750 / 1750 / 1746 of 2000 |
| has_dino_many_neighbor | delta_tail_to_head_mean_confusion_pp | -0.067 [-0.377, 0.282] | -0.068 [-0.402, 0.285] | -0.100 [-0.371, 0.155] | 1758 / 1758 / 1773 of 2000 |
| has_same_superclass_many | delta_accuracy_pp | 0.141 [-0.225, 0.503] | 0.150 [-0.242, 0.516] | 0.151 [-0.223, 0.513] | 2000 / 2000 / 2000 of 2000 |
| has_same_superclass_many | delta_tail_to_head_mean_confusion_pp | 0.217 [-0.165, 0.579] | 0.225 [-0.180, 0.567] | 0.172 [-0.181, 0.518] | 2000 / 2000 / 2000 of 2000 |

LOCO refits both adjusted estimators after dropping each Few class in turn. The results measure single-class influence conditional on this trained seed. They provide neither training-seed uncertainty nor evidence of causal identification or independent replication. A zero or non-finite full estimate has no sign-stability statistic. A zero omission estimate is not a sign reversal.

| Exposure | Outcome | Estimator | Valid omissions | LOCO range | Fraction with same sign | Sign reversals | Largest change when omitted (name and ID) | Max absolute change |
|---|---|---|---:|---:|---:|---:|---|---:|
| max_dino_similarity_to_many | delta_accuracy_pp | Adjusted beta | 30/30 | [-0.372, -0.256] | 1.000 | 0 | tiger (88) | 0.059 |
| max_dino_similarity_to_many | delta_accuracy_pp | Partial Spearman | 30/30 | [-0.379, -0.259] | 1.000 | 0 | woman (98) | 0.061 |
| max_dino_similarity_to_many | delta_tail_to_head_mean_confusion_pp | Adjusted beta | 30/30 | [0.238, 0.420] | 1.000 | 0 | telephone (86) | 0.119 |
| max_dino_similarity_to_many | delta_tail_to_head_mean_confusion_pp | Partial Spearman | 30/30 | [0.223, 0.382] | 1.000 | 0 | telephone (86) | 0.095 |
| has_dino_many_neighbor | delta_accuracy_pp | Adjusted beta | 30/30 | [-0.214, 0.209] | 0.667 | 10 | tank (85) | 0.219 |
| has_dino_many_neighbor | delta_accuracy_pp | Partial Spearman | 30/30 | [-0.224, 0.242] | 0.867 | 4 | tank (85) | 0.240 |
| has_dino_many_neighbor | delta_tail_to_head_mean_confusion_pp | Adjusted beta | 30/30 | [-0.197, 0.126] | 0.967 | 1 | tank (85) | 0.193 |
| has_dino_many_neighbor | delta_tail_to_head_mean_confusion_pp | Partial Spearman | 30/30 | [-0.209, 0.119] | 0.967 | 1 | tank (85) | 0.188 |
| has_same_superclass_many | delta_accuracy_pp | Adjusted beta | 30/30 | [0.048, 0.234] | 1.000 | 0 | willow_tree (96) | 0.094 |
| has_same_superclass_many | delta_accuracy_pp | Partial Spearman | 30/30 | [0.058, 0.253] | 1.000 | 0 | sunflower (82) | 0.104 |
| has_same_superclass_many | delta_tail_to_head_mean_confusion_pp | Adjusted beta | 30/30 | [0.149, 0.301] | 1.000 | 0 | tank (85) | 0.083 |
| has_same_superclass_many | delta_tail_to_head_mean_confusion_pp | Partial Spearman | 30/30 | [0.170, 0.313] | 1.000 | 0 | tank (85) | 0.087 |

For the continuous exposure the descriptive groups use ranks 1–15 (lower) and 16–30 (upper) after sorting ascending by (maximum DINO similarity, class ID). This split is for descriptions only. The associations use the continuous values. Group counts include finite outcomes.

| Exposure | Outcome | Exposed / unexposed n | Exposed / unexposed mean (pp) | Exposed / unexposed median (pp) | Exposed minus unexposed mean (pp) |
|---|---|---:|---:|---:|---:|
| max_dino_similarity_to_many | delta_accuracy_pp | 15 / 15 | 1.533 / 2.800 | 0.000 / 3.000 | -1.267 |
| max_dino_similarity_to_many | delta_tail_to_head_mean_confusion_pp | 15 / 15 | 0.044 / -0.010 | 0.029 / -0.029 | 0.053 |
| has_dino_many_neighbor | delta_accuracy_pp | 28 / 2 | 2.143 / 2.500 | 1.500 / 2.500 | -0.357 |
| has_dino_many_neighbor | delta_tail_to_head_mean_confusion_pp | 28 / 2 | 0.013 / 0.071 | 0.029 / 0.071 | -0.058 |
| has_same_superclass_many | delta_accuracy_pp | 21 / 9 | 2.762 / 0.778 | 3.000 / 0.000 | 1.984 |
| has_same_superclass_many | delta_tail_to_head_mean_confusion_pp | 21 / 9 | 0.034 / -0.022 | 0.057 / -0.029 | 0.056 |

Pair-level DID uses C[a→b] = P(prediction=b given true=a) and delta C = BCL − LC-only. Each exposed pair is matched with replacement to the nearest unexposed pair on the LC-only baseline directional confusion. The endpoint bootstrap independently resamples Many and Few IDs, weights pairs by the product of multiplicities, and recomputes matching in each replicate. Confusion effects are in percentage points and margins are in logit units.

| Relation | Direction | Outcome | Exposed/control/matched pairs | Unadjusted DID | Matched DID [95% endpoint-cluster interval] | Valid replicates | Baseline-confusion SMD before → after |
|---|---|---|---:|---:|---:|---:|---:|
| same_superclass | Head_to_Tail | delta_confusion_BCL_minus_LC | 35/1015/35 | -0.185 | 0.486 [-1.120, 1.552] | 2000/2000 | 0.774 → 0.010 |
| same_superclass | Head_to_Tail | delta_raw_margin_BCL_minus_LC | 35/1015/35 | -0.497 | -1.825 [-2.266, 0.386] | 2000/2000 | 0.774 → 0.010 |
| same_superclass | Head_to_Tail | delta_adjusted_margin_BCL_minus_LC | 35/1015/35 | -0.497 | -1.825 [-2.266, 0.386] | 2000/2000 | 0.774 → 0.010 |
| same_superclass | Tail_to_Head | delta_confusion_BCL_minus_LC | 35/1015/35 | 0.012 | 1.086 [-2.438, 2.467] | 2000/2000 | 0.865 → 0.090 |
| same_superclass | Tail_to_Head | delta_raw_margin_BCL_minus_LC | 35/1015/35 | -0.375 | -0.899 [-1.331, 0.417] | 2000/2000 | 0.865 → 0.090 |
| same_superclass | Tail_to_Head | delta_adjusted_margin_BCL_minus_LC | 35/1015/35 | -0.375 | -0.899 [-1.331, 0.417] | 2000/2000 | 0.865 → 0.090 |
| dino_topk_union | Head_to_Tail | delta_confusion_BCL_minus_LC | 62/988/62 | -0.406 | 0.032 [-0.891, 1.167] | 2000/2000 | 1.089 → 0.192 |
| dino_topk_union | Head_to_Tail | delta_raw_margin_BCL_minus_LC | 62/988/62 | -0.387 | -1.187 [-1.901, 0.448] | 2000/2000 | 1.089 → 0.192 |
| dino_topk_union | Head_to_Tail | delta_adjusted_margin_BCL_minus_LC | 62/988/62 | -0.387 | -1.187 [-1.901, 0.448] | 2000/2000 | 1.089 → 0.192 |
| dino_topk_union | Tail_to_Head | delta_confusion_BCL_minus_LC | 62/988/62 | 0.256 | 0.194 [-2.019, 3.309] | 2000/2000 | 1.088 → 0.222 |
| dino_topk_union | Tail_to_Head | delta_raw_margin_BCL_minus_LC | 62/988/62 | -0.418 | -0.800 [-1.297, 0.129] | 2000/2000 | 1.088 → 0.222 |
| dino_topk_union | Tail_to_Head | delta_adjusted_margin_BCL_minus_LC | 62/988/62 | -0.418 | -0.800 [-1.297, 0.129] | 2000/2000 | 1.088 → 0.222 |

## H3 — Optimization and margin trajectory

The gradient table shows exposed-minus-unexposed contrasts at each checkpoint. Common-input eval and native three-view train-mode gradients are separate estimands. Positive conflict-fraction differences mean more negative-cosine observations in the exposed group. Raw and objective-weighted ratios use the existing LC and BCL gradient norms. The [global native diagnostics](offline_native_three_view_gradient_global_metrics.csv) include backbone and per-view input results.

| Mode | Exposure | Metric | 40 | 100 | 160 | 180 | 200 |
|---|---|---|---:|---:|---:|---:|---:|
| common_input_eval | dino_many_neighbor | W_y_gradient_cosine | 0.098 | 0.102 | 0.108 | 0.075 | 0.083 |
| common_input_eval | dino_many_neighbor | W_y_conflict_fraction | -0.036 | -0.036 | 0.000 | 0.000 | -0.036 |
| common_input_eval | dino_many_neighbor | W_y_raw_BCL_to_LC_norm_ratio | 0.012 | -0.012 | 0.020 | 0.014 | 0.013 |
| common_input_eval | dino_many_neighbor | W_y_weighted_BCL_to_LC_norm_ratio | 0.004 | -0.003 | 0.006 | 0.004 | 0.004 |
| common_input_eval | dino_many_neighbor | same_feature_gradient_cosine | 0.169 | -0.049 | 0.032 | -0.055 | -0.023 |
| common_input_eval | dino_many_neighbor | same_feature_conflict_fraction | -0.114 | 0.057 | -0.057 | 0.014 | 0.014 |
| common_input_eval | dino_many_neighbor | same_feature_raw_BCL_to_LC_norm_ratio | 0.002 | -0.047 | -0.078 | -0.149 | -0.108 |
| common_input_eval | dino_many_neighbor | same_feature_weighted_BCL_to_LC_norm_ratio | 0.001 | -0.014 | -0.023 | -0.045 | -0.033 |
| common_input_eval | same_superclass_many | W_y_gradient_cosine | 0.079 | 0.019 | -0.029 | 0.041 | 0.070 |
| common_input_eval | same_superclass_many | W_y_conflict_fraction | -0.048 | -0.048 | 0.000 | 0.000 | -0.048 |
| common_input_eval | same_superclass_many | W_y_raw_BCL_to_LC_norm_ratio | 0.001 | 0.005 | -0.003 | 0.005 | 0.004 |
| common_input_eval | same_superclass_many | W_y_weighted_BCL_to_LC_norm_ratio | 0.000 | 0.002 | -0.001 | 0.002 | 0.001 |
| common_input_eval | same_superclass_many | same_feature_gradient_cosine | 0.065 | 0.055 | 0.045 | 0.035 | 0.027 |
| common_input_eval | same_superclass_many | same_feature_conflict_fraction | 0.006 | -0.083 | 0.003 | -0.013 | -0.013 |
| common_input_eval | same_superclass_many | same_feature_raw_BCL_to_LC_norm_ratio | 0.006 | 0.005 | -0.017 | 0.006 | 0.043 |
| common_input_eval | same_superclass_many | same_feature_weighted_BCL_to_LC_norm_ratio | 0.002 | 0.002 | -0.005 | 0.002 | 0.013 |
| native_three_view_train | dino_many_neighbor | W_y_gradient_cosine | 0.145 | 0.202 | 0.131 | 0.094 | 0.092 |
| native_three_view_train | dino_many_neighbor | W_y_conflict_fraction | -0.143 | -0.214 | -0.214 | -0.179 | -0.214 |
| native_three_view_train | dino_many_neighbor | W_y_raw_BCL_to_LC_norm_ratio | 0.006 | 0.016 | 0.038 | 0.034 | 0.032 |
| native_three_view_train | dino_many_neighbor | W_y_weighted_BCL_to_LC_norm_ratio | 0.002 | 0.005 | 0.011 | 0.010 | 0.010 |
| native_three_view_train | dino_many_neighbor | feature_v1_gradient_cosine | 0.003 | 0.002 | -0.049 | -0.134 | -0.144 |
| native_three_view_train | dino_many_neighbor | feature_v1_conflict_fraction | -0.093 | 0.007 | -0.064 | 0.229 | 0.271 |
| native_three_view_train | dino_many_neighbor | feature_v1_raw_BCL_to_LC_norm_ratio | -0.002 | 0.006 | 0.001 | -0.005 | -0.009 |
| native_three_view_train | dino_many_neighbor | feature_v1_weighted_BCL_to_LC_norm_ratio | -0.001 | 0.002 | 0.000 | -0.002 | -0.003 |
| native_three_view_train | dino_many_neighbor | feature_v2_gradient_cosine | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | dino_many_neighbor | feature_v2_conflict_fraction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | dino_many_neighbor | feature_v2_raw_BCL_to_LC_norm_ratio | -238452571.232 | -81741948.712 | -370320262.404 | -464610680.189 | -492009119.833 |
| native_three_view_train | dino_many_neighbor | feature_v2_weighted_BCL_to_LC_norm_ratio | -143071542.739 | -49045169.227 | -222192157.442 | -278766408.114 | -295205471.900 |
| native_three_view_train | dino_many_neighbor | feature_v3_gradient_cosine | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | dino_many_neighbor | feature_v3_conflict_fraction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | dino_many_neighbor | feature_v3_raw_BCL_to_LC_norm_ratio | -145256295.530 | -3642352.801 | 96758748.987 | -449520180.389 | -575823119.600 |
| native_three_view_train | dino_many_neighbor | feature_v3_weighted_BCL_to_LC_norm_ratio | -87153777.318 | -2185411.681 | 58055249.392 | -269712108.233 | -345493871.760 |
| native_three_view_train | same_superclass_many | W_y_gradient_cosine | 0.141 | 0.143 | 0.098 | 0.127 | 0.120 |
| native_three_view_train | same_superclass_many | W_y_conflict_fraction | -0.190 | -0.127 | -0.127 | -0.238 | -0.127 |
| native_three_view_train | same_superclass_many | W_y_raw_BCL_to_LC_norm_ratio | 0.004 | 0.014 | 0.001 | -0.005 | -0.005 |
| native_three_view_train | same_superclass_many | W_y_weighted_BCL_to_LC_norm_ratio | 0.001 | 0.004 | 0.000 | -0.002 | -0.001 |
| native_three_view_train | same_superclass_many | feature_v1_gradient_cosine | 0.073 | 0.035 | 0.030 | -0.018 | -0.002 |
| native_three_view_train | same_superclass_many | feature_v1_conflict_fraction | -0.156 | 0.010 | -0.102 | 0.051 | 0.044 |
| native_three_view_train | same_superclass_many | feature_v1_raw_BCL_to_LC_norm_ratio | -0.005 | -0.003 | -0.007 | -0.004 | -0.008 |
| native_three_view_train | same_superclass_many | feature_v1_weighted_BCL_to_LC_norm_ratio | -0.002 | -0.001 | -0.002 | -0.001 | -0.002 |
| native_three_view_train | same_superclass_many | feature_v2_gradient_cosine | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | same_superclass_many | feature_v2_conflict_fraction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | same_superclass_many | feature_v2_raw_BCL_to_LC_norm_ratio | -84637271.911 | 206177011.280 | -41914455.009 | -61659609.330 | -91434252.727 |
| native_three_view_train | same_superclass_many | feature_v2_weighted_BCL_to_LC_norm_ratio | -50782363.147 | 123706206.768 | -25148673.005 | -36995765.598 | -54860551.636 |
| native_three_view_train | same_superclass_many | feature_v3_gradient_cosine | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | same_superclass_many | feature_v3_conflict_fraction | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| native_three_view_train | same_superclass_many | feature_v3_raw_BCL_to_LC_norm_ratio | -50087088.935 | 41945149.895 | 96593487.875 | -221229126.143 | -224200556.783 |
| native_three_view_train | same_superclass_many | feature_v3_weighted_BCL_to_LC_norm_ratio | -30052253.361 | 25167089.937 | 57956092.725 | -132737475.686 | -134520334.070 |

The margin table compares the BCL-minus-LC change between exposed and unexposed classes.

| Exposure | Source | Epoch | Raw margin contrast | Logit-adjusted margin contrast |
|---|---|---:|---:|---:|
| dino_many_neighbor | LT_train_probe | 40 | -0.325 | -0.923 |
| dino_many_neighbor | LT_train_probe | 100 | -0.873 | -0.863 |
| dino_many_neighbor | LT_train_probe | 160 | 0.554 | 0.749 |
| dino_many_neighbor | LT_train_probe | 180 | 0.493 | -0.009 |
| dino_many_neighbor | LT_train_probe | 200 | 0.667 | -0.331 |
| same_superclass_many | LT_train_probe | 40 | -0.207 | -0.572 |
| same_superclass_many | LT_train_probe | 100 | -0.177 | -0.233 |
| same_superclass_many | LT_train_probe | 160 | 0.459 | 0.454 |
| same_superclass_many | LT_train_probe | 180 | 0.941 | 0.734 |
| same_superclass_many | LT_train_probe | 200 | 1.016 | 0.730 |
| dino_many_neighbor | balanced_TEST_read_only | 200 | 0.100 | 0.106 |
| same_superclass_many | balanced_TEST_read_only | 200 | 0.184 | 0.079 |

The saved native diagnostics verified equality of running_mean, running_var and num_batches_tracked after every diagnostic batch at all five checkpoints. The TEST margin rows come from the saved epoch-200 logits.

## Required decision questions

1. **Specific weakness.** No topology-localized BCL weakness is supported by the seed-42 evidence.
2. **Evidence strength.** UNSUPPORTED. Exploratory evidence conditional on seed 42.
3. **Counterfactual or stop.** **Stop this BCL-specific direction at Phase 2.** Few-class gain heterogeneity alone leaves the mechanism unresolved. The pre-specified class direction and matched Tail-to-Head contrasts provide insufficient stable localization.

## Stop line

Stop this BCL-specific direction at Phase 2. Human review is required before any further experiment.
