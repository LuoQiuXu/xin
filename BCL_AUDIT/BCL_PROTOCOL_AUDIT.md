# BCL CIFAR-100-LT Protocol Audit

**Status:** protocol audit / Stage -1  
**Scope:** CIFAR-100-LT, IF=100, ResNet-32, one-seed 200-epoch pilot  
**Research status:** BCL is a candidate research base. No BCL defect or improvement method is assumed.

## 1. Research objective

The immediate objective is not to design a new loss or gate. It is to establish a defensible CIFAR-100-LT BCL implementation and then determine whether BCL exhibits a reproducible, topology-localized weakness.

The first controlled comparison is:

- **Group A — LC-only:** receives the same three independently sampled views (v_1,v_2,v_3); all three are concatenated through the shared ResNet-32, but only v_1 logits enter logit-compensated cross-entropy.
- **Group B — BCL:** receives the same three-view protocol and concatenated shared-backbone forward; v_1 logits enter LC, while v_2/v_3 sample embeddings and classifier-derived prototypes enter the BCL objective.

Do not add gradient surgery, prototype refinement, semantic gates, or other proposed improvements in Stage -1.

## 2. Paper-grounded CIFAR protocol

Primary source: Zhu et al., *Balanced Contrastive Learning for Long-Tailed Visual Recognition*, CVPR 2022, Sections 3–4 and Tables 3–4.

For CIFAR-100-LT, IF=100:

| Item | 200-epoch pilot setting |
|---|---:|
| Dataset | CIFAR-100-LT |
| Imbalance factor | 100 |
| Backbone | CIFAR ResNet-32 |
| Batch size | 256 |
| Base learning rate | 0.15 |
| Warmup | 5 epochs, linearly warm to 0.15 |
| LR drops | after 160 and 180 completed epochs, multiply LR by 0.1 |
| Weight decay | 5e-4 |
| BCL temperature | 0.1 |
| Logit-compensation coefficient λ | 2.0 |
| BCL coefficient μ | 0.6 |
| Sample/prototype MLP hidden dimension | 512 |
| Sample/prototype MLP output dimension | 128 |
| Precision | FP32 for the first pilot |
| Optimizer | SGD, momentum 0.9 (see qualification below) |
| Training length | 200 epochs |

The paper reports **51.93%** Top-1 for BCL on CIFAR-100-LT IF=100 at 200 epochs (Table 3 / Table 4). It reports **53.9%** at 400 epochs. Table 2's **50.8% LC-only versus 53.9% BCL** is a 400-epoch ablation. Do not compare the 200-epoch BCL result directly against the 400-epoch LC-only number.

The accessible CIFAR implementation paragraph does not explicitly spell out the CIFAR optimizer and momentum. SGD with momentum 0.9 is selected as the pilot implementation convention, consistent with the released BCL large-scale training code and the referenced CIFAR long-tail recipe. Record this as an implementation choice, not as a separately verified CIFAR-specific paper statement.

## 3. Source-code audit: what is and is not provided

### 3.1 Official BCL repository

Repository: https://github.com/FlamieZhu/Balanced-Contrastive-Learning

Verified from the public default branch:

- `main.py` exposes `--dataset` choices `inat` and `imagenet`; it does not expose a CIFAR-100 training route.
- `main.py` builds ResNet-50 / ResNeXt-50 models, not CIFAR ResNet-32.
- `dataset/cifar.py` contains an imbalanced CIFAR dataset utility, but this is not wired into the public `main.py` training entry point.
- `loss/contrastive.py` contains the released `BalSCL` implementation.
- `models/resnext.py` contains the large-scale BCL model and `NormedLinear`.

Therefore this project must implement a **standalone, paper-faithful CIFAR port**. Do not describe it as an unmodified run of the official CIFAR code; that CIFAR entry point is not present in the inspected public repository.

### 3.2 Three-view forward and BatchNorm

In the released BCL training loop, the three views are concatenated along the batch dimension before the shared model forward:

`inputs = cat([v1, v2, v3], dim=0)`

The shared backbone therefore computes its BatchNorm2d statistics over the concatenated
(v_1,v_2,v_3) distributions. The classification logits are taken from the first
B outputs; the contrastive embeddings are taken from the (v_2,v_3) chunks.

There is a second, easy-to-miss coupling point: the released sample projection head
is also applied to the full 3B feature tensor before the code selects the
(v_2,v_3) projected outputs. That head contains BatchNorm1d. Its batch statistics
therefore also mix v_1,v_2,v_3.

Consequently, although the BCL objective directly consumes only the (v_2,v_3)
projection outputs, in train mode its gradients can flow indirectly to v_1
backbone features through cross-view BatchNorm statistics (in both the backbone
and sample projection head). This is a view-level gradient coupling, not a third
shared trainable-parameter conflict site.

**Paired-control correction:** LC-only and BCL now both concatenate (v_1,v_2,v_3)
through the shared backbone. LC-only uses only the v_1 logits in its objective;
BCL uses v_1 for LC and v_2/v_3 for the contrastive objective. Thus backbone
BatchNorm2d sees a matched three-view input distribution in both groups.

The BCL sample projection MLP is applied to all 3B features, so its BatchNorm1d
also mixes v_1,v_2,v_3. This projection MLP is not executed in LC-only because it
does not contribute to the LC objective. Keep this remaining branch-specific
BatchNorm1d difference explicit. The training script records first-minibatch
per-view pre-BN activation moments and running statistics at epochs 40/100/160/
180/200 in `batchnorm_audit.jsonl`.

In train mode, the shared backbone BatchNorm2d couples view rows in both groups.
The BCL sample-projection BatchNorm1d introduces an additional BCL-specific
view-coupling path. This is a view-level coupling effect, not a third shared
trainable-parameter conflict site. Do not add another training group in this
first paired pilot.

### 3.3 Shared-gradient targets and branch-specific modules

The relevant computation graph is:

- Classification: (v_1 \rightarrow f_\theta \rightarrow W \rightarrow \mathcal L_{LC})
- Contrastive samples: (v_2,v_3 \rightarrow f_\theta \rightarrow h_{sample} \rightarrow z_2,z_3 \rightarrow \mathcal L_{BCL})
- Contrastive prototypes: (W \rightarrow h_{proto} \rightarrow c_1,\ldots,c_K \rightarrow \mathcal L_{BCL})

The parameters shared by both losses are:

1. **Backbone parameters** (\theta)
2. **Classifier weights** (W), because BCL reaches (W) through the prototype MLP

The sample projection MLP and prototype MLP receive BCL gradients only. The
prototype MLP is a gradient-transformation path from BCL through its Jacobian to
(W), not a third cross-branch conflict site. The sample MLP's BatchNorm1d does,
however, couple the three views in train mode; BCL gradients can reach the
v_1 feature rows through those batch statistics even though only v_2,v_3
projected outputs enter the BCL loss.

The merged offline diagnostic observes only:

- parameter-level: (\nabla_{W_y}\mathcal L_{LC}) versus (\nabla_{W_y}\mathcal L_{BCL})
- representation-level: (\nabla_{f(x)}\mathcal L_{LC}) versus (\nabla_{f(x)}\mathcal L_{BCL})

For the representation-level comparison, the script uses the same fixed v_2
probe feature tensor f(x) for both losses: LC is computed by applying W to that
same f(x), while BCL uses that same f(x) through the sample MLP, paired with v_3.
The prototype path uses the same classifier W through the prototype MLP.

The offline gradient probe runs in eval mode (frozen BN running statistics,
without cross-example BN gradient coupling) to isolate local loss-gradient
interaction. Train-mode cross-view BN effects are measured separately by the
training-time BatchNorm audit. Do not compare gradients at different augmented
views and call that a same-point gradient conflict.

### 3.4 Classifier-head qualification: do not copy the ImageNet scale into CIFAR

The official BCL large-scale `NormedLinear` normalizes features and weights and multiplies the cosine output by `s=30`. This is verified in `models/resnext.py`, but that model is used by the released ImageNet/iNaturalist training path.

The BCL paper's CIFAR implementation paragraph does not explicitly state that CIFAR-100 uses this scaled cosine classifier. The referenced PaCo CIFAR implementation provides a CIFAR ResNet-32 with a standard linear classifier by default (`use_norm=False`); its optional `NormedLinear` has no `s=30` multiplier.

**Pilot decision:** use a standard `nn.Linear(64, 100)` classifier for the CIFAR ResNet-32 port. Keep the choice explicit in the audit report. Do not import the BCL ImageNet `s=30` setting into CIFAR. If a direct CIFAR-specific BCL source is found later, compare it before changing this choice.

Useful reference implementation for CIFAR architecture and augmentation semantics:
https://github.com/JIA-Lab-research/Parametric-Contrastive-Learning
- `PaCo/LT/models/resnet_cifar.py`
- `PaCo/LT/paco_cifar.py`
- `PaCo/LT/dataset/imbalance_cifar.py`

## 4. CIFAR-100-LT data protocol

Use the original balanced CIFAR-100 **training** split only to construct the long-tailed training subset and the DINOv2 class topology.

For fine-class index (i=0,\ldots,99), use the standard exponential rule:

[
n_i = \operatorname{int}\left(500\times 0.01^{i/99}\right)
]

Requirements:

- Preserve official fine-label IDs (0\ldots99); do not shuffle class order before assigning counts.
- Use `int()` truncation, not rounding.
- Fix the long-tail sample-selection seed to 0 for the paired runs.
- Keep the training RNG seed separate (pilot seed 42).
- Save the selected original training indices and the exact 100-class count vector.
- Validate that the first class has 500 samples and the last has 5.
- Read CIFAR-100 fine/coarse labels from the original training metadata; verify every fine class maps to exactly one coarse class and each of the 20 superclasses contains five fine classes.

Shot groups for this audit are fixed as: Many (>100), Medium (20\text{--}100), Few (<20). Keep the thresholds in the output metadata.

### 4.0 Strict TRAIN-only file-access boundary

Do not assume that `torchvision.datasets.CIFAR100(train=True)` is file-level
TRAIN-only. In torchvision 0.29.1, its inherited `_check_integrity()` iterates
over both `train_list` and `test_list`, so a nominal TRAIN constructor can
still hash/read the TEST pickle before returning the TRAIN data.

The audited port therefore uses these strict paths:
- Training DataLoader and offline probes: `CIFAR100TrainOnly` in
  `cifar100lt/data.py`, which checks only `train_list`; torchvision's
  separate metadata validation still checks the CIFAR-100 meta file.
- Static taxonomy / DINO audit: directly reads and checksum-validates only the
  original `train` and `meta` pickle files; it does not instantiate a
  torchvision CIFAR dataset. When adding DINO after the static audit, use
  `--reuse-static-audit`; the script validates and reuses the saved index and
  pair table rather than rewriting the static audit outputs.
- Final benchmark: `CIFAR100(train=False)` is permitted only after both
  200-epoch training runs have completed.

Do not instantiate the stock `CIFAR100(train=True)` before final benchmark
evaluation. A test must guard that the train-only loader never checks
`test_list`.

### 4.1 Superclass pair-count audit

Before interpreting any within-superclass Head–Tail result, report:

- number of Many, Medium, and Few fine classes in each superclass
- number of unordered within-superclass Many–Few pairs
- number of unordered cross-superclass Many–Few pairs
- the complete pair list, not just the aggregate count

If the within-superclass Many–Few pair count is small or concentrated in a few superclasses, flag the limitation. Do not infer a stable topology effect from a handful of pairs.

### 4.2 DINOv2 similarity matrix — training data only

Generate the (100\times100) visual class-similarity matrix from the original **balanced CIFAR-100 training split (500 images per class)**. The script must never instantiate or read the CIFAR-100 test split for this computation.

Pilot defaults:

- frozen `dinov2_vits14` (record exact model identifier and checkpoint/cache provenance)
- resize (32\times32\rightarrow224\times224) using Bicubic interpolation
- ImageNet normalization: mean ((0.485,0.456,0.406)), std ((0.229,0.224,0.225))
- extract normalized CLS features, average within each fine class, L2-normalize class means, then compute cosine similarity
- save `DINO_CLASS_SIMILARITY.npy`, class mean features, and a JSON metadata file
- report DINO Top-K neighbors using a documented symmetric-union rule; default (K=5)

Do not use test images, test-derived statistics, or test labels for this topology prior. Test data may be used only for the final benchmark evaluation after training, never for topology construction, checkpoint selection, or hyperparameter selection.

## 5. Augmentation semantics for the CIFAR port

The BCL paper specifies AutoAugment + Cutout for (v_1), and SimAugment for (v_2,v_3). The referenced PaCo CIFAR code provides a concrete CIFAR implementation:

- (v_1): RandomCrop(32, padding=4), RandomHorizontalFlip, CIFAR AutoAugment policy, ToTensor, Cutout(length=16), CIFAR-100 normalization.
- (v_2,v_3): RandomResizedCrop(32, scale=(0.2,1.0)), RandomHorizontalFlip, ColorJitter(0.4,0.4,0.4,0.1) with (p=0.8), RandomGrayscale((p=0.2)), GaussianBlur(([0.1,2.0])) with (p=0.5), ToTensor, CIFAR-100 normalization.
- CIFAR-100 normalization: mean ((0.4914,0.4822,0.4465)), std ((0.2023,0.1994,0.2010)).

Record the exact torchvision / augmentation implementation version. Do not substitute RandAugment for AutoAugment in the CIFAR classification view.

## 6. Stage -1 pilot and deliverables

Run only one paired seed at first:

- seed 42 for both groups
- Group A: LC-only, three views concatenated through the backbone; LC uses only v_1 logits; 200 epochs
- Group B: BCL, the same three-view concatenated backbone forward; LC uses v_1 logits and BCL uses v_2/v_3 plus prototypes; 200 epochs
- FP32, batch size 256, no AMP, no gradient accumulation
- same long-tail sample indices, same seed, same ResNet-32 backbone/classifier initialization protocol
- save checkpoints after human-numbered epochs 40, 100, 160, 180, and 200. The epoch-160 checkpoint must be saved after completing 160 epochs and **before** the LR drop applied at the start of epoch 161; similarly save epoch 180 before the next drop.
- record per-epoch time, total time, peak CUDA memory, train losses, LR, and final Top-1 / Many / Medium / Few accuracy
- evaluate the balanced CIFAR-100 test split only after training, for final reporting; never use it for model selection

The 200-epoch BCL paper result of 51.93% is a sanity reference, not a hard pass/fail threshold. If the result is materially outside the expected neighborhood, first audit the data split, transforms, BCL loss, classifier, LR schedule, and checkpoint/evaluation protocol. Do not “tune until it matches.”

## 7. What Stage -1 must not do

- no 400-epoch run
- no extra seeds 43/44
- no gradient probes during the training loop; after both runs finish, use the single offline diagnostic script described below
- no new method, gate, prototype correction, gradient projection, or semantic reweighting
- no test-set DINO features
- no automatic checkpoint selection using the test set
- no uploading datasets, checkpoints, or large binary outputs to GitHub

The next decision is made only after the protocol audit, the data/topology counts, and the two 200-epoch pilot runs are available.


## 8. Merged offline diagnostics and stage gate

After the paired 200-epoch runs and the final epoch-200 benchmark evaluation,
run one offline script. This combines the previously separated representation
and classifier diagnostics; do not create another training stage for these
measurements.

### 8.1 Fixed training probe set

- Select five fixed images per fine class from the saved IF=100 training indices
  (500 probes total), using probe seed 777.
- Cache v_1, v_2, v_3, and deterministic evaluation views once; reuse exactly
  the same tensors for every checkpoint.
- The probe source is the selected long-tailed CIFAR-100 training subset only.
  The offline probe must not load the CIFAR-100 test split.
- For LC-only and BCL checkpoints at epochs 40/100/160/180/200, report per-class
  probe accuracy and raw / logit-adjusted true-class margins.

### 8.2 Dual-layer gradient comparison

Run this only on BCL checkpoints. The LC-only projection heads are not trained,
so their BCL gradients are not a meaningful comparator.

At each BCL checkpoint, set the model to eval mode and freeze BN running
statistics. Use the same v_2 feature tensor f(x) for both losses:

- LC gradient: v_2 feature f(x) -> classifier W -> logit-compensated CE.
- BCL gradient: the same f(x) -> sample projection MLP -> BCL, with v_3 as its
  paired contrastive view and classifier W -> prototype MLP -> class prototypes.

Report:
- classifier row W_y: cosine(grad_W_y LC, grad_W_y BCL), both gradient norms,
  and BCL/LC norm ratio for each class y;
- same probe feature f(x): cosine(grad_f(x) LC, grad_f(x) BCL), both norms,
  and the fraction of negative-cosine probes per class.

The prototype MLP is only the differentiable transformation path from W to
class prototypes. Do not describe it as an independent cross-branch conflict
site. Eval-mode gradient diagnostics intentionally isolate the local loss
interaction; the train-mode BatchNorm view coupling is reported separately by
the training-time BN audit.

### 8.3 Head–Tail confusion and paired DID

Use the saved epoch-200 final-test predictions/logits only for the final
benchmark confusion analysis. Do not use test results to select checkpoints,
tune hyperparameters, construct DINO similarity, or define semantic overlap.

For model M and directed class pair a -> b:

`C_{a->b}^M = P(\hat y=b | y=a, M)`

`Delta C_{a->b} = C_{a->b}^{BCL} - C_{a->b}^{LC-only}`

Report Head->Tail and Tail->Head separately. For each direction, group the
predefined pairs by CIFAR-100 coarse-superclass membership and compute:

`DID = E[Delta C | SameSuperclass] - E[Delta C | DifferentSuperclass]`

Also report pairwise raw-logit and logit-adjusted margins for the same pairs.
If DINO Top-K union is available, report it as a separate exogenous relation
slice; do not merge DINO and coarse-superclass labels into one outcome-derived
overlap score. Report pair counts alongside every DID. A one-seed DID is
descriptive only; do not claim seed-level statistical significance.

### 8.4 Mechanism-aligned Oracle gate

Do not execute an Oracle in this first one-seed reconnaissance.

Only after additional paired seeds show repeatable, mechanism-consistent evidence
across gradient direction/norm, margins, directional confusion, and
Overall/Many/Medium/Few performance may one Oracle be selected to test the
specific mechanism supported by the evidence:

1. **Prototype Oracle:** replace W-derived prototypes with balanced reference
   feature prototypes. This isolates prototype reliability.
2. **Relation Oracle:** suppress or down-weight same-superclass negative
   prototypes in the contrastive denominator. This isolates semantic relation
   mismatch.
3. **Gradient Oracle:** remove only the BCL component conflicting with LC:
   `g_BCL* = g_BCL - min(0, g_LC^T g_BCL) / (||g_LC||^2 + eps) * g_LC`.
   Test the classifier-W and backbone scopes separately if this Oracle is
   justified by the observed gradient evidence.

These are mechanism-aligned diagnostic Oracles, not an unconstrained
performance upper bound. Do not run all three as a fishing expedition. Stop if
the selected Oracle has no stable benefit or improves one class group only by
sacrificing another. No numeric go/no-go threshold was pre-registered; do not
invent a threshold after seeing the results.
