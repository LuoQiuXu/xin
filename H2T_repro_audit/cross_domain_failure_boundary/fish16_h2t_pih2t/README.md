# Fish16 loading adaptation and smoke validation

Branch: `experiment/fish16-h2t-pih2t-cross-domain-20261031`.
Interpreter: `D:\app\minicondaaa\envs\xiaoju\python.exe`.

The latest user instruction authorizes the original 19c8e357 initialization,
CPU tests and real-data smoke only. **Do not start the formal 90/10-epoch pilot**.
There is no formal-training or TEST-evaluation entry point in this directory.

## Original-source loading issue and approved adaptation

The specified upstream file defines `ResNet_FE` and `resnet50_fe`, but no
`ResNet50_FE` symbol. Its `resnet50_fe(pretrained=True)` passes `resnet50_fe`
into a URL table that contains only `resnet50`, producing a verified KeyError.
The upstream source remains unchanged.

The approved adapter allocates the **original**
`ResNet_FE(Bottleneck, [3, 4, 6, 3])` through its existing factory, then loads
all backbone tensors strictly before any inference. It returns the 2048-channel
layer4 map. No torchvision backbone is substituted.

Official URL:
`https://download.pytorch.org/models/resnet50-19c8e357.pth`.
Downloaded bytes: 102,502,400.
SHA256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`.

Independent cache:
`D:\h2t_cross_domain_results\fish16_weight_cache_20261031_01\`.
The existing `resnet50-11ad3fa6.pth` is never used or overwritten.

The official file uses legacy tar serialization, unsupported by the current
`weights_only=True` reader. Only after an exact pinned SHA check is this one
trusted official file read with the legacy compatibility loader. Validation
then establishes:

- All 265 backbone parameter / floating-buffer tensors match by key and shape.
- Only `fc.weight` (1000×2048) and `fc.bias` (1000) are discarded for the feature
  extractor; the Fish classifier has 16 outputs.
- The 53 missing keys are exclusively legacy `num_batches_tracked` counters.
  PyTorch's strict versionless-BN compatibility rule supplies zero int64
  counters. No learned weight or floating BN statistic is filled or invented.
- `load_state_dict(strict=True)` succeeds without missing/unexpected keys, and
  every source backbone tensor is exactly equal after loading.
- A new tensor-only state file is written without replacing the downloaded
  original. All subsequent model construction uses `weights_only=True` and
  validates this file's recorded SHA.

`ADAPTER_PROVENANCE.json` records these facts and both hashes. Its tensor-only
file SHA is
`c1bc33a75ef4d4dd607aa1cb0635ed42d547265a41ad0467081276e5009a7b3e`.
Plain and PIF models initialize identical backbone and classifier tensors;
PIF adds exactly the existing two shared Conv1d parameters after layer4, before
GAP. The existing default PIF initialization is retained, with RNG isolation
so it cannot perturb common head initialization.

## Preserved operations and controlled adaptation

H2T calls the immutable official `methods.H2T`, rho=0.25, with the original
batch-shared random channel replacement (512/2048 channels). A saved CPU RNG
state previews the same indices for logging and is restored before the native
operation, so logging consumes no extra draw.

PIF is imported from the immutable review `PermutationInvariantFusion`.
PI-H2TF calls the existing review implementation with current classifier
weights, `paper_theoretical_bounds`, and `ratio_detach=True`. This is the
specified local review reconstruction, not newly verified PI-H2T author code.

Stage-II freezes extractor parameters and the complete BN state for all four
conditions and only updates the classifier. This explicitly differs from
original H2T configurations with `shift_bn=True`. Complete extractor state
hashes before/after each phase verify the freeze.

The locked manifest is never regenerated. Recipient/donor sampling uses a
separate saved common schedule, with recipients uniform over classes and donors
uniform over natural train instances, both with replacement. No Head-only,
different-class, different-image or different-track restriction is imposed.
Image transforms use per-position seeds so all matched recipients, and both
fusion donor streams, are identical. Every batch records actual image-tensor
digests outside the repository; Stage-I also records MixUp RNG and lambda.

The literal `ColorJitter(0.4)` config means brightness jitter only. Train is
RandomResizedCrop(224), horizontal flip, that jitter, ToTensor and ImageNet
normalization; validation is Resize(256), CenterCrop(224), ToTensor and the same
normalization. Both image-weighted and class-macro Head/Medium/Tail validation
statistics are labelled explicitly.

## Validation scope and commands

`preflight.py` preserves the original failed loading-contract diagnosis while
checking all 27,133 image paths/SHAs and exact split/group support. That original
native-loader failure remains a valid historical diagnostic; the user-approved
adapter resolves loading in this new directory.

CPU tests:

    D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -m unittest discover -s H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t -p "test_*.py" -v

Smoke entry (requires a new, nonexistent external output directory):

    D:\app\minicondaaa\envs\xiaoju\python.exe -B -X utf8 -u H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py --output-dir D:\h2t_cross_domain_results\fish16_smoke_seed42_20261031_01 --micro-batch 16

The smoke first discards four real-batch Stage-II gradient tests. Then it runs
one complete train epoch for each Plain/PIF Stage-I branch, evaluates A0/B0 on
validation, and runs one epoch each for A1/A2/B1/B2 from matched branch-specific
smoke checkpoints. It does not reuse batch-test updates for these epochs.
Stage-I uses the pilot's first warmup LR 0.002; Stage-II uses the fixed
10-epoch schedule's first LR 0.01. Neither schedule is rescaled to one epoch.
Effective batch=16, AMP, four workers; all 16,269 train samples/recipient draws
are processed. No TEST model evaluation, bootstrap inference or method-effect
claim is performed at this validation stage.

Logs and CPU validation:
`D:\h2t_cross_domain_results\fish16_validation_seed42_20261031_01\`.
Smoke results/checkpoints remain outside Git. A CUDA OOM must be retained and
the process stopped; the only authorized retry is a fresh complete process and
new output directory with micro-batch=8, accumulation=2 for every condition.
Other errors/NaNs stop the smoke and preserve `FAILURE.json`.
