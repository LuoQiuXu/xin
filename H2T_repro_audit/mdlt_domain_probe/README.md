# OfficeHome-MLT donor-domain mechanism pilot

This directory implements only the protocol in
[CODEX_OFFICEHOME_DOMAIN_TRANSFER_PILOT.txt](CODEX_OFFICEHOME_DOMAIN_TRANSFER_PILOT.txt).
It does not import or change any CIFAR experiment or official H2T source.

## Data provenance and leakage controls

- MDLT commit: 2efbfefd34542e365293f798d79f70cee5e54303.
- Official split: https://raw.githubusercontent.com/YyzHarry/multi-domain-imbalance/2efbfefd34542e365293f798d79f70cee5e54303/mdlt/dataset/split/OfficeHome.csv
- SHA256: 0a6c4e7800f629b8c6b67ba71dd238a83d4f513ac77694ab58796e528985a5b5.
- protocol/OfficeHome.csv is the byte-identical pinned CSV. The user's
  data/OfficeHome.txt has this same hash. The image root can retain its extracted
  OfficeHomeDataset_10072016 name; paths are joined against this explicit root.
  Images and the user's original metadata are never renamed or modified.
- All four domains appear in all splits. This is multi-domain cell imbalance,
  not unseen-domain OOD generalization.
- Audit reads CSV labels as metadata for counts/disjointness but performs only
  filesystem stat calls on test paths. No test images are decoded.
- OfficeHomeImages refuses to construct a test dataset until all 12 seed x arm
  epoch-10 checkpoints exist with matching sealed SHA256 hashes.
- Stage I selects only val domain-class macro accuracy, with earliest ties.
  Stage II has no best checkpoint selection: epoch 10 is the primary endpoint.
- No CIFAR TEST or CIFAR checkpoint is used.

## Fixed model, optimization and feature protocol

ImageNet-1K V1 pretrained torchvision ResNet-18, 65-way linear FC. There is no
random-backbone fallback. The 512-dimensional GAP representation is used.

Stage I: seeds 42,43,44; 5001 successful Adam updates, lr=5e-5, weight_decay=0;
4 images per domain per batch, total batch16; AMP. Evaluate every 100 updates
and at update5001. BN running_mean/running_var/num_batches_tracked stay fixed;
BN affine parameters are trainable, matching MDLT's actual m.eval() semantics.
Default GradScaler overflow retries reuse the exact already-augmented batch;
skipped attempts are logged and do not count as successful updates.

The pinned MDLT source uses RandomResizedCrop(224,scale=(0.7,1.0)),
RandomHorizontalFlip(), ColorJitter(0.3,0.3,0.3,0.3),
RandomGrayscale(), ImageNet normalization. Eval uses Resize((224,224)),
not shortest-side-only Resize(224). Source provenance hashes are saved in
protocol/reference_provenance.json. The full local reference source is ignored
under output/reference_source; the corresponding pinned URLs remain reproducible.
The task's ResNet18 and batch4/domain intentionally differ from MDLT's default
ResNet50 and batch24/domain; no claim of full MDLT training reproduction is made.

Stage II: eval-transform fixed float32 train/val GAP cache; frozen entire
backbone and BN; only independent copies of the Stage I classifier are trained.
SGD lr=.01, momentum=.9, weight_decay=.0005; cosine T_max=10; ten epochs;
feature batch128; each epoch has N_train receiver draws, floor/ceil counts per
class, uniform instance draws within a class. AMP is used for Stage I image
training; these small cached-feature FC updates use float32.
This is a fixed-feature mechanism pilot, not an image-augmentation-exact H2T
reproduction. Complete-channel substitution commutes with GAP; a CPU test
checks this on [B,512,7,7] maps.

## Four paired arms

1. DR-NoFusion: class-balanced recipients, no replacement.
2. H2T-Global: natural global-train instance donor sampling.
3. H2T-DomainMatched: same global donor class; uniform instance donor from the
   receiver's domain and that class. Empty pool alone triggers fallback to the
   corresponding Global donor.
4. H2T-CrossDomain: same global donor class; uniform instances of that class in
   the union of all other domains. Empty pool alone triggers the same fallback.

Every actual minibatch shares one uniform, without-replacement 128/512 mask,
rho=.25. All arms share recipients, donor classes, batch order, mask plan,
initial FC and optimizer settings. Receiver labels are always the targets.
DR has virtual donors only for paired-plan diagnostics; it performs no fusion.
Independent NumPy SeedSequence streams isolate receiver, global, conditional
donor and mask random draws. This matches original H2T's batch-shared mask
granularity and substitution formula, not its precise torch.randperm RNG path.
Plans and actual training control hashes are saved; no donor/class filtering,
new loss, LowVS, rho search or post-test adaptation is implemented.

## Commands (PowerShell)

Use the user's interpreter for every command. All outputs must be fresh unique
directories under this experiment directory. Existing output directories are
rejected, including partially completed runs; nothing is silently resumed.

    $py = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
    $project = 'D:\xiangmuxunlian\xin'
    $module = "$project\H2T_repro_audit\mdlt_domain_probe"
    $images = "$project\data\OfficeHomeDataset_10072016"
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $py -m compileall -q $module
    & $py -m unittest discover -s $module -p 'test_officehome_domain_probe.py' -v
    & $py "$module\officehome_domain_probe.py" audit --data-root $images --out "$module\results\metadata_UNIQUE"
    & $py "$module\officehome_domain_probe.py" smoke --data-root $images --out "$module\results\smoke_UNIQUE"
    & $py "$module\officehome_domain_probe.py" run --data-root $images --smoke-dir "$module\results\smoke_UNIQUE" --out "$module\results\pilot_UNIQUE"

The smoke is seed42, Stage I exactly20 successful updates and at most2 batches
per Stage II arm; it evaluates/caches train and val only. Its timing is a rough
hardware estimate, with worker startup costs. Full run requires a passing
same-source, same-data smoke. Full run always uses all three prescribed seeds.
There are no flags to tune seeds, rho, epochs, loss, backbone or learning rates.

## Outputs and interpretation

Each run contains config.json, environment.json, dataset_audit.json, domain/cell/
class counts, status.json and execution.log. Each seed contains stage1_epochs.csv
(update-based), stage1_summary.json, stage2_epochs.csv, state_audit.json,
feature_manifest.json, sampling_plan_audit.json, donor_diagnostics.csv and
donor_fallbacks.csv.

Local output/ subdirectories hold checkpoints, GAP caches, plans and predictions;
they are ignored by Git. Small results are retained outside output/.
Full-run final_checkpoints.lock.json is created before test_access.json and
before any test dataset construction. Test then runs once per fixed final
classifier; there is no CLI to pick a test-performing checkpoint.

Main result files: epoch10_summary.csv, epoch10_mean.csv,
validation_epoch10_summary.csv, per_domain.csv, domain_class_metrics.csv,
worst_cells.json (all ties, with support), paired_changes.csv,
donor_diagnostics.csv and final_report.md.

Accuracy values are percentages; delta values are percentage points.
Domain-class Many>=60, Medium20..59, Few<20 use TRAIN cell counts.
Group accuracy is weighted by evaluation images; val/test have respectively
5/10 images per cell, so group cell-macro and image-weighted values coincide.
DomainClassMacro averages the 260 evaluated cells equally.
These definitions differ from CIFAR class-shot metrics. Worst cells with a
small test support must be interpreted cautiously.

Report per seed and three-seed equal means. Reusing test images across seeds
does not create independent image-level training replicates. The predeclared
direction check is Cross<Matched on Few or macro in at least2/3 seeds, with
all paired controls verified. Report Global-DR Few direction separately.
Even a consistent difference does not establish a useful new module.
If directions are mixed/small, say evidence is insufficient and stop.
No automatic follow-up experiment or hyperparameter search is authorized.
