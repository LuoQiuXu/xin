# C3 — Exclude direct Many-class feature perturbation

## Hypothesis and scope

C1 improved Few accuracy but reduced Many accuracy compared with C2. C3 tests whether **direct perturbation of Many-class inputs** contributes to the Many loss. It does not prove Many-class representation or semantic richness, nor rule out indirect effects through the shared Softmax classifier.

Use the exact original C1 class-conditional normalized adversarial feature gradient and its original class weight w_c, ADVER_REG=1.0. C3's only intervention is:

    delta_i^C3 = 0           if train_count[y_i] > 100
                 delta_i^C1  otherwise

Counts are taken from the sealed long-tail TRAIN split in prepared/prepared_manifest.json. The threshold is **strictly >100**, matching the original Many metric, not a hardcoded class ID range. Medium/Few classes keep C1's perturbation bitwise unchanged.

Retain C1's exact objective, including weighted CE on **all** examples:

    L = alpha * mean_i[w_yi CE(fc(f_i + delta_i), y_i)]
        + (1-alpha) * mean_i[CE(fc(f_i), y_i)]

    alpha = min((epoch+1)/30, 0.3), 0-based epoch

For Many examples, weighted adversarial CE becomes weighted clean CE. Many examples remain in both loss terms; the shared classifier still receives gradients from all classes. C3 should not alter the feature map, feature extractor, class weights, alpha, sampler, classifier optimizer or learning-rate schedule.

## Sealed original protocol

- Repository: LuoQiuXu/xin
- C3 branch: experiment/ccsam-c3-no-many-perturb-20261108
- Baseline commit: f916957ea4a7b90cf82dc5839eb193e044d881dd
- Experiment directory: H2T_repro_audit/ccsam_stage2_controlled_cifar100_20261106
- Clean Stage I seed42 epoch184, checkpoint SHA256 7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88
- CIFAR-100-LT IF=100, 10,847 long-tail TRAIN images
- Validation is 1,980 images from official TRAIN, classes 1..99 with 20/class; class 0 absent
- Official CIFAR-100 TEST is strictly forbidden throughout preparation, Smoke, training and analysis
- Stage II seeds 123,124,125; all share the same frozen Stage I representation
- Batch size 64; 30 epochs; 170 batches per epoch; 5,100 steps per seed
- SGD(lr=0.01, momentum=0.9, weight_decay=0.0002, nesterov=False), constant LR
- Deterministic crop/flip/normalize and class-aware sampling plans reused exactly
- FP32, no AMP, no Mixup, no gradient accumulation; frozen backbone/BN buffers
- Original C0/C1/C2 results remain untouched: 9 completed runs and all summary artifacts
- C3-only new directories: results/seed_123/C3_NO_MANY_PERTURB and corresponding seed_124/125
- Original batch_trace.csv and classifier checkpoints are git-ignored, so they must remain available locally
- The published original checksum index is formal_launch_20261106_01/ARTIFACT_SHA256.csv

## Acceptance gates and execution order

1. Run deterministic synthetic CPU tests with test_c3_semantics.py. Verify strict Many threshold including count=100 versus 101, duplicate labels, all-Many and all-nonMany equivalence, and gradient finiteness.
2. Run train_c3.py --preflight. It must find the original nine local runs and verify all original results against the previously published SHA256 index; check the sealed Stage I checkpoint, prepared arrays, transform and sampling plans.
3. Run train_c3.py --smoke-only. For each of three seeds, evaluate real batch0 and batch1 using the original continuous augmentation RNG, compare C3 with C1/C2 under exactly identical feature/head states, verify C3 Many delta is exactly zero and non-Many delta bitwise equals C1, check autograd reference and one disposable classifier SGD update. No formal training.
4. Only after all prior gates pass, run train_c3.py --formal. Train exactly C3 for three seeds ×30 epochs = 90 new epochs, 15,300 new optimizer steps. Original C0 per-batch trace hashes must match at every step. No rerun, no automatic overwrite, no resume or changing hyperparameters.
5. Run analyze_c3.py only after the three seeds complete. Use fixed epoch 30, paired C3−C1, C3−C2, C3−C0, original C1−C2 comparisons and per-class descriptive changes. No best-epoch cherry-picking. Archive results and source diff, preserving original SHA checks.

## Interpretation

C3−C1 estimates the effect of **removing direct Many perturbations**. C3−C2 estimates the **remaining non-Many perturbation effect**. Head recovery with Few benefit retained supports exploring selective perturbation; no recovery suggests shared classifier boundary effects or non-Many perturbation may dominate. Do not assume that head-class semantic richness is proven by this experiment.

Three Stage-II seeds share a single frozen backbone. The TRAIN-derived validation partition was used in development and is NOT an independent official TEST set. Do not assert statistical significance or paper-level novelty from this single contrast.

## Command sequence

Use Windows PowerShell; after switching to C3 branch, enter the experiment directory.

    & "D:\app\minicondaaa\envs\xiaoju\python.exe" -m unittest -v test_c3_semantics
    & "D:\app\minicondaaa\envs\xiaoju\python.exe" train_c3.py --preflight
    & "D:\app\minicondaaa\envs\xiaoju\python.exe" train_c3.py --smoke-only
    & "D:\app\minicondaaa\envs\xiaoju\python.exe" train_c3.py --formal
    & "D:\app\minicondaaa\envs\xiaoju\python.exe" analyze_c3.py

Do not execute original train_ccsam.py --formal, prepare_ccsam.py, or original analyze_ccsam.py again. Those belong to sealed earlier experiments.
