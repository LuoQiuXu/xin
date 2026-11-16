# CC-SAM cross-dataset implementation

This implementation follows the frozen ImageNet-LT and Places-LT protocol. It keeps the existing D0, D1 and D2 loss module and checks the official CC-SAM sampler SHA256 before loading its source. Each of the six seed-and-dataset pairs has one Stage-I encoder and classifier, one ordered TRAIN feature cache, and one continuous 30-epoch sampling plan.

## Dataset adapter corrections

- Places-LT uses the official single-letter path component and joins nested category components with hyphens. The adapter reads only the official manifest rows from the selected split.
- The 115,846 ImageNet-LT TRAIN filename and original label pairs match the official TXT. The Parquet shards store those pairs in a different physical order. The adapter projects the filenames and labels, verifies exact membership, and exposes them in official TXT order. Image payloads are loaded lazily in process-local Arrow row groups.
- The Windows reader closes Parquet file handles explicitly. The initial CPU suite exposed WinError 32 during fixture cleanup. The failure log and fixture were preserved under validation/.

## Execution and verification

- Stage I uses torchvision ResNet-50 with weights=None, the frozen transforms and MixUp, and a fresh RNG and optimizer for each seed. The warm-up LR reaches 0.01 at epoch 5 and the cosine schedule reaches 0 at epoch 90. CUDA GradScaler uses its standard skip and scale-update behavior in formal Stage I. The Smoke gate requires finite unscaled gradients.
- Cached TRAIN features are extracted in eval mode in FP32. Stage II reads the shared float32 memmap and checks the sealed classifier, plan, and actual per-epoch index stream. The D2 teacher is loaded from the matching verified D0 epoch-30 checkpoint. Smoke uses a separate disposable D0 update to test the same teacher provenance before any formal checkpoint exists.
- The final evaluator verifies all six Stage-I checkpoints and all eighteen Stage-II records and artifact hashes before constructing either official test dataset. It computes fixed Stage-I and Stage-II predictions from a single loader pass per dataset, followed by descriptive paired analysis.
- Run directories are exclusive. Failures retain their logs, status, and partial files. FORMAL_EXECUTION_REPORT.md records actual execution outcomes and links them to the implementation source hashes.

## November 12 execution and repair

- Preflight exited successfully with 115,846 ImageNet-LT TRAIN rows, 62,500 Places-LT TRAIN rows, and six verified int32 sampling plans. The preflight certificate records the exact source hashes used at that time.
- The first Smoke command exited with code 1 before the first batch. The runner tried to reset peak CUDA memory statistics before initializing the PyTorch CUDA allocator. The traceback and the partial Smoke config are preserved. The allocator had recorded no peak measurement at that point.
- The run recorder now creates the initial status and log, initializes CUDA, and then resets the peak counter. The recorder also captures initialization errors before entering the workload.
- After the repair, all 15 Python files compiled and all 17 CPU tests passed, including mocked tests for the CUDA call order and failure recording. Real GPU verification of the repair is pending. The first Smoke exit stopped this execution. Stage I, Stage II and final TEST evaluation have zero completed runs.
- The saved preflight certificate belongs to the initial source snapshot. A fresh execution must seal the repaired source and preserve the first attempt's artifacts. The repair affects startup bookkeeping. The frozen protocol and D0/D1/D2 objective source match Git HEAD.
