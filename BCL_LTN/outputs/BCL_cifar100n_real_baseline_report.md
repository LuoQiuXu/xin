# BCL_cifar100n_real_seed42 — noisy-label baseline report

## Protocol
- CIFAR-100-LT IF=100; fixed 10,847-image TRAIN subset; no resampling.
- Regime: `cifar100n_real` (official CIFAR-100N human fine noisy_label).
- Original BCL ResNet-32, three views, FP32, batch 256, 200 epochs.
- Objective: 2.0 × LC + 0.6 × BCL; LC prior and BCL class-count argument use observed noisy-label counts.
- SGD 0.15, momentum 0.9, weight decay 0.0005, 5-epoch warmup, decay ×0.1 after epochs 160 and 180.
- One balanced TEST evaluation after all 200 epochs and the five TRAIN diagnostics.

## Noise realization
- Requested rate: N/A (real-world human annotation noise)
- Realized rate: 0.359731
- Flipped: 3902 / 10847
- Index SHA256: be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7
- TRAIN and meta source SHA256: {"train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b", "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"}
- Noise manifest SHA256: 761d8bd3e11c957c16e4072f1fc0436b966f1e27b914288e99959db4aae4dff5
- Noisy-label SHA256: a19cd32002d581a832ca314e2b7d0561a9e7e71cc8c6cad68141c81e45cb7d5c
- Full clean and observed noisy class counts: D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\class_counts_comparison.csv
- Clean-label SHA256 (offline only): 1244d78d646ff1a4e10c6913c000f83e1ea39eaa95b31455d2a2702e2084ffc9
- Mask SHA256 (offline only): 909ef4a47f9d5b8f9103131e7e137bd6d6fd8b01cf6ceb9f965b9d9062951d58

| Clean-defined group | Classes | Clean source samples | Flipped source samples | Source flip rate | Observed destination labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Many | 35 | 8824 | 3066 | 34.75% | 7007 |
| Medium | 35 | 1718 | 695 | 40.45% | 2606 |
| Few | 30 | 305 | 141 | 46.23% | 1234 |

## Five-checkpoint TRAIN diagnostics
Numerators and denominators by original clean source group are in D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\train_diagnostics.csv.

| Epoch | Clean overall | Clean Many | Clean Medium | Clean Few | Memorization overall | Memorization Many | Memorization Medium | Memorization Few |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 48.80% | 53.28% | 28.74% | 16.46% | 10.71% | 10.83% | 10.07% | 11.35% |
| 100 | 58.01% | 58.61% | 57.97% | 37.20% | 18.96% | 19.54% | 17.27% | 14.89% |
| 160 | 61.93% | 64.92% | 47.70% | 45.73% | 19.20% | 18.88% | 20.14% | 21.28% |
| 180 | 80.48% | 80.91% | 80.55% | 64.63% | 36.31% | 36.79% | 34.68% | 34.04% |
| 200 | 83.86% | 84.86% | 80.65% | 68.90% | 38.85% | 39.37% | 36.69% | 38.30% |

## Final TEST (raw logits)

| Metric | Noisy BCL | Clean BCL reference | Delta (pp) |
|---|---:|---:|---:|
| Overall | 39.73% | 50.76% | -11.03 |
| Many | 50.23% | 66.54% | -16.31 |
| Medium | 41.14% | 51.74% | -10.60 |
| Few | 25.83% | 31.20% | -5.37 |

## Runtime and provenance
- Training seconds: 3385.3
- Max CUDA allocated: 1.878 GiB
- Max CUDA reserved: 2.242 GiB
- Actual training optimizer updates: 8600
- Initial model state SHA256: 11c52fe9e2b082b0c36ef025f19ae2d963dbc19c456e0d5aa17aac14fffdf440
- Environment: {"python": "3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]", "python_executable": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.14.1+cu132", "torchvision": "0.29.1+cu132", "cuda": "13.2", "cudnn": 92400, "device": "cuda:0", "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "gpu_total_memory_bytes": 8546484224, "gpu_free_before_model_bytes": 7338983424, "amp": false, "tf32": false, "seed": 42, "precision": "fp32", "batch_size": 256, "epochs": 200, "indices_sha256": "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7", "noisy_labels_sha256": "a19cd32002d581a832ca314e2b7d0561a9e7e71cc8c6cad68141c81e45cb7d5c", "noise_manifest_sha256": "761d8bd3e11c957c16e4072f1fc0436b966f1e27b914288e99959db4aae4dff5", "train_source_sha256": {"train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b", "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"}, "source_sha256": {"BCL_LTN/CODEX_RUN.txt": "3cbf3f3924c3f21f6245e6d998ffedd5a61a0d36fc08c7268aa20aeedbc6313f", "BCL_LTN/make_noisy_lt.py": "655f56facc3b31ab93c64c035beeec232014cad1474692a7c3e5fb0643e64a82", "BCL_LTN/train_noisy_bcl.py": "dcc84755f12c36b51b0468fa172c09c5b833ef5e138f1634ae3c4cf0634ee859", "BCL_AUDIT/cifar100lt/data.py": "216bda58acd66cbb1decf606d9bde9bc67d060bb6f4d18a72a5982fb7e9bb735", "BCL_AUDIT/cifar100lt/losses.py": "31af83140b3b12dd74686ff7b88e2848553c2465f21b3f03c4956a779782b4ae", "BCL_AUDIT/cifar100lt/models.py": "2dc10c65e696cc772e6df95e1f782ce36a59aaec6a85eebb73d946d83fcc29cc", "BCL_AUDIT/cifar100lt/train.py": "b1ba2b95369dc8a709bff56775d2a80cc7dcd3c03ae2c66f623e8b0126b21ec6"}, "regime": "cifar100n_real", "smoke_report_sha256": "e3f9b7c0f98bc417f75596b32687521ffa2741f29dc763e5e0b0a165bb1e6992"}
- Checkpoints: D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\checkpoints\epoch_040.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\checkpoints\epoch_100.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\checkpoints\epoch_160_pre_decay.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\checkpoints\epoch_180_pre_decay.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_cifar100n_real_seed42_20261127\BCL_cifar100n_real_seed42\checkpoints\epoch_200.pth
- Checkpoint file SHA256: {"epoch_040.pth": "37cf16158ada495d2d552e57d99428a7fe7c47cc8c2437339ac5d59f7f40647f", "epoch_100.pth": "09db62daec6a9d93dba80cd09a0d55fc17f8847885e5946067348c68c6a6b573", "epoch_160_pre_decay.pth": "c36c890abefc707f12556a9031940a135e8d8f814c6ca3c61fa372d45e2791fc", "epoch_180_pre_decay.pth": "35d3db72523b632a53145f87bece3c49bb8ea3937b8fe121e9dd8c70a41a760e", "epoch_200.pth": "77c42aaff3e119246f113250fac85d0ed11df105177754fa266b4ea7a2de2992"}

## Leakage and split audit
- Training batches expose only three views, observed noisy labels, and original source indices.
- Saved clean labels and the mask are read for input integrity checks and separate eval-mode TRAIN diagnostics.
- Both loss functions and the LC prior use observed noisy labels/counts only.
- TEST pickle opens before the final gate: 0. Final TEST evaluator passes: 1. TEST pickle opens during evaluation (integrity check and load): 2.
- Training completed with the fixed FP32 batch and no OOM, retries, or protocol deviations.
- This runs the original BCL algorithm unchanged on regime `cifar100n_real` as the paired non-robust baseline.
