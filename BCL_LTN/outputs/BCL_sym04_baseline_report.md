# BCL_sym04_seed42 — symmetric-noise baseline report

## Protocol
- CIFAR-100-LT IF=100; fixed 10,847-image TRAIN subset; no resampling.
- Independent Bernoulli(0.4) symmetric label noise, seed 42; replacement uniformly among other 99 classes.
- Original BCL ResNet-32, three views, FP32, batch 256, 200 epochs.
- Objective: 2.0 × LC + 0.6 × BCL; LC prior and BCL class-count argument use observed noisy-label counts.
- SGD 0.15, momentum 0.9, weight decay 0.0005, 5-epoch warmup, decay ×0.1 after epochs 160 and 180.
- One balanced TEST evaluation after all 200 epochs and the five TRAIN diagnostics.

## Noise realization
- Requested rate: 0.4000
- Realized rate: 0.403706
- Flipped: 4379 / 10847
- Index SHA256: be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7
- TRAIN and meta source SHA256: {"train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b", "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"}
- Noise manifest SHA256: 0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8
- Noisy-label SHA256: 691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f
- Full clean and observed noisy class counts: D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\class_counts_comparison.csv
- Clean-label SHA256 (offline only): 1244d78d646ff1a4e10c6913c000f83e1ea39eaa95b31455d2a2702e2084ffc9
- Mask SHA256 (offline only): 839fc0f0bbbbd0838df93542f2b3601893675c1eb7641d3808f6ee9742842b68

| Clean-defined group | Classes | Clean source samples | Flipped source samples | Source flip rate | Observed destination labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Many | 35 | 8824 | 3587 | 40.65% | 6752 |
| Medium | 35 | 1718 | 681 | 39.64% | 2559 |
| Few | 30 | 305 | 111 | 36.39% | 1536 |

## Five-checkpoint TRAIN diagnostics
Numerators and denominators by original clean source group are in D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\train_diagnostics.csv.

| Epoch | Clean overall | Clean Many | Clean Medium | Clean Few | Memorization overall | Memorization Many | Memorization Medium | Memorization Few |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 29.93% | 34.71% | 11.09% | 1.55% | 0.84% | 0.84% | 0.88% | 0.90% |
| 100 | 48.14% | 52.00% | 36.26% | 7.73% | 2.28% | 2.17% | 3.08% | 0.90% |
| 160 | 52.64% | 56.01% | 42.91% | 13.92% | 2.76% | 2.62% | 3.38% | 3.60% |
| 180 | 75.03% | 77.95% | 69.24% | 27.32% | 9.27% | 9.23% | 9.54% | 9.01% |
| 200 | 77.57% | 80.50% | 71.46% | 30.93% | 11.33% | 11.54% | 10.28% | 10.81% |

## Final TEST (raw logits)

| Metric | Noisy BCL | Clean BCL reference | Delta (pp) |
|---|---:|---:|---:|
| Overall | 32.22% | 50.76% | -18.54 |
| Many | 50.43% | 66.54% | -16.11 |
| Medium | 35.66% | 49.40% | -13.74 |
| Few | 6.97% | 31.20% | -24.23 |

## Runtime and provenance
- Training seconds: 4969.7
- Max CUDA allocated: 1.878 GiB
- Max CUDA reserved: 2.242 GiB
- Actual training optimizer updates: 8600
- Initial model state SHA256: 11c52fe9e2b082b0c36ef025f19ae2d963dbc19c456e0d5aa17aac14fffdf440
- Environment: {"python": "3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]", "python_executable": "D:\\app\\minicondaaa\\envs\\xiaoju\\python.exe", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.14.1+cu132", "torchvision": "0.29.1+cu132", "cuda": "13.2", "cudnn": 92400, "device": "cuda:0", "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "gpu_total_memory_bytes": 8546484224, "gpu_free_before_model_bytes": 7338983424, "amp": false, "tf32": false, "seed": 42, "precision": "fp32", "batch_size": 256, "epochs": 200, "indices_sha256": "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7", "noisy_labels_sha256": "691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f", "noise_manifest_sha256": "0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8", "train_source_sha256": {"train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b", "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"}, "source_sha256": {"BCL_LTN/CODEX_RUN.txt": "3cbf3f3924c3f21f6245e6d998ffedd5a61a0d36fc08c7268aa20aeedbc6313f", "BCL_LTN/make_noisy_lt.py": "655f56facc3b31ab93c64c035beeec232014cad1474692a7c3e5fb0643e64a82", "BCL_LTN/train_noisy_bcl.py": "6117d17e57579506bb5015ec2405cc8e38ca996d0759c5b84eb99df466962884", "BCL_AUDIT/cifar100lt/data.py": "216bda58acd66cbb1decf606d9bde9bc67d060bb6f4d18a72a5982fb7e9bb735", "BCL_AUDIT/cifar100lt/losses.py": "31af83140b3b12dd74686ff7b88e2848553c2465f21b3f03c4956a779782b4ae", "BCL_AUDIT/cifar100lt/models.py": "2dc10c65e696cc772e6df95e1f782ce36a59aaec6a85eebb73d946d83fcc29cc", "BCL_AUDIT/cifar100lt/train.py": "b1ba2b95369dc8a709bff56775d2a80cc7dcd3c03ae2c66f623e8b0126b21ec6"}, "smoke_report_sha256": "eeca2f4aa92b08c3f8b5d65eda5fe9d50ceb28e65622b9e8fc0e8737814387af"}
- Checkpoints: D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\checkpoints\epoch_040.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\checkpoints\epoch_100.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\checkpoints\epoch_160_pre_decay.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\checkpoints\epoch_180_pre_decay.pth, D:\xiangmuxunlian\xin\BCL_LTN\outputs\BCL_sym04_seed42\BCL_sym04_seed42\checkpoints\epoch_200.pth
- Checkpoint file SHA256: {"epoch_040.pth": "70e94a2df57049d5e68d0d0a04db489d0ccb23aab20db7f556b51124f904a0cc", "epoch_100.pth": "a8b852ae62f5bdc144df08789368e5b9046be5f29f4a355b3fcb9498db80c053", "epoch_160_pre_decay.pth": "10f0cb41db390f04a9d401433cc121396ab67fad7eb11dee0095c40350d75428", "epoch_180_pre_decay.pth": "d52f89a85337c73b1f22b5bc247dc8a982c03e33c7f80715a48d3189e1ab3fcb", "epoch_200.pth": "e58e465a9cde73de3c5b499553cc5239225fcd876b3783a205f4dd8117d9cac5"}

## Leakage and split audit
- Training batches expose only three views, observed noisy labels, and original source indices.
- Saved clean labels and the mask are read for input integrity checks and separate eval-mode TRAIN diagnostics.
- Both loss functions and the LC prior use observed noisy labels/counts only.
- TEST pickle opens before the final gate: 0. Final TEST evaluator passes: 1. TEST pickle opens during evaluation (integrity check and load): 2.
- Training completed with the fixed FP32 batch and no OOM, retries, or protocol deviations.
- This is synthetic instance-independent symmetric noise; it does not establish robustness to real-world annotation noise.
