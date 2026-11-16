# CC-SAM cross-dataset execution

Recorded 2026-11-16T12:04:28+08:00.

- Branch: experiment/ccsam-cross-dataset-formal-20261112
- Git HEAD: a3977ae07f3fd33c1cac9673909a300792818c7f
- Implementation files with worktree changes: 15
- The per-run configs contain exact SHA256 hashes for the implementation and protocol.
- Python: 3.12.12. PyTorch: 2.14.1+cu132. torchvision: 0.29.1+cu132.
- PyArrow: 26.0.1. CUDA: 13.2. cuDNN: 92400.
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU. Device memory: 8546484224 bytes.
- CPU unit tests: Ran 22 tests · exit 0
- Syntax check: Compiled 18 Python files · exit 0
- Preflight: completed. CUDA self-check: completed. Smoke: completed. Final evaluation: failed.
- Official test split access started: True.

Worktree source changes at execution time

- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_data.py
- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_models.py
- M H2T_repro_audit/ccsam_cross_dataset_formal_20261112/test_cross_dataset_data.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/analyze_results.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cross_dataset_smoke.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/cuda_statistics.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/diagnostics/amp_gradient_probe/gradient_evidence.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/evaluate_final.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/experiment_common.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/extract_features.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/prepare_experiment.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/run_cross_dataset.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/test_pipeline.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/train_stage1.py
- ?? H2T_repro_audit/ccsam_cross_dataset_formal_20261112/train_stage2.py

## Execution stopped

ValueError: ImageNet-LT test Parquet does not match official manifest membership; missing example ('ILSVRC2012_val_00029930.JPEG', 0)

Final evaluation error — ValueError: ImageNet-LT test Parquet does not match official manifest membership; missing example ('ILSVRC2012_val_00029930.JPEG', 0)

- imagenet_lt Smoke completed, Stage-I batches completed 2, real autograd equivalence True.

- places_lt Smoke completed, Stage-I batches completed 2, real autograd equivalence True.

## Stage I

| Dataset | Seed | State | Completed epochs | Checkpoint SHA256 |
| --- | ---: | --- | ---: | --- |
| imagenet_lt | 123 | completed | 90 | eca33a8df37074b3a6b23b135dde0ed6b62114fa208754e8607a7fd7421a054c |
| imagenet_lt | 124 | completed | 90 | 7ec0f9b8a4e6da14ade93f12a28d731f47086987d61b6acb49247180fb1b2bae |
| imagenet_lt | 125 | completed | 90 | a22fb0b48b5da48db6b195ea089a76b6bdace16be9a89cd068329255bd13d67c |
| places_lt | 123 | completed | 90 | 2147090d49e6f6e3b3bfdbaaa55f096e9d4cf190b2e668e24cef577dd6f7e439 |
| places_lt | 124 | completed | 90 | b6195fa59b177ab6bc36bff72f768c930c1a4db156e5f0f7dbaa630941b0790f |
| places_lt | 125 | completed | 90 | 03ee8235207596dbc5d51b74fdb98719c06ed936cf4b2bc832f0a69ef083e3e5 |

## Stage II

| Dataset | Seed | Arm | State | Completed epochs | Checkpoint SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| imagenet_lt | 123 | D0 | completed | 30 | 2ce8bfe579e7b34513c87b6f9fb8efe4d6c2be7c639fc10b617a6a12c4398e5d |
| imagenet_lt | 123 | D1 | completed | 30 | e35eff2912fd26d9d0db9f0623bc027e667f27168d64e2c5e7be86a82a853bbd |
| imagenet_lt | 123 | D2 | completed | 30 | 98fd9f8780a7f3dfeb2e0642ccf4d666ff4228e8aad9dd0fa7965602eb3a97ec |
| imagenet_lt | 124 | D0 | completed | 30 | e40502ff82103f851d25ac9bd4e1bffa02cac57daedc4c33774c335ac4450a54 |
| imagenet_lt | 124 | D1 | completed | 30 | 6af6f4dac9b282c9250d7886f92d2af8ec977514be7e6428790105d78b1790ea |
| imagenet_lt | 124 | D2 | completed | 30 | dbf34dce2c42bb72bdef924a91464d8e9c38c10875c11e0366d629085897dde6 |
| imagenet_lt | 125 | D0 | completed | 30 | cf4b9b93a37c38bbdf1f997cc9cdb6b8f31ec7829f9ca6925d52d2f2a8bd81df |
| imagenet_lt | 125 | D1 | completed | 30 | d2d763d1d59bd04bb8ee98db413fc2489661de3cd9cd0decdf5f7d3717533982 |
| imagenet_lt | 125 | D2 | completed | 30 | 0b7f5ddaa80ec5cb1f0f926b91249de6ecaf470e0ea713da0e8686eb4a235661 |
| places_lt | 123 | D0 | completed | 30 | 822e663e6d733493ed542a5aa1853b6ac59ef4ea151de429c2e178018183f820 |
| places_lt | 123 | D1 | completed | 30 | ef4ede6897209f40b77a1e5601221012a4b738fd9cb293a5193c40d75d54072f |
| places_lt | 123 | D2 | completed | 30 | a278547a62daa72d7982b754c28df3979ebc0ed08cd57ecdfcf11f28883043e6 |
| places_lt | 124 | D0 | completed | 30 | c525362e7f32386ef15afe42cb67761db4ecaaf0be696d956635aea92356bb05 |
| places_lt | 124 | D1 | completed | 30 | 3612c9c0bf20ed8c08fed2d989d1bf5783799dac62784458ebe0709d71d1fbf9 |
| places_lt | 124 | D2 | completed | 30 | 675c6313efa14e7f3f68441b89e1c6c94c3cd26b6d56eb1c4754a94244e436da |
| places_lt | 125 | D0 | completed | 30 | a442bdcd63ba681de0d7306f461ccb97c98b775846bda708d8f139b121d95290 |
| places_lt | 125 | D1 | completed | 30 | bdfb0f3f497a2d2c1074ba00e684454995929f068019de11baaec425f0b1f34f |
| places_lt | 125 | D2 | completed | 30 | 88f752499a76bc75cf079a2b98bb7c012a13d8568538abb9cd7f2862a69ac3a0 |

Maximum recorded PyTorch CUDA allocation: 1,079,101,952 bytes.

All formal checkpoint checks completed at 2026-11-16T12:04:25+08:00.
- imagenet_lt access started 2026-11-16T12:04:26+08:00, completed None, evaluated rows None.
