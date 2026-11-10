# CC-SAM-TPR: low-rank tail parameter reallocation

This is an isolated experiment branch based on the completed C4 implementation.

## Branch and provenance

- Branch: `experiment/ccsam-tpr-lowrank-20261110`
- Base: C4 commit `4ab34ad00c86bcac60d765123f153b76ff097f47`
- Dataset: CIFAR-100-LT IF=100
- Backbone/features: frozen Clean Stage-I ResNet-32, 64-D GAP
- Stage-II seeds: 123, 124, 125
- Primary epoch: 30
- Official CIFAR-100 TEST: prohibited

## Files

- `TPR_PROTOCOL.md`: frozen method and execution protocol.
- `CODEX_CCSAM_TPR_20261110.txt`: complete Codex implementation/run task.
- `tpr_model.py`: isolated low-rank residual, dominant-subspace regularizer, confusion-pair selection/diversity penalty, and Many-only KD.
- `test_tpr_model.py`: CPU unit tests for the core model/loss semantics.
- `train_tpr.py`, `tpr_inputs.py`, and `analyze_tpr.py` implement the gated TPR run, sealed input checks, and fixed-epoch analysis.

## Local execution

From the TPR directory on the branch above, run the phases in order. Check each exit code before continuing.

```powershell
$TPRPython = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
& $TPRPython -B -m py_compile tpr_model.py tpr_inputs.py train_tpr.py analyze_tpr.py test_tpr_model.py
& $TPRPython -B -m pytest -q -p no:cacheprovider test_tpr_model.py
& $TPRPython -B -u train_tpr.py --phase preflight --seeds 123 124 125
& $TPRPython -B -u train_tpr.py --phase smoke --seeds 123 124 125 --prepare-missing
& $TPRPython -B -u train_tpr.py --phase formal --seeds 123 124 125
& $TPRPython -B -u train_tpr.py --phase analyze --seeds 123 124 125
```

The one-time Smoke flag authorizes the TPR confusion and SVD preparation. The graph uses all 10,847 clean Stage II TRAIN images, as specified by the task and protocol section 3. Section 5 item 8 still refers to the holdout. Positive edges use the mean of the two row-normalized confusion rates, ordered by descending score and then ascending class IDs. Scores select the pairs. The diversity penalty averages their squared cosines with equal weight.

Each seed initializes A with the existing orthogonal initializer and B with exact zeros. The fixed top-8 basis comes from one FP32 CPU SVD per seed. The ratio denominator and cosine normalization use epsilon 1e-12. Batch diagnostics contain all loss components and both factor gradient norms. Epoch CSVs report batch means for those scalars and sample-weighted Many top-1 agreement.

The final factor checkpoints, complete pairing traces, batch diagnostics and transcript stay local. Their paths and checksums are recorded in the artifact manifest.
