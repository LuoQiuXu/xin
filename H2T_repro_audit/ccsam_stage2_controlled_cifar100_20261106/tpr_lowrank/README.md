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
- Codex is instructed to add the isolated training runner and analysis code under this directory. Do not modify sealed C0-C4 sources or outputs.

## Local execution

From the repository root, read the Codex task file and follow its strict order:
1. verify branch/worktree and all input hashes;
2. run CPU unit tests and static checks;
3. run preflight;
4. run real-data Smoke for all three seeds;
5. only if every gate passes, run the three-seed 30-epoch formal experiment;
6. analyze fixed epoch 30.

The exact CLI commands are to be finalized by Codex after inspecting the existing C4 runner and local environment; do not guess checkpoint paths or run arguments.

Large checkpoints and batch traces should follow the repository's existing local-only artifact policy. Record SHA256 hashes and include the artifact manifest in the final report.
