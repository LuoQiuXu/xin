# Low-VS Factorial Pilot

This directory contains the pre-registered protocol, policy primitives, CPU unit tests, and Codex execution instructions for a small 2×2 mechanism pilot.

## Scientific question

Does frequency-stratified, sample-conditioned head-donor affinity help tail receivers, and does protecting the least favorable channel transfers inside the original Low-VS Bottom48 pool help independently or jointly?

## Contents

- PROTOCOL.md: fixed hypotheses, arms, data/training protocol, diagnostics and stop rules.
- policy.py: pure donor-stratification, TRAIN-only prototype-affinity, exact linear margin-contribution, and protected-mask functions.
- test_policy.py: CPU tests for the policy primitives.
- CODEX_LOWVS_FACTORIAL.txt: full local integration and run instructions.

The runner must be integrated against the user's existing sealed local helpers because the full Clean Stage-I data/cache directory is intentionally not committed to GitHub. Codex must inspect the real online_masks trace and reuse the original helper implementations rather than guessing or rewriting them. The runner must be created only in this directory and all outputs must remain in results/.

## Run after the runner has been implemented and reviewed

From repository root, using the existing experiment Python environment:

    python H2T_repro_audit/lowvs_factorial/run_factorial.py unit
    python H2T_repro_audit/lowvs_factorial/run_factorial.py preflight
    python H2T_repro_audit/lowvs_factorial/run_factorial.py prepare

Then execute the eight fixed training commands and the report command in CODEX_LOWVS_FACTORIAL.txt, sequentially. Do not use official CIFAR-100 TEST.
