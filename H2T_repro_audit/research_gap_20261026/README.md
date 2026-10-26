# H2T → PI-H2T: evidence-first research gap study

## Why this study exists

The goal is **not** to prove original H2T is wrong or invent a better attention/ratio module. It is to identify one defensible limitation, check what the original authors and newer works already solved, and determine whether any existing local experiment motivates a properly controlled next intervention.

Start with the author claims in H2T (AAAI 2024) and PI-H2T (TPAMI 2026), not with a random collection of new tests.

## One priority reproduction anomaly to resolve

Local report: H2T/PI_H2T_PIF_CAUSAL_DIAGNOSTIC_REPORT.md

Under its local PIF Stage-I checkpoint and Stage-II seed 42, fixed epoch 10:
- No fusion, BN update (T2): Overall 45.69%.
- Local native PI H2TF, BN update (T3): Overall 34.82%; donor mass about 0.7697.
- Matched global donor-mass scale (T4): Overall 37.26%.

These results show **a local anomaly only**, not a failure of the published PI-H2T. The local implementation was independently reconstructed and may differ from the full paper protocol. Check math, direction of r / 1-r, classifier, PIF, initialization, checkpoint and data protocol before attributing cause. Prior PIF ratio B vs shuffled C did NOT show stable correspondence benefit, while the CE-only setting did. Do not hide negative results.

## Procedure

1. Verify local historical report anchors and file hashes.
2. Read the complete primary papers in PAPER_SOURCE_REGISTRY.json. Verify each proposed claim by exact section, equation, figure/table, PDF page number and paper version. Citation by abstract alone is not enough for formula/protocol claims.
3. Create a four-way crosswalk: (a) author claim, (b) code actually active, (c) controlled existing observation, (d) unresolved causal question. Make a separate row for every module or coupling of interest.
4. Distinguish author H2T, author PI-H2T, the repository's archived H2T upstream, and independently reconstructed PI-H2T candidate code.
5. Audit the local PIF+H2TF failure as a **paper-to-implementation parity question first**. Where parity is unknown, label BLOCKED, not confirmed flaw.
6. Identify one scientific question (at most), justify why authors/FeatRecon/SEL/DBG have not settled it, and define a minimal falsifiable paired intervention. Produce a preregistration, but do **not** launch GPU training in this iteration.
7. If paper-to-implementation parity is inadequate, the next recommendation must be a parity repair/reproduction audit, not a new feature fusion method.

## Run the local evidence verifier

From repository root:

```bash
python -m unittest discover -s H2T_repro_audit/research_gap_20261026 -p "test_*.py" -v
python H2T_repro_audit/research_gap_20261026/verify_evidence.py --output H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01
```

The utility uses Python's standard library only, does not load a checkpoint, does not import H2T training code and does not access the CIFAR official TEST. It fails on a missing/changed quoted source. It verifies source location and hash, not the correctness of a published scientific claim.

For the rest of the paper reading, parity check, deliverable names and scientific gate, open CODEX_H2T_PI_H2T_GAP_STUDY.txt.

**No GPU experiment until a mechanism gap is verified and its causal interpretation can be meaningfully separated from reproduction mismatches.**
