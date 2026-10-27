# PI-H2T author-source provenance gate

**Status: BLOCKED_FOR_CLAIM_OF_EXACT_AUTHOR_REPRODUCTION**.

This audit validates quoted local files only. It does not verify PDF page
numbers, current GitHub branch/tag availability or authors' unpublished code.
Even if all anchors match, no GPU training is authorized by this report.

## Upstream inventory (preliminary, recheck live)

- Author-referenced repository: Keke921/H2T
- Checked commit: e0c849235d9655fca31cd9fa8a5f38f92a77c7e8
- Observed branches: main
- Observed tree entries: 58
- README identifies the AAAI 2024 H2T sample code, not a complete
  PI-H2T implementation. This is NOT proof of global non-availability.

## Line-anchored claims

| Field | Present local evidence | Status | PI author code? |
|---|---|---|---|
| PIF_FORM | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:27 | PAPER_FORM_LOCAL_CANDIDATE | NOT_VERIFIED |
| PIF_INIT | H2T_repro_audit/pi_h2t_strict_20261027/PARITY_LOCK.md:20 | UNVERIFIED_AUTHOR_PI | NOT_VERIFIED |
| FUSION_FORM | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:35 | PAPER_FORM_LOCAL_CANDIDATE | NOT_VERIFIED |
| W_REFERENCE | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:37 | UNVERIFIED_AUTHOR_PI | NOT_VERIFIED |
| R_GRADIENT | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:38 | UNVERIFIED_AUTHOR_PI | NOT_VERIFIED |
| BN_STAGE2 | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:45 | UNVERIFIED_AUTHOR_PI | NOT_VERIFIED |
| MODEL_ARCH | H2T_repro_audit/pi_h2t_strict_20261027/PARITY_LOCK.md:28 | INHERITED_ORIGINAL_H2T_NOT_PI_VERIFIED | NOT_VERIFIED |
| STAGE1_OPT | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:42 | INHERITED_ORIGINAL_H2T_NOT_PI_VERIFIED | NOT_VERIFIED |
| STAGE2_OPT | H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md:43 | UNVERIFIED_AUTHOR_PI | NOT_VERIFIED |
| CKPT_SELECTION | H2T_repro_audit/pi_h2t_strict_20261027/PARITY_LOCK.md:35 | LOCAL_PROTOCOL_VARIANT | NOT_VERIFIED |
| AUTHOR_CODE | H2T_official_clean/H2T-main/README.md:15 | NOT_LOCATED_IN_CHECKED_AUTHOR_REPO | NOT_LOCATED |

## Scientific gate

The source README and old local reconstruction cannot establish which
PIF initialization, Eq8 W-reference, detach rule, BN-running-buffer mode
or checkpoint selection the PI-H2T authors actually used.

Codex MUST independently read the full PI-H2T journal/arXiv paper,
supplement, Code Availability statement, and the actual author GitHub
refs and files. Record commit-qualified code paths or PDF page numbers.
Only paper/author evidence may resolve an UNKNOWN field; NEVER choose
frozen/current W or another parameter because its TEST result is larger.

Outcomes: AUTHOR_PI_CODE_VERIFIED; PAPER_RECONSTRUCTION_WITH_EXPLICIT_GAPS;
or BLOCKED_NEEDS_AUTHOR_CONFIRMATION. No training in this study.
