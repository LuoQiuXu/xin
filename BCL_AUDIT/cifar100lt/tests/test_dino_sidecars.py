import csv
import hashlib
import json
import sys

import numpy as np
import pytest

from BCL_AUDIT.scripts import analyze_paired_bcl as analysis
from BCL_AUDIT.scripts import audit_cifar100lt_topology as topology


STATIC_PAIRS = "head_tail_pair_topology.csv"
DINO_PAIRS = "head_tail_pair_topology_dino.csv"
DINO_SUMMARY = "data_audit_dino_summary.json"
DINO_FILES = (
    DINO_PAIRS,
    DINO_SUMMARY,
    "DINO_CLASS_MEAN_FEATURES.npy",
    "DINO_CLASS_SIMILARITY.npy",
    "dino_metadata.json",
)


def _write_pairs(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
    }


@pytest.fixture
def pairs():
    return [
        {
            "head_fine_id": 6,
            "head_fine_name": "head",
            "tail_fine_id": tail,
            "tail_fine_name": f"tail {tail}",
            "head_coarse_id": 0,
            "tail_coarse_id": coarse,
            "same_superclass": coarse == 0,
            "audit_note": f"original field {tail}",
        }
        for tail, coarse in ((1, 0), (2, 1), (3, 1))
    ]


@pytest.fixture
def static_audit(tmp_path, monkeypatch, pairs):
    out_dir = tmp_path / "audit"
    out_dir.mkdir()
    indices = np.array([1, 2, 3, 6], dtype=np.int64)
    counts = np.bincount(indices, minlength=7)
    summary = {
        "dataset": "CIFAR-100",
        "source_split": "original balanced CIFAR-100 training split only",
        "test_split_loaded": False,
        "split_seed": 0,
        "imbalance_factor": 0.01,
        "class_counts": counts.tolist(),
        "long_tail_train_size": indices.size,
        "num_many_classes": 1,
        "num_medium_classes": 0,
        "num_few_classes": 3,
        "total_many_few_pairs": len(pairs),
        "within_superclass_many_few_pairs": 1,
        "cross_superclass_many_few_pairs": 2,
        "dino_similarity_computed": False,
    }
    (out_dir / "class_counts.csv").write_text("fine_id,count\n6,1\n", encoding="utf-8")
    (out_dir / "superclass_counts.csv").write_text("coarse_id,count\n0,2\n", encoding="utf-8")
    (out_dir / "data_audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    np.save(out_dir / "cifar100lt_if100_train_indices_seed0.npy", indices)
    _write_pairs(out_dir / STATIC_PAIRS, pairs)

    images = np.zeros((7, 2, 2, 3), dtype=np.uint8)
    labels = np.arange(7)
    monkeypatch.setattr(
        topology, "_load_original_train", lambda _: (images, labels, None, None, None, None)
    )
    monkeypatch.setattr(topology, "_make_long_tail_indices", lambda *_: (counts, indices))
    monkeypatch.setattr(
        sys, "argv",
        [
            "audit_cifar100lt_topology.py", "--data-root", str(tmp_path),
            "--output-dir", str(out_dir), "--compute-dino", "--reuse-static-audit",
        ],
    )
    return out_dir, summary


@pytest.mark.parametrize("existing_dino_columns", [True, False])
def test_dino_sidecars_preserve_static_audit(static_audit, monkeypatch, pairs, existing_dino_columns):
    out_dir, static_summary = static_audit
    if existing_dino_columns:
        _write_pairs(
            out_dir / STATIC_PAIRS,
            [dict(row, dino_cosine_similarity="", dino_topk_union="") for row in pairs],
        )
    before = _hashes(out_dir)
    assert len(before) == 5

    class_means = np.ones((7, 2), dtype=np.float32)
    similarity = np.eye(7, dtype=np.float32)
    similarity[6, 1:4] = similarity[1:4, 6] = [0.375, 0.125, -0.625]
    metadata = {
        "model": "dinov2_vits14", "elapsed_seconds": 0,
        "peak_cuda_memory_bytes": 0, "topk": 5,
    }
    calls = []

    def fake_dino(**options):
        calls.append(options["topk"])
        return class_means, similarity, {(1, 6), (2, 6)}, metadata

    monkeypatch.setattr(topology, "_compute_dino_similarity", fake_dino)
    topology.main()

    after = _hashes(out_dir)
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == set(DINO_FILES)
    assert analysis.read_csv(out_dir / DINO_PAIRS) == [
        dict(
            {name: str(value) for name, value in row.items()},
            dino_cosine_similarity=cosine, dino_topk_union=near,
        )
        for row, cosine, near in zip(
            pairs, ("0.37500000", "0.12500000", "-0.62500000"), ("True", "True", "False"),
        )
    ]
    summary = json.loads((out_dir / DINO_SUMMARY).read_text(encoding="utf-8"))
    assert summary == {
        **static_summary,
        "dino_enrichment_completed": True,
        "dino_topk_union_many_few_pairs": 2,
        "dino_topk_union_within_superclass_many_few_pairs": 1,
        "dino_topk_union_cross_superclass_many_few_pairs": 1,
    }
    np.testing.assert_array_equal(np.load(out_dir / "DINO_CLASS_MEAN_FEATURES.npy"), class_means)
    np.testing.assert_array_equal(np.load(out_dir / "DINO_CLASS_SIMILARITY.npy"), similarity)
    assert json.loads((out_dir / "dino_metadata.json").read_text(encoding="utf-8")) == metadata

    topology.main()
    assert calls == [5]
    assert _hashes(out_dir) == after


@pytest.mark.parametrize("existing_output", DINO_FILES)
def test_partial_dino_outputs_stop_before_loading_data(static_audit, monkeypatch, existing_output):
    out_dir, _ = static_audit
    (out_dir / existing_output).write_bytes(b"previous partial output")
    before = _hashes(out_dir)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Partial outputs must stop the run before loading data or DINOv2")

    monkeypatch.setattr(topology, "_load_original_train", unexpected_load)
    monkeypatch.setattr(topology, "_compute_dino_similarity", unexpected_load)
    with pytest.raises(FileExistsError, match="DINOv2 outputs are partial or inconsistent"):
        topology.main()
    assert _hashes(out_dir) == before


def _summarize_pairs(rows):
    labels = np.array([1, 2, 3, 6])
    bcl_predictions = np.array([6, 2, 3, 1])
    logits = np.zeros((4, 7))
    return analysis.summarize_did(
        analysis.compute_pair_effects(labels, labels, bcl_predictions, logits, logits, logits, logits, rows)
    )


def test_analyzer_prefers_matching_dino_sidecar(tmp_path, pairs):
    _write_pairs(tmp_path / STATIC_PAIRS, pairs)
    dino_rows = [
        dict(row, dino_topk_union=near)
        for row, near in zip(reversed(pairs), (False, "", True))
    ]
    _write_pairs(tmp_path / DINO_PAIRS, dino_rows)
    selected = analysis.load_pair_topology(tmp_path)
    assert selected == analysis.read_csv(tmp_path / DINO_PAIRS)

    summary = _summarize_pairs(selected)
    assert [row for row in summary if row["relation"] == "same_superclass"] == _summarize_pairs(
        analysis.read_csv(tmp_path / STATIC_PAIRS)
    )
    dino_summary = [row for row in summary if row["relation"] == "dino_topk_union"]
    assert {row["slice"] for row in dino_summary} == {"DINO_near", "DINO_not_near"}
    assert all(row["num_pairs"] == 1 for row in dino_summary)


def test_analyzer_falls_back_without_fabricating_dino_relations(tmp_path, pairs):
    _write_pairs(tmp_path / STATIC_PAIRS, pairs)
    selected = analysis.load_pair_topology(tmp_path)
    assert selected == analysis.read_csv(tmp_path / STATIC_PAIRS)
    assert {row["relation"] for row in _summarize_pairs(selected)} == {"same_superclass"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tail_fine_id": 4}, "pair identities differ"),
        ({"tail_coarse_id": 4}, "superclass assignments differ"),
        ({"same_superclass": False}, "superclass assignments differ"),
    ],
)
def test_analyzer_rejects_inconsistent_dino_pairs(tmp_path, pairs, change, message):
    _write_pairs(tmp_path / STATIC_PAIRS, pairs)
    _write_pairs(tmp_path / DINO_PAIRS, [dict(pairs[0], **change), *pairs[1:]])
    with pytest.raises(RuntimeError, match=message):
        analysis.load_pair_topology(tmp_path)


def test_analyzer_rejects_incomplete_dino_pair_table(tmp_path, pairs):
    _write_pairs(tmp_path / STATIC_PAIRS, pairs)
    _write_pairs(tmp_path / DINO_PAIRS, pairs[:-1])
    with pytest.raises(RuntimeError, match="row count differs"):
        analysis.load_pair_topology(tmp_path)
