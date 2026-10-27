"""Offline tests: fail closed on missing/misquoted or overclaimed evidence."""
import tempfile
import unittest
from pathlib import Path
from run_source_gate import audit, render_report


def fixture():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    (root / "original.md").write_text(
        "This repo contains the sample code for our AAAI 2024: H2T\n"
        "Ratio reference classifier is uncertain\n", encoding="utf-8"
    )
    field = {
        "id": "W_REFERENCE",
        "claim": "Meaning of W",
        "status": "UNVERIFIED_AUTHOR_PI",
        "paper_pointer": "PI-H2T Eq8",
        "path": "original.md",
        "anchor": "Ratio reference classifier",
        "author_pi_code": "NOT_VERIFIED",
        "question": "Frozen or live?",
    }
    doc = {"schema_version": 1,
           "upstream": {"repo": "Keke921/H2T",
                        "branch_names_observed": ["main"],
                        "complete_tree_observed": True,
                        "checked_head": "snapshot",
                        "tree_size_observed": 58},
           "fields": [field]}
    return temp, root, doc


class EvidenceGateTests(unittest.TestCase):
    def test_line_anchor_and_digest(self):
        tmp, root, doc = fixture()
        with tmp:
            rows = audit(root, doc)
            self.assertEqual(rows[0]["line"], 2)
            self.assertEqual(len(rows[0]["sha256"]), 64)
            self.assertEqual(rows[0]["verification_scope"], "LOCAL_TEXT_ANCHOR_ONLY")

    def test_report_refuses_exact_parity(self):
        tmp, root, doc = fixture()
        with tmp:
            report = render_report(doc, audit(root, doc))
            self.assertIn("BLOCKED_FOR_CLAIM_OF_EXACT_AUTHOR_REPRODUCTION", report)
            self.assertIn("No training", report)

    def test_absent_anchor_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(AssertionError):
            doc["fields"][0]["anchor"] = "invented quote"
            audit(root, doc)

    def test_traversal_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(ValueError):
            doc["fields"][0]["path"] = "../../protected.md"
            audit(root, doc)

    def test_missing_path_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(FileNotFoundError):
            doc["fields"][0]["path"] = "missing.md"
            audit(root, doc)

    def test_duplicate_claim_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(ValueError):
            doc["fields"].append(doc["fields"][0].copy())
            audit(root, doc)

    def test_false_verified_author_pi_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(ValueError):
            doc["fields"][0]["author_pi_code"] = "VERIFIED"
            audit(root, doc)

    def test_unrecognized_status_fails(self):
        tmp, root, doc = fixture()
        with tmp, self.assertRaises(ValueError):
            doc["fields"][0]["status"] = "OFFICIAL_MATCH"
            audit(root, doc)

    def test_empty_and_incomplete_inventory_fail(self):
        tmp, root, doc = fixture()
        with tmp:
            doc["fields"] = []
            with self.assertRaises(ValueError):
                audit(root, doc)
            doc["fields"] = [fixture()[2]["fields"][0]]
            doc["upstream"]["complete_tree_observed"] = False
            with self.assertRaises(ValueError):
                audit(root, doc)


if __name__ == "__main__":
    unittest.main()
