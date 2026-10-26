"""Tests for provenance guardrails; stdlib only, no model import."""
import tempfile
import unittest
from pathlib import Path
from verify_evidence import verify

def fixture(src="confirmed experimental statement"):
    temp=tempfile.TemporaryDirectory()
    root=Path(temp.name)
    (root/"sample.md").write_text(src,encoding="utf8")
    claim={"id":"E1","source":"sample.md","anchor":"experimental statement",
           "observation":"some claim","boundary":"not independently causal","type":"experiment"}
    return temp,root,{"schema_version":1,"claims":[claim]}

class EvidenceTests(unittest.TestCase):
    def test_exact_anchor(self):
        tmp,root,doc=fixture()
        with tmp:
            rows=verify(root,doc)
            self.assertEqual(rows[0]["line"],1)
            self.assertEqual(rows[0]["status"],"LOCAL_TEXT_MATCH_ONLY")
            self.assertEqual(len(rows[0]["sha256"]),64)

    def test_mismatch_stops(self):
        tmp,root,doc=fixture("different source")
        with tmp,self.assertRaises(AssertionError): verify(root,doc)

    def test_escaping_repo_root_stops(self):
        tmp,root,doc=fixture()
        with tmp,self.assertRaises(ValueError):
            doc["claims"][0]["source"]="../../other.md"
            verify(root,doc)

    def test_duplicate_ids_stop(self):
        tmp,root,doc=fixture()
        with tmp,self.assertRaises(ValueError):
            doc["claims"].append(doc["claims"][0].copy())
            verify(root,doc)

    def test_empty_ledger_stops(self):
        tmp,root,doc=fixture()
        with tmp,self.assertRaises(ValueError):
            doc["claims"]=[]
            verify(root,doc)

if __name__=="__main__":unittest.main()
