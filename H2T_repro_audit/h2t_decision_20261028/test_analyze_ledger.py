import tempfile,unittest
from pathlib import Path
from analyze_ledger import check,summarize
def fixture(d):
    root=Path(d);(root/"old.md").write_text("existing",encoding="utf-8")
    return root,{"schema_version":1,"no_gpu":True,"records":[{"id":"R","branch":"b","path":"old.md","claim":"c","status":"NO_GO","evidence":"e","confound":"n","next":"stop"}]}
class Tests(unittest.TestCase):
    def test_pass(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);self.assertEqual(len(check(r,j)),1)
            self.assertIn("No new GPU trial",summarize(check(r,j)))
    def test_missing(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);j["records"][0]["path"]="none"
            with self.assertRaises(FileNotFoundError):check(r,j)
    def test_escape(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);j["records"][0]["path"]="../out"
            with self.assertRaises(ValueError):check(r,j)
    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);j["records"][0]["status"]="SOTA"
            with self.assertRaises(ValueError):check(r,j)
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);j["records"].append(j["records"][0].copy())
            with self.assertRaises(ValueError):check(r,j)
    def test_train_disallowed(self):
        with tempfile.TemporaryDirectory() as d:
            r,j=fixture(d);j["no_gpu"]=False
            with self.assertRaises(ValueError):check(r,j)
if __name__=="__main__":unittest.main()
