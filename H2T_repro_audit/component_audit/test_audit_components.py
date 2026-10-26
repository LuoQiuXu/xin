"""Unit tests for the read-only H2T component audit helper."""
import ast
import unittest

from audit_components import (
    ComponentVisitor,
    build_component_status,
    categories_for,
    parse_config_text,
)


class ConfigParserTests(unittest.TestCase):
    def test_nested_scalar_values_and_comments(self):
        parsed = parse_config_text(
            """
name: 'example #1'
mixup: True # active only in this stage
sampler:
  type: "weighted sampler"
  dual_sample:
    enable: True
    type: 'long-tailed'
H2T: 0.25
alpha: null
"""
        )
        self.assertEqual(parsed["name"], "example #1")
        self.assertIs(parsed["mixup"], True)
        self.assertEqual(parsed["sampler.type"], "weighted sampler")
        self.assertIs(parsed["sampler.dual_sample.enable"], True)
        self.assertEqual(parsed["sampler.dual_sample.type"], "long-tailed")
        self.assertAlmostEqual(parsed["H2T"], 0.25)
        self.assertIsNone(parsed["alpha"])


class StaticScannerTests(unittest.TestCase):
    def test_categories_cover_core_component_families(self):
        self.assertIn("feature_fusion", categories_for("H2T"))
        self.assertIn("normalization", categories_for("nn.BatchNorm2d"))
        self.assertIn("attention_candidate", categories_for("nn.MultiheadAttention"))
        self.assertIn("sampling", categories_for("ClassAwareSampler"))
        self.assertIn("classifier_head", categories_for("nn.Linear"))

    def test_ast_visitor_records_definition_call_and_no_grad(self):
        source = """
import torch
import torch.nn as nn

def train_step(x, y):
    norm = nn.BatchNorm2d(4)
    z = H2T(x, y, rho=0.25)
    with torch.no_grad():
        return z
"""
        visitor = ComponentVisitor(source, "fixture.py")
        visitor.visit(ast.parse(source))
        self.assertTrue(any(s["name"] == "train_step" for s in visitor.symbols))
        self.assertTrue(any(c["name"] == "nn.BatchNorm2d" for c in visitor.calls))
        self.assertTrue(any(c["name"] == "H2T" for c in visitor.calls))
        self.assertTrue(any(c["name"] == "torch.no_grad" for c in visitor.contexts))

    def test_status_uses_config_as_evidence_not_runtime_proof(self):
        symbols = [
            {"name": "H2T", "kind": "function", "path": "methods.py", "line": 10, "scope": "<module>"},
            {"name": "mixup_data", "kind": "function", "path": "methods.py", "line": 1, "scope": "<module>"},
        ]
        calls = [
            {"name": "H2T", "path": "train_stage2.py", "line": 20, "scope": "train"},
            {"name": "mixup_data", "path": "train_stage1.py", "line": 30, "scope": "train"},
        ]
        status = {x["name"]: x for x in build_component_status(
            symbols, calls,
            {"mixup": True},
            {"H2T": 0.25, "mixup": False},
        )}
        self.assertIn("enabled by selected Stage 2 config", status["H2T"]["status"])
        self.assertIn("Stage 1 path enabled", status["mixup_data"]["status"])
        self.assertIn("Stage 2 mixup is false", status["mixup_data"]["evidence_notes"][-1])


if __name__ == "__main__":
    unittest.main()
