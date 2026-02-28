import json
import unittest

from pipeline.graphrag.schema import (
    ConfidenceSummary,
    GraphEntity,
    GraphRAGAgentResponse,
    HITLDecision,
)
from pipeline.hitl import evaluate_hitl_policy
from pipeline.tools import _normalize_likelihood, tool_likely_on_system


class TestGraphRAGContract(unittest.TestCase):
    def test_likelihood_normalization_bounds(self):
        self.assertEqual(_normalize_likelihood(-1.0), 0.0)
        self.assertEqual(_normalize_likelihood(0.3), 0.3)
        self.assertLessEqual(_normalize_likelihood(7.0), 1.0)
        self.assertGreater(_normalize_likelihood(7.0), 0.0)

    def test_invalid_cve_rejected(self):
        payload = json.loads(tool_likely_on_system("NOT-A-CVE"))
        self.assertIn("error", payload)
        self.assertEqual(payload["direct_count"], 0)
        self.assertEqual(payload["inferred_count"], 0)

    def test_hitl_trigger_on_low_confidence(self):
        payload = {
            "entity": {"type": "cve", "id": "CVE-2024-0001"},
            "direct_evidence": [],
            "inferred_candidates": [{"cve_id": "CVE-2024-0002", "likelihood": 0.2}],
            "citations": [],
            "confidence_summary": {"overall": 0.2},
        }
        decision = evaluate_hitl_policy(payload)
        self.assertTrue(decision["required"])
        self.assertGreaterEqual(len(decision["reasons"]), 1)

    def test_schema_contract_required_fields(self):
        contract = GraphRAGAgentResponse(
            status="ok",
            query="cooccurrence for CVE-2021-44228",
            entity=GraphEntity(type="cve", id="CVE-2021-44228"),
            confidence_summary=ConfidenceSummary(overall=0.77, rationale="test"),
            hitl=HITLDecision(required=False, reasons=[]),
        )
        as_dict = contract.model_dump() if hasattr(contract, "model_dump") else contract.dict()
        for key in [
            "status",
            "query",
            "entity",
            "direct_evidence",
            "inferred_candidates",
            "citations",
            "confidence_summary",
            "hitl",
            "recommended_actions",
        ]:
            self.assertIn(key, as_dict)


if __name__ == "__main__":
    unittest.main()

