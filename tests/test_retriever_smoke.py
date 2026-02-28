import unittest

from pipeline.graphrag.retriever import retrieve_hybrid


class TestRetrieverSmoke(unittest.TestCase):
    def test_retrieve_hybrid_returns_contract_shape(self):
        payload = retrieve_hybrid("cooccurrence for CVE-2021-44228", top_k=5)
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
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()

