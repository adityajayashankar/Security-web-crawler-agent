import unittest

from pipeline.graphrag.retriever import _compute_second_hop_from_seed_map


class TestMultiHopPropagation(unittest.TestCase):
    def test_second_order_candidates_are_generated_with_paths(self):
        hop1_rows = [
            {"cve_id": "CVE-2024-0002", "score": 0.9},
            {"cve_id": "CVE-2024-0003", "score": 0.8},
        ]
        seed_neighbors = {
            "CVE-2024-0002": [
                {"cve_id": "CVE-2024-0100", "score": 0.7, "rel_type": "CO_OCCURS_WITH", "signals": ["s1"], "reasons": []},
                {"cve_id": "CVE-2024-0101", "score": 0.5, "rel_type": "CORRELATED_WITH", "signals": ["s2"], "reasons": []},
            ],
            "CVE-2024-0003": [
                {"cve_id": "CVE-2024-0100", "score": 0.6, "rel_type": "CORRELATED_WITH", "signals": ["s3"], "reasons": []},
                {"cve_id": "CVE-2024-0102", "score": 0.9, "rel_type": "CO_OCCURS_WITH", "signals": ["s4"], "reasons": []},
            ],
        }

        rows = _compute_second_hop_from_seed_map(
            origin_marker="CVE-2024-0001",
            hop1_rows=hop1_rows,
            seed_neighbors_map=seed_neighbors,
            top_k=5,
        )
        ids = {r["cve_id"] for r in rows}
        self.assertIn("CVE-2024-0100", ids)
        self.assertIn("CVE-2024-0102", ids)

        any_path = next(r for r in rows if r["cve_id"] == "CVE-2024-0100")
        self.assertEqual(any_path["hop"], 2)
        self.assertTrue(any_path["path"][0] == "CVE-2024-0001")
        self.assertTrue(any_path["path"][2] == "CVE-2024-0100")

    def test_origin_and_first_hop_are_excluded_from_second_hop(self):
        hop1_rows = [
            {"cve_id": "CVE-2024-0002", "score": 0.9},
        ]
        seed_neighbors = {
            "CVE-2024-0002": [
                {"cve_id": "CVE-2024-0001", "score": 0.8, "rel_type": "CO_OCCURS_WITH", "signals": [], "reasons": []},  # origin
                {"cve_id": "CVE-2024-0002", "score": 0.7, "rel_type": "CO_OCCURS_WITH", "signals": [], "reasons": []},  # seed itself
                {"cve_id": "CVE-2024-0200", "score": 0.6, "rel_type": "CORRELATED_WITH", "signals": [], "reasons": []},
            ],
        }
        rows = _compute_second_hop_from_seed_map(
            origin_marker="CVE-2024-0001",
            hop1_rows=hop1_rows,
            seed_neighbors_map=seed_neighbors,
            top_k=5,
        )
        ids = {r["cve_id"] for r in rows}
        self.assertNotIn("CVE-2024-0001", ids)
        self.assertNotIn("CVE-2024-0002", ids)
        self.assertIn("CVE-2024-0200", ids)


if __name__ == "__main__":
    unittest.main()

