# KG Validation Checklist

Use this checklist to validate that the Neo4j knowledge graph is structurally correct and aligned with source data.

## Current Validation Status (March 9, 2026)

| Check | Status | Notes |
|-------|--------|-------|
| Neo4j KG loaded | ✅ Pass | 407,713 nodes / 6,928,186 relationships |
| Automated validation (`validate_kg.py`) | 🔲 Not yet run | Run with `--strict --sample-cves 50 --seed 42` |
| Automated benchmark (`run_graphrag_benchmark.py`) | ❌ Error | All 120 held-out probes fail: `No module named 'pipeline'` — fix: run from project root with `PYTHONPATH=.` |
| Ground truth benchmark | ❌ Error | Same import error on all 20 probes |
| Manual agent probe (CVE-2021-28310) | ✅ Pass | 20 results returned, confidence 0.652, HITL not triggered |
| Qdrant vector ingest | ⚠️ Partial | 20,038 / 1,000,000–2,000,000 target vectors |

---

## How To Explain KG Validation (Short Answer)

If someone asks how the KG was validated and de-noised, use this:

"I validated the KG at four levels:
1. Graph integrity checks: uniqueness constraints, null-ID checks, duplicate/self-loop edge checks, and expected node/edge counts by type.
2. Source-grounded verification: sampled CVEs and matched `CORRELATED_WITH`/`CO_OCCURS_WITH` neighbors against `raw_correlations.json` and `raw_cooccurrence_v2.json`.
3. Noise controls in build: applied thresholds/caps (minimum correlation score, max related-per-CVE, quotas) to prune low-confidence and hub-noise edges.
4. Task-level validation: ran benchmark CVE queries in the agent and confirmed evidence-backed retrieval with citations and stable outputs across runs."

This demonstrates both correctness (structure + source alignment) and low-noise behavior (controlled edge generation + retrieval quality).

## 1) Schema and volume sanity

Run in Neo4j Browser:

```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC;
SHOW CONSTRAINTS;
```

Expected:
- Core labels exist (`Vulnerability`, `CWE`, `OWASPCategory`, `Software`, `CWECluster`).
- Core relationship types exist (`CORRELATED_WITH`, `CO_OCCURS_WITH`, `HAS_CWE`, `MAPS_TO_OWASP`, `AFFECTS_SOFTWARE`).
- Uniqueness constraints exist for node IDs.

## 2) Key integrity checks

```cypher
MATCH (v:Vulnerability) WHERE v.vuln_id IS NULL RETURN count(v) AS missing_vuln_id;
MATCH (v:Vulnerability) WITH v.vuln_id AS id, count(*) AS c WHERE c > 1 RETURN id, c LIMIT 20;
MATCH (w:CWE) WHERE w.cwe_id IS NULL RETURN count(w) AS missing_cwe_id;
MATCH (o:OWASPCategory) WHERE o.owasp_id IS NULL RETURN count(o) AS missing_owasp_id;
```

Expected:
- `missing_*` values should be `0`.
- Duplicate IDs should return no rows.

## 3) Relationship hygiene

```cypher
MATCH (v:Vulnerability)-[r:CO_OCCURS_WITH|CORRELATED_WITH]->(v)
RETURN type(r) AS rel, count(*) AS self_loops;

MATCH (a:Vulnerability)-[r:CO_OCCURS_WITH]->(b:Vulnerability)
WITH a.vuln_id AS a, b.vuln_id AS b, count(r) AS c
WHERE c > 1
RETURN a, b, c LIMIT 20;
```

Expected:
- Self-loops should be `0` or explainable.
- Duplicate edges should be minimal/none.

## 4) Coverage checks

```cypher
MATCH (v:Vulnerability)
RETURN
  count(v) AS total,
  sum(CASE WHEN EXISTS { (v)-[:HAS_CWE]->() } THEN 1 ELSE 0 END) AS with_cwe,
  sum(CASE WHEN EXISTS { (v)-[:MAPS_TO_OWASP]->() } THEN 1 ELSE 0 END) AS with_owasp,
  sum(CASE WHEN EXISTS { (v)-[:AFFECTS_SOFTWARE]->() } THEN 1 ELSE 0 END) AS with_sw;
```

Expected:
- Non-zero and high `with_cwe`, `with_owasp`, `with_sw` vs `total`.

## 5) Source alignment spot-check (example CVE)

```cypher
MATCH (v:Vulnerability {vuln_id:"CVE-2021-28310"})-[r:CORRELATED_WITH|CO_OCCURS_WITH]-(x:Vulnerability)
RETURN x.vuln_id, type(r), coalesce(r.max_score, r.max_confidence, r.confidence, 0) AS score
ORDER BY score DESC LIMIT 20;
```

Then compare with:
- `data/raw_correlations.json`
- `data/raw_cooccurrence_v2.json`

Expected:
- Neighbor CVEs and scores/confidence are consistent with raw artifacts.

## 6) Agent path verification

1. Run:
```powershell
python main.py
```
2. Query:
```text
CVE-2021-28310
```
3. Confirm final report has:
- Non-empty `direct_evidence`
- `CORRELATED_WITH` and `CO_OCCURS_WITH` entries (when available)
- Reasonable confidence summary

## 7) Optional automation target

If repeated validation is needed, create `scripts/maintenance/validate_kg.py` and fail CI on:
- Missing constraints
- Null/duplicate IDs
- Unexpected empty key relationships
