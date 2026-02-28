# DeplAI Project Status Report (2026-02-27)

## 1) Executive Summary

The core pipeline is operational end-to-end:

- Data artifacts are present and large enough for production-scale reasoning.
- Neo4j KG is loaded with millions of edges and is being queried by the agent.
- LangGraph now does call `graphrag_query` first for CVE correlation/co-occurrence questions.

Primary compromises still affecting quality/performance:

1. Vector index mismatch and incompleteness:
   - Remote Qdrant collection is reachable but has **0 points**.
   - Local Qdrant has **20,038 points** only (partial index), while latest full chunk build was **6,252,241** chunks.
2. Full embedding/upsert runtime is currently too high for full refresh cadence.
3. LLM backend reliability is degraded by model rate limits and decommissioned model IDs.
4. Some source and training-layer imbalances remain (not blockers, but quality risks).

---

## 2) End-to-End Metrics (Beginning to End)

### A. Raw Data + Build Outputs (`data/`)

Source: `scripts/maintenance/_report_stats.py` (run on 2026-02-27)

| Artifact | Records | Size |
|---|---:|---:|
| `raw_nvd.json` | 328,000 | 283.5 MB |
| `raw_blogs.json` | 228 | 2.6 MB |
| `raw_correlations.json` | 328,000 | 830.4 MB |
| `raw_cooccurrence_v2.json` | 891,953 | 184.4 MB |
| `raw_cooccurrence.json` | 891,953 | 184.4 MB |
| `raw_cisa_kev.json` | 1,529 | 1.8 MB |
| `raw_epss.json` | 305,266 | 8.6 MB |
| `raw_exploitdb.json` | 0 | 0.0 MB |
| `raw_github.json` | 3,000 | 10.0 MB |
| `raw_mitre_attack.json` | 5 (top-level keys) | 0.0 MB |
| `raw_papers.json` | 1,450 | 2.5 MB |
| `raw_vendor_advisories.json` | 851 | 0.8 MB |
| `raw_closed.json` | 1,550 | 1.2 MB |
| `raw_cwe_chains.json` | 4 (top-level keys) | 5.1 MB |
| `raw_kev_clusters.json` | 160,594 | 41.2 MB |
| `training_pairs.jsonl` | 2,636,528 | 2,067.3 MB |
| `vuln_dataset.jsonl` | 325,941 | 1,278.7 MB |
| **TOTAL** |  | **4,902.4 MB** |

### B. Dataset Quality Coverage (`vuln_dataset.jsonl`, full-count checks)

- Total rows: **325,941**
- With CVSS: **306,051** (**93.90%**)
- With EPSS: **304,212** (**93.33%**)
- With CWE: **253,984** (**77.92%**)
- With affected software: **283,266** (**86.91%**)

### C. Training Pair Distribution

- Total training pairs: **2,636,528**
- Correlation + co-occurrence layers: **1,885,890 (71.5%)**
- Layer counts:
  - `vulnerability_cooccurrence`: 1,283,158 (48.7%)
  - `vulnerability_correlation`: 602,732 (22.9%)
  - `vulnerability_intelligence`: 306,137 (11.6%)
  - `audit_evidence`: 306,051 (11.6%)
  - `risk_scoring`: 131,449 (5.0%)
  - `pentesting_intelligence`: 4,813 (0.2%)
  - `remediation_learning`: 1,993 (0.1%)
  - `execution_context`: 195 (0.0%)

Thin-layer checkpoint in pipeline:
- `execution_context` threshold is 200; current count is **195** (below threshold).

### D. Correlation/Co-occurrence Structure Stats

`raw_correlations.json`:
- Rows: **328,000**
- Rows with related vulnerabilities: **302,393**
- Total related links emitted: **5,807,760**
- Avg related links per row: **17.707**
- Median / p95 / max related links per row: **20 / 20 / 20**

`raw_cooccurrence_v2.json`:
- Total pairs: **891,953**
- Negative inference rules: **63**
- Stack profiles: **22,192**
- By source:
  - `product_cooccurrence`: 574,457 (**64.4%**)
  - `high_conf_same_stack`: 133,910 (**15.01%**)
  - `vendor_kev`: 85,493 (**9.58%**)
  - `temporal_kev`: 54,749 (**6.14%**)
  - `cwe_can_precede`: 23,255 (**2.61%**)
  - `ransomware_kev`: 19,494 (**2.19%**)
  - `attack_chain`: 458
  - `conditional_same_stack`: 118
  - `remediation_tie`: 19

### E. Neo4j KG Load Metrics (live query)

Current graph (`bolt://127.0.0.1:7687`):

Nodes:
- `Vulnerability`: 326,969
- `Software`: 79,991
- `CWE`: 737
- `OWASPCategory`: 10
- `CWECluster`: 6
- **Total nodes**: **407,713**

Relationships:
- `CORRELATED_WITH`: 5,120,356
- `CO_OCCURS_WITH`: 891,953
- `AFFECTS_SOFTWARE`: 499,086
- `HAS_CWE`: 254,048
- `MAPS_TO_OWASP`: 162,704
- `CONTAINS_CWE`: 39
- **Total relationships**: **6,928,186**

### F. GraphRAG Indexing Metrics

Latest observed full chunk build log:
- `Chunk build complete: raw=6,252,241 deduped=6,252,241`

Qdrant status:
- Remote collection `vuln_kg_evidence_v1`: **0 points**
- Local collection `data/qdrant/vuln_kg_evidence_v1`: **20,038 points**
  - This is **0.32%** of the last full chunk build volume.

### G. Agent Runtime Metrics (live run probe)

Command run: `python main.py` with query `CVE-2021-28310`

Observed behavior:
- Step 1 forced tool call: `graphrag_query(...)` (top_k=20, max_hops=2, use_vector=false)
- Final report contained:
  - `direct_evidence`: **20**
  - `inferred_candidates`: **0**
  - Evidence breakdown:
    - `CO_OCCURS_WITH`: **2**
    - `CORRELATED_WITH`: **18**
  - `confidence_summary.overall`: **0.652**

This confirms KG traversal is active in the agent path.

---

## 3) Where the Workflow Is Compromised

## C1. Vector retrieval path is effectively disabled in practice

What is happening:
- Agent/retriever path is currently graph-first and usually graph-only (`use_vector=false` defaults).
- Even if vector is enabled, configured remote Qdrant has 0 points.

Impact:
- No semantic/vector evidence contribution.
- Hybrid retrieval confidence and diversity are underutilized.

Root cause:
- Indexing went into local Qdrant earlier, while runtime points to remote Qdrant URL.
- Full remote backfill was not completed.

## C2. Embedding backlog is too large for one-shot indexing

What is happening:
- Full evidence set is ~6.25M chunks.
- Prior observed run pace indicates multi-day full upsert if done monolithically.

Impact:
- Freshness lag between KG/data updates and vector index availability.
- Operational friction (index jobs seem “stuck” to users).

Root cause:
- Correlation source is very dense (5.8M related links alone).
- Single-pass full indexing without source-sliced checkpoints.

## C3. LLM backend reliability is noisy

What is happening:
- Rate limits and retries on Groq/OpenRouter.
- Decommissioned model IDs still present in fallback list (`mixtral-8x7b-32768`, `gemma2-9b-it`).

Impact:
- Intermittent slowdowns.
- Occasional fallback pathways and unstable synthesis quality.

## C4. Data/source quality imbalance

What is happening:
- `raw_exploitdb.json` currently has 0 entries.
- Some sources are relatively small (e.g., blogs 228).
- `execution_context` training layer is below threshold.

Impact:
- Less diverse evidence for exploit/procedure style recommendations.
- Potentially weaker context-aware tooling output.

## C5. Secrets hygiene risk observed during debugging

What is happening:
- Credentials were exposed in terminal during troubleshooting earlier.

Impact:
- Security risk if keys were copied/logged elsewhere.

---

## 4) Fix Plan (Upcoming Days)

## Day 0 (today): Stabilize runtime path

1. Pick one Qdrant target and stick to it:
   - Option A: remote cloud (recommended for scale)
   - Option B: local only (for dev)
2. Set env consistently (`QDRANT_URL`, `QDRANT_COLLECTION`, `GRAPHRAG_USE_VECTOR`).
3. Remove decommissioned models from `pipeline/model_loader.py`.
4. Rotate exposed API keys and DB credentials.

Targets by end of Day 0:
- `main.py` query returns graph evidence consistently without `needs_human_review`.
- No decommissioned-model errors in logs.

## Day 1: Fast usable vector baseline (not full backfill yet)

1. Index in slices (not monolith):
   - dataset slice
   - co-occurrence slice
   - top-ranked correlations slice
2. Use caps per run:
   - `GRAPHRAG_MAX_TOTAL_CHUNKS` set to a staged value (e.g., 200k then 500k).
3. Turn on vector retrieval after first successful slice (`AGENT_GRAPHRAG_USE_VECTOR=1`).

Targets by end of Day 1:
- Remote Qdrant points >= **500k**
- `graphrag_query` rationale shows non-zero vector candidates.

## Day 2: Scale up backfill safely

1. Continue chunk slices until >= 2M points.
2. Prioritize high-signal relations first (top score correlations + KEV-driven co-occurrence).
3. Validate latency and memory during indexing.

Targets by end of Day 2:
- Remote Qdrant points >= **2M**
- Query latency acceptable for CLI usage.

## Day 3+: Fullness + quality hardening

1. Complete backfill toward full chunk coverage if needed for your use case.
2. Improve sparse layers:
   - bring `execution_context` above threshold.
3. Restore/repair weak sources (ExploitDB pipeline).
4. Add regression checks:
   - sample CVE suite with expected co-occurrence/correlation evidence.

Targets:
- `execution_context >= 200`
- Stable evidence breakdown on benchmark CVEs.

---

## 5) Success Criteria (Definition of “Working Properly”)

The agent is considered fully healthy when:

1. For CVE query, first tool call is `graphrag_query`.
2. Final report includes non-empty `direct_evidence` with both:
   - `CORRELATED_WITH` and
   - `CO_OCCURS_WITH` (when available in graph).
3. Vector contribution is non-zero on production Qdrant.
4. Model fallback path has no decommissioned model errors.
5. Data build + KG load + retrieval checks pass in one routine run.

---

## 6) Verification Commands

Neo4j counts:

```cypher
MATCH (n:Vulnerability) RETURN count(n);
MATCH ()-[r:CORRELATED_WITH]->() RETURN count(r);
MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r);
```

Agent sanity probe:

```powershell
$env:NEO4J_PASSWORD="***"
python main.py
# query: CVE-2021-28310
```

Qdrant status (Python):

```python
from qdrant_client import QdrantClient
c = QdrantClient(url="https://<your-qdrant>")
print(c.get_collection("vuln_kg_evidence_v1"))
```

