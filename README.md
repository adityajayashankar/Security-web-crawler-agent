# Vulnerability GraphRAG Agent
Pipeline
Complete run order from scratch on Windows PowerShell. Covers data collection, KG build, vector indexing, and agent runtime.

## Architecture Overview

```
Data Sources (10+)                  Neo4j Knowledge Graph
  NVD / EPSS / CISA KEV               407k nodes / 6.9M relationships
  MITRE ATT&CK / CWE                  Vulnerability ↔ CWE ↔ OWASP ↔ Software
  GitHub Advisories                   CORRELATED_WITH / CO_OCCURS_WITH edges
  Blogs / Papers / Vendors  ──────►  StackProfile / CWECluster / NegativeRule
  Closed sources / ExploitDB                        │
              │                                     ▼
              ▼                       Qdrant Vector Index (bge-small-en-v1.5)
  master_vuln_context.jsonl            vuln_kg_evidence_v1 (planned 1M-2M chunks)
              │                                     │
              └──────────────► LangGraph Agent (main.py)
                                multi-hop hybrid GraphRAG + HITL policy
```

## Key Script Map

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Master orchestrator — collect → correlate → build → validate |
| `data/build_master_dataset.py` | Join all raw artifacts into one `master_vuln_context.jsonl` |
| `scripts/kg/load_kg_master.py` | Load Neo4j KG from master JSONL (preferred) |
| `scripts/kg/load_kg.py` | Legacy multi-file Neo4j loader |
| `scripts/maintenance/validate_kg.py` | Automated KG integrity + source-alignment checks |
| `scripts/maintenance/graphrag_embed_index_qdrant_cpu.py` | CPU-optimized streaming embed + upsert to Qdrant |
| `scripts/maintenance/graphrag_embed_local.py` | Phase 1 of two-phase ingest: embed to local JSONL cache |
| `scripts/maintenance/graphrag_upsert_cache.py` | Phase 2 of two-phase ingest: upsert cached vectors to Qdrant |
| `eval/run_graphrag_benchmark.py` | Held-out retrieval quality benchmark |
| `main.py` | Interactive agent CLI |

---

## Phase 0 — Environment Setup

### 0a) Open project and activate venv

```powershell
cd C:\Users\adity\dataset-deplai
.\.venv\Scripts\Activate.ps1
```

### 0b) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 0c) Configure `.env`

If `.env` does not exist:

```powershell
Copy-Item .env.example .env
```

Edit `.env` or set the following in every new PowerShell session:

```powershell
# ── Qdrant Cloud ──────────────────────────────────────────────────────────────
$env:QDRANT_URL="https://<your-cluster>.cloud.qdrant.io"
$env:QDRANT_API_KEY="<your_qdrant_api_key>"
$env:QDRANT_COLLECTION="vuln_kg_evidence_v1"
$env:QDRANT_TEXT_FIELD="text"

# ── Neo4j ─────────────────────────────────────────────────────────────────────
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="<your_neo4j_password>"

# ── LLM backends (Groq → OpenRouter → Ollama fallback chain) ──────────────────
$env:GROQ_API_KEY="<your_groq_api_key>"
# $env:OPENROUTER_API_KEY="<optional>"

# ── Optional source-collection keys ───────────────────────────────────────────
# $env:GITHUB_TOKEN="<github_token>"
# $env:TAVILY_API_KEY="<tavily_key>"          # blog crawler
# $env:REDDIT_CLIENT_ID="<reddit_id>"
# $env:REDDIT_CLIENT_SECRET="<reddit_secret>"
# $env:HACKERONE_USERNAME="<h1_user>"
# $env:HACKERONE_API_TOKEN="<h1_token>"
# $env:MSRC_API_KEY="<msrc_key>"
# $env:CISCO_CLIENT_ID="<cisco_id>"
# $env:CISCO_CLIENT_SECRET="<cisco_secret>"

# ── Keep vector retrieval OFF until ingest completes ──────────────────────────
$env:GRAPHRAG_USE_VECTOR="0"
$env:AGENT_GRAPHRAG_USE_VECTOR="0"
```

---

## Phase 1 — Data Collection & Graph Inputs

This phase crawls all sources, builds correlation/co-occurrence graphs, and produces graph input artifacts.

### 1a) Full pipeline (collect → correlate → build → validate)

```powershell
python run_pipeline.py
```

**Graph-relevant pipeline stages (in order):**

| # | Stage | Key output |
|---|-------|-----------|
| 1 | Crawl NVD | `data/raw_nvd.json` (328k CVEs) |
| 2 | Crawl EPSS | `data/raw_epss.json` |
| 3 | Crawl GitHub Advisories | `data/raw_github.json` |
| 4 | Crawl Blogs (agentic) | `data/raw_blogs.json` |
| 5 | Crawl ExploitDB | `data/raw_exploitdb.json` |
| 6 | Crawl CISA KEV | `data/raw_cisa_kev.json` |
| 7 | Crawl Research Papers | `data/raw_papers.json` |
| 8 | Crawl MITRE ATT&CK | `data/raw_mitre_attack.json` |
| 9 | Crawl Vendor Advisories | `data/raw_vendor_advisories.json` |
| 10 | Crawl Closed Sources | `data/raw_closed.json` |
| 11 | Build CVE correlations | `data/raw_correlations.json` |
| 12 | Collect CWE chains | `data/raw_cwe_chains.json` |
| 13 | Cluster KEV campaigns | `data/raw_kev_clusters.json` |
| 14 | Build co-occurrence v2 | `data/raw_cooccurrence_v2.json` |
| 15 | Build core vulnerability dataset | `data/vuln_dataset.jsonl` |
| 18 | Validate dataset | quality report to stdout |

**Useful flags:**

```powershell
# Skip all crawling (use existing raw_*.json files)
python run_pipeline.py --skip-crawl

# Skip crawl + correlation build (use existing raw_*.json + raw_correlations.json)
python run_pipeline.py --from-build

# Collect only (no build)
python run_pipeline.py --only collect

# Open sources only (no closed/semi-private APIs)
python run_pipeline.py --open-only

# Limit NVD total (faster dev runs)
python run_pipeline.py --nvd-total 50000

# Dry run (show what would run, no execution)
python run_pipeline.py --dry-run
```

### 1b) Validate dataset quality (recommended)

```powershell
python scripts/analysis/validate_dataset.py
```

Optional — fast heuristic mode without loading the tokenizer:

```powershell
python scripts/analysis/validate_dataset.py --no-tokenizer
```

Auto-fix bad examples (drop outputs <80 chars, dedup):

```powershell
python scripts/analysis/validate_dataset.py --fix
```

### 1c) Dataset statistics

```powershell
python scripts/analysis/analyze_dataset.py
```

Or run both together:

```powershell
python scripts/maintenance/run_both_scripts.py
```

---

## Phase 2 — Build Master Dataset

Joins all raw artifacts into a single enriched JSONL used by the KG loader and Qdrant indexer.
This is the canonical final dataset for graph runtime.

```powershell
python data/build_master_dataset.py
```

Output: `data/master_vuln_context.jsonl`

**Options:**

```powershell
# Disable enrichment from additional raw sources (only vuln_dataset + correlations + cooc)
python data/build_master_dataset.py --no-use-all-raw

# Disable negative_rules metadata row embedding
python data/build_master_dataset.py --no-embed-negative-rules

# Cap items per raw source (useful for memory-constrained environments)
python data/build_master_dataset.py --max-source-items 50000
```

---

## Phase 3 — Load Knowledge Graph (Neo4j)

**Preferred — single master file:**

```powershell
python scripts/kg/load_kg_master.py --master-file data/master_vuln_context.jsonl
```

**Legacy — multi-file loader:**

```powershell
python scripts/kg/load_kg.py
```

Expected graph after load:

| Metric | Value |
|--------|-------|
| Vulnerability nodes | ~326,969 |
| Total nodes | ~407,713 |
| CORRELATED_WITH edges | ~5.12M |
| CO_OCCURS_WITH edges | ~891k |
| Total relationships | ~6.9M |
| Canonical final dataset | `data/master_vuln_context.jsonl` |
| Planned vector chunks to store | `1,000,000 - 2,000,000` |

---

## Phase 4 — KG Validation & Benchmark Gate

Run this before promoting the vector index.

### 4a) KG integrity validation

```powershell
python scripts/maintenance/validate_kg.py `
  --strict `
  --sample-cves 50 `
  --seed 42 `
  --output-json eval/results/kg_validation.json `
  --output-md   eval/results/kg_validation.md
```

### 4b) Held-out retrieval benchmark

```powershell
python eval/run_graphrag_benchmark.py `
  --benchmark-file  eval/heldout_cve_benchmark.jsonl `
  --ground-truth    eval/ground_truth_benchmark.jsonl `
  --max-probes 120 --top-k 20 --max-hops 2 --strict `
  --output-json eval/results/graphrag_eval.json `
  --output-csv  eval/results/graphrag_eval.csv
```

Notes:
- `eval/heldout_cve_benchmark.jsonl` is auto-generated from raw artifacts if missing.
- `eval/ground_truth_benchmark.jsonl` is the curated analyst-labeled ground truth (already present).

Expected artifacts:

```
eval/results/kg_validation.json
eval/results/kg_validation.md
eval/heldout_cve_benchmark.jsonl   ← auto-generated
eval/results/graphrag_eval.json
eval/results/graphrag_eval.csv
```

---

## Phase 5 — Vector Indexing (Qdrant)

The embedding model is `BAAI/bge-small-en-v1.5` (384-dim, CPU-optimized).
Observed throughput: ~15.6 vectors/sec on CPU.
Planned storage target: `1,000,000 - 2,000,000` chunks (`~18 - 36` hours on CPU at current throughput).

### 5a) Verify Qdrant connectivity

```powershell
python -c "
from qdrant_client import QdrantClient; import os
c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'])
print(c.get_collections())
"
```

### 5b) Smoke test (2k vectors)

```powershell
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py `
  --max-vectors 2000 --batch-size 64 --qdrant-batch-size 256
```

### 5c) Planned ingest target (1M chunks, streaming)

```powershell
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py `
  --max-vectors 1000000 --batch-size 64 --qdrant-batch-size 256
```

Optional larger target (2M chunks):

```powershell
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py `
  --max-vectors 2000000 --batch-size 64 --qdrant-batch-size 256
```

### 5d) Resume if interrupted

```powershell
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py `
  --resume-from <done_count> --max-vectors <1000000_or_2000000> --batch-size 64 --qdrant-batch-size 256
```

### 5e) Alternative two-phase ingest (embed locally, then upsert)

Use this when embedding and uploading are on separate machines:

```powershell
# Phase 1 — embed to local cache
python scripts/maintenance/graphrag_embed_local.py `
  --out data/vector_cache/graphrag_vectors.jsonl.gz

# Phase 2 — upsert cache to Qdrant
python scripts/maintenance/graphrag_upsert_cache.py `
  --cache data/vector_cache/graphrag_vectors.jsonl.gz
```

### 5f) Enable vector retrieval after ingest completes

```powershell
$env:GRAPHRAG_USE_VECTOR="1"
$env:AGENT_GRAPHRAG_USE_VECTOR="1"
```

> **Troubleshooting:** If you see a dimension mismatch error, delete and recreate the Qdrant collection at 384 dims before re-running ingest.

---

## Phase 6 — Run the Agent

```powershell
python main.py
```

At the prompt, enter a CVE ID or free-text vulnerability question.

**Quick validation query:**

```
CVE-2021-28310
```

Expected: multi-step GraphRAG tool calls, `direct_evidence` list with ≥1 entry, a confidence score, and a structured final report.

**Agent tuning env vars:**

```powershell
$env:AGENT_GRAPHRAG_TOP_K="20"      # number of graph+vector candidates
$env:AGENT_GRAPHRAG_MAX_HOPS="2"    # graph traversal depth
$env:AGENT_GRAPHRAG_USE_VECTOR="1"  # enable vector retrieval (after ingest)
```

---

## Open Items

1. ExploitDB crawler currently returns 0 records (all known mirror URLs are failing). Contributions welcome.

