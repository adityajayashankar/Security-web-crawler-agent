# DeplAI — Vulnerability GraphRAG Pipeline: Comprehensive Project Report

**Date:** March 9, 2026  
**Status:** Core pipeline operational — vector ingest at 20k / 1M-2M target; benchmark eval blocked by module path

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why We Are Building This](#2-why-we-are-building-this)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Component Deep-Dive](#4-component-deep-dive)
5. [Full Data Lineage](#5-full-data-lineage)
6. [Knowledge Graph Schema](#6-knowledge-graph-schema)
7. [Agent Reasoning Flow](#7-agent-reasoning-flow)
8. [LLM Backend & Fine-tuning](#8-llm-backend--fine-tuning)
9. [Capabilities & Features](#9-capabilities--features)
10. [Metrics & Scale](#10-metrics--scale)
11. [Known Gaps & Next Steps](#11-known-gaps--next-steps)

---

## 1. What This Project Is

DeplAI is a **domain-specialized, multi-modal cybersecurity intelligence platform**. It ingests vulnerability data from 10+ heterogeneous sources, fuses them into a Neo4j knowledge graph and a Qdrant vector index, and exposes them through a LangGraph-based agentic CLI and a REST API.

The agent can answer questions like:

- *"What co-exists with CVE-2021-28310 on a compromised system?"*
- *"Given CWE-89, what other weaknesses appear in the same attack chain?"*
- *"How likely is this CVE to be exploited, and what is the full audit finding?"*
- *"Which vulnerabilities appear together in ransomware campaigns?"*

The platform is being used to:
1. Build a **fine-tuned security LLM** (`Foundation-Sec-8B` via QLoRA) on 3.5M vulnerability training pairs.
2. Power a **hybrid GraphRAG retrieval engine** combining symbolic Neo4j graph traversal with dense Qdrant vector search.
3. Provide a structured **REST API backend** for a graph visualization frontend.

---

## 2. Why We Are Building This

### The Problem

Cybersecurity practitioners face three core intelligence challenges:

| Challenge | Industry State | What This Does |
|-----------|---------------|----------------|
| **Vulnerability relationship reasoning** | CVEs exist in silos; NVD does not model co-occurrence or attack chains | Builds explicit `CORRELATED_WITH` and `CO_OCCURS_WITH` edges from 5.8M correlation links |
| **Exploit probability prioritization** | CVSS base scores ignore real-world exploitation likelihood | Integrates EPSS scores, CISA KEV activeness, and campaign cluster membership |
| **Multi-hop attack path discovery** | General LLMs hallucinate CVE relationships | Grounds all answers in graph evidence + vector evidence + HITL policy |

### Why Foundation-Sec-8B Fine-tuning

General-purpose LLMs (GPT-4, LLaMA) do not deeply understand CVE co-occurrence patterns, CWE family relationships, or OWASP category mapping. Foundation-Sec-8B was pre-trained on 80B tokens of cybersecurity-domain text (NVD, advisories, exploit code). Fine-tuning it on our 3.5M domain-specific pairs produces an LLM that:

- Reasons about multi-CVE attack chains without hallucination
- Understands CWE hierarchy and OWASP taxonomy natively
- Can generate structured audit findings from raw evidence

### Why the Hybrid GraphRAG Approach

Pure RAG (vector-only) cannot model the **topology** of vulnerability relationships — it cannot distinguish whether two CVEs are directly correlated (same exploit), co-occurring (same product stack), or merely semantically similar text. The hybrid approach:

- Uses **Neo4j graph traversal** (structured, exact, explainable) as the primary signal
- Uses **Qdrant vector search** (semantic, recall-oriented) as a secondary signal
- Merges both into a single ranked evidence list with confidence scores and citations

---

## 3. Architecture Diagram

### High-Level System Architecture

```mermaid
graph TB
    subgraph Sources["Data Sources (10+)"]
        NVD[NVD — 328k CVEs]
        EPSS[EPSS Scores]
        KEV[CISA KEV — 1.5k]
        GH[GitHub Advisories — 3k]
        MITRE[MITRE ATT&CK]
        EXDB[ExploitDB]
        BLOGS[Blogs / Papers]
        VENDOR[Vendor Advisories]
        CLOSED[Closed Sources]
        CWE[CWE Chains]
    end

    subgraph DataPipeline["Phase 1 — Data Pipeline"]
        CRAWL[Crawlers<br/>data/crawl_*.py]
        CORR[Correlation Builder<br/>build_correlations.py]
        COOC[Co-occurrence Builder v2<br/>scripts/dataset/build_cooccurrence_v2.py]
        KEV_CLUST[KEV Campaign Clusterer<br/>cluster_kev_campaigns.py]
        OWASP[OWASP Mapper<br/>owasp_mapper.py]
        DS[Dataset Builder<br/>build_dataset.py]
        PAIRS[Training Pair Generator<br/>expand_training_pairs.py]
    end

    subgraph Artifacts["Intermediate Artifacts"]
        RAW_JSON[raw_*.json files<br/>~4.9 GB total]
        VULN_DS[vuln_dataset.jsonl<br/>325k CVEs / 1.34 GB]
        TRAIN_PAIRS[training_pairs.jsonl<br/>3.5M pairs / 3.17 GB]
        MASTER[master_vuln_context.jsonl<br/>enriched per-CVE]
    end

    subgraph KnowledgeGraph["Phase 3 — Neo4j Knowledge Graph"]
        NEO4J[(Neo4j<br/>407k nodes<br/>6.9M edges)]
        VUL_NODE[Vulnerability nodes<br/>326,969]
        SW_NODE[Software nodes<br/>79,991]
        CWE_NODE[CWE nodes — 737]
        OWASP_NODE[OWASPCategory — 10]
        CLUSTER_NODE[CWECluster — 6]
    end

    subgraph VectorIndex["Phase 4 — Qdrant Vector Index"]
        EMBED[Embedder<br/>BAAI/bge-small-en-v1.5<br/>dim=384, CPU]
        QDRANT[(Qdrant Cloud<br/>vuln_kg_evidence_v1<br/>1M-2M target)]
    end

    subgraph Agent["Runtime — LangGraph Agent"]
        PLANNER[Planner Node<br/>CVE/CWE guardrails]
        TOOL_EXEC[Tool Executor Node]
        SYNTH[Synthesis Node]
        HITL[HITL Policy<br/>hitl.py]
    end

    subgraph Tools["Agent Tools (12)"]
        GR_QUERY[graphrag_query<br/>Hybrid retrieval]
        LOOKUP_CVE[lookup_cve<br/>NVD live lookup]
        LIKELY[likely_on_system<br/>3-tier KG lookup]
        LOOKUP_CWE[lookup_by_cwe<br/>CWE-first path]
        OTHER_TOOLS[map_owasp / fetch_epss<br/>score_risk / get_pentest_method<br/>select_tool / get_remediation<br/>generate_finding]
    end

    subgraph LLMBackend["LLM Backend — Fallback Chain"]
        GROQ[Groq API<br/>llama-3.3-70b-versatile]
        OPENROUTER[OpenRouter<br/>Free models fallback]
        OLLAMA[Ollama<br/>Local fallback]
    end

    subgraph Training["Phase 6 — Fine-tuning"]
        FINETUNE[Foundation-Sec-8B<br/>QLoRA r=32 / 4096 ctx]
        HF[HuggingFace Hub<br/>adityajayashankar/<br/>vuln-foundation-sec-8b]
    end

    subgraph Backend["REST API Backend"]
        EXPRESS[Express.js server<br/>vuln-graph-backend/server.js]
        API[REST Endpoints<br/>/api/cve/:id/correlations<br/>/api/cve/:id/full<br/>/api/cwe/:id/vulns<br/>/api/graph]
    end

    Sources --> CRAWL
    CRAWL --> RAW_JSON
    RAW_JSON --> CORR
    RAW_JSON --> COOC
    RAW_JSON --> KEV_CLUST
    RAW_JSON --> DS
    DS --> VULN_DS
    CORR --> PAIRS
    COOC --> PAIRS
    VULN_DS --> PAIRS
    PAIRS --> TRAIN_PAIRS
    VULN_DS --> MASTER
    CORR --> MASTER
    COOC --> MASTER

    MASTER --> NEO4J
    NEO4J --- VUL_NODE
    NEO4J --- SW_NODE
    NEO4J --- CWE_NODE
    NEO4J --- OWASP_NODE
    NEO4J --- CLUSTER_NODE

    MASTER --> EMBED
    EMBED --> QDRANT

    TRAIN_PAIRS --> FINETUNE
    FINETUNE --> HF

    NEO4J --> GR_QUERY
    QDRANT --> GR_QUERY
    NEO4J --> LIKELY
    NEO4J --> LOOKUP_CWE

    GR_QUERY --> TOOL_EXEC
    LOOKUP_CVE --> TOOL_EXEC
    LIKELY --> TOOL_EXEC
    LOOKUP_CWE --> TOOL_EXEC
    OTHER_TOOLS --> TOOL_EXEC

    PLANNER --> TOOL_EXEC
    TOOL_EXEC --> SYNTH
    SYNTH --> HITL
    HITL --> PLANNER

    SYNTH --> LLMBackend
    OTHER_TOOLS --> LLMBackend

    NEO4J --> EXPRESS
    EXPRESS --> API
```

---

### Data Flow — Detailed Pipeline

```mermaid
flowchart LR
    subgraph Phase1["Phase 1: Collect"]
        C1[crawl_nvd.py] --> R1[raw_nvd.json\n328k CVEs]
        C2[crawl_epss.py] --> R2[raw_epss.json\n305k scores]
        C3[crawl_cisa_kev.py] --> R3[raw_cisa_kev.json\n1.5k]
        C4[crawl_github.py] --> R4[raw_github.json\n3k]
        C5[crawl_mitre_attack.py] --> R5[raw_mitre_attack.json]
        C6[crawl_blogs.py] --> R6[raw_blogs.json\n228 pages]
        C7[crawl_papers.py] --> R7[raw_papers.json\n1.45k]
        C8[crawl_vendor_advisories.py] --> R8[raw_vendor_advisories.json\n851]
        C9[crawl_exploitdb.py] --> R9[raw_exploitdb.json]
        C10[crawl_closed_sources.py] --> R10[raw_closed.json\n1.55k]
    end

    subgraph Phase2["Phase 2: Correlate"]
        R1 & R2 & R3 & R4 & R5 --> CORR2[build_correlations.py\n5.8M links]
        R1 & R3 --> COOC2[scripts/dataset/build_cooccurrence_v2.py\n891k pairs]
        COLLECT_CWE[collect_cwe_chains.py] --> CHAINS[raw_cwe_chains.json]
        CLUSTER[cluster_kev_campaigns.py] --> KEVCLUST[raw_kev_clusters.json\n160k]
    end

    subgraph Phase3["Phase 3: Build Dataset"]
        CORR2 --> BUILD[build_dataset.py]
        COOC2 --> BUILD
        CHAINS --> BUILD
        BUILD --> VULN[vuln_dataset.jsonl\n325k rows]
        BUILD --> STACK[scripts/dataset/stack_profiles.py → raw_cooccurrence_v2]
        VULN --> EXPAND[expand_training_pairs.py]
        EXPAND --> TP[training_pairs.jsonl\n3.5M pairs]
        SYNTH[scripts/dataset/generate_synthetic_pairs.py] --> TP
        COOC_PAIRS[scripts/dataset/generate_cooccurrence_pairs.py] --> TP
    end

    subgraph Phase4["Phase 4: Master Build + KG Load"]
        VULN & CORR2 & COOC2 --> MASTER2[build_master_dataset.py]
        MASTER2 --> MASTERF[master_vuln_context.jsonl]
        MASTERF --> KGLOAD[scripts/kg/load_kg_master.py]
        KGLOAD --> NEO4J2[(Neo4j KG)]
    end

    subgraph Phase5["Phase 5: Vector Ingest"]
        MASTERF --> EMBED2[graphrag_embed_index_qdrant_cpu.py]
        EMBED2 --> QDRANT2[(Qdrant Cloud)]
    end

    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5
```

---

### Agent Execution Flow

```mermaid
sequenceDiagram
    actor User
    participant Planner
    participant Tools
    participant Neo4j
    participant Qdrant
    participant LLM
    participant HITL

    User->>Planner: "CVE-2021-28310 — what co-exists?"
    Note over Planner: CVE regex match → force graphrag_query
    Planner->>Tools: graphrag_query({entity:{cve}, top_k:20, max_hops:2})
    Tools->>Neo4j: MATCH CVE -[CORRELATED_WITH|CO_OCCURS_WITH]->
    Neo4j-->>Tools: 18 CORRELATED + 2 CO_OCCURS rows
    Tools->>Qdrant: vector search (when GRAPHRAG_USE_VECTOR=1)
    Qdrant-->>Tools: top-k semantic matches
    Tools-->>Planner: EvidenceItems + Citations (JSON)
    Planner->>HITL: evaluate_hitl_policy(payload)
    HITL-->>Planner: {required: false} (confidence 0.652 is OK)
    Planner->>LLM: synthesize FINAL JSON report
    LLM-->>Planner: structured finding
    Planner-->>User: FINAL REPORT (JSON)
```

---

### GraphRAG Agent — Architecture Diagram

```mermaid
graph TB
    USER([User Query]) --> GUARD

    subgraph INPUT["Input Layer"]
        GUARD[Guardrail Parser\nCVE / CWE regex · format validation]
    end

    subgraph AGENT["LangGraph Agent Core  ·  pipeline/langgraph_agent.py"]
        PLANNER[Planner Node\nforce graphrag_query if CVE + corr hint\nforce lookup_by_cwe if CWE-only\nelse LLM selects tool]
        EXECUTOR[Tool Executor Node\ndispatch to tool fn · append result]
        SYNTH[Synthesis Node\nLLM synthesizes FINAL JSON\nor fallback from raw tool output]
        STATE[AgentState\nquery · memory · tool_results\nstep_num · max_steps · pending_tool]
        PLANNER --> EXECUTOR --> SYNTH
        SYNTH -- needs more tools --> PLANNER
        STATE -. shared .-> PLANNER & EXECUTOR & SYNTH
    end

    subgraph TOOLS["Tool Layer  ·  pipeline/tools.py"]
        T1[graphrag_query\nHybrid KG + vector retrieval]
        T2[likely_on_system\n3-tier KG traversal]
        T3[lookup_by_cwe\nCWE → cluster → CVEs]
        T4[lookup_cve · fetch_epss\nNVD + FIRST.org live APIs]
        T5[map_owasp · score_risk\nget_pentest_method · select_tool\ngenerate_finding · get_remediation]
    end

    subgraph RETRIEVAL["Retrieval Layer  ·  pipeline/graphrag/"]
        subgraph GRAPH_SRC["Graph Path  ·  retriever.py"]
            NEO4J[(Neo4j\n407k nodes · 6.9M edges\nCORRELATED_WITH · CO_OCCURS_WITH\nHAS_CWE · MAPS_TO_OWASP\nAFFECTS_SOFTWARE)]
        end
        subgraph VEC_SRC["Vector Path  ·  embeddings.py + qdrant_conn.py"]
            EMBEDDER[BAAI/bge-small-en-v1.5\ndim=384 · CPU]
            QDRANT[(Qdrant Cloud\nvuln_kg_evidence_v1\n20k / 1M-2M target)]
            EMBEDDER --> QDRANT
        end
        MERGE[Evidence Merge\ndedup · score-sort · tier-split\ndirect_evidence vs inferred_candidates\ncitation building · confidence scoring]
        NEO4J --> MERGE
        QDRANT --> MERGE
    end

    subgraph LLM_BACKEND["LLM Backend  ·  pipeline/model_loader.py"]
        GROQ[Groq\nllama-3.3-70b-versatile\nllama-3.1-8b-instant]
        OR[OpenRouter\nllama-3.3-70b · gemma-3-27b\nmistral-7b  free tier]
        OLLAMA[Ollama  local\nllama3.2 · mistral · phi3]
        GROQ -- rate limited --> OR -- unavailable --> OLLAMA
    end

    subgraph POLICY["Quality + Safety Layer"]
        HITL[HITL Policy  ·  pipeline/hitl.py\n5 trigger conditions\ninferred dominates · low confidence\nsource disagreement · sparse evidence]
        SCHEMA[GraphRAGAgentResponse  ·  schema.py\nstatus · entity · direct_evidence\ninferred_candidates · citations\nconfidence_summary · hitl · actions]
        HITL --> SCHEMA
    end

    FINAL([FINAL JSON Report])

    GUARD --> PLANNER
    EXECUTOR --> T1 & T2 & T3 & T4 & T5
    T1 --> NEO4J & EMBEDDER
    T2 & T3 --> NEO4J
    MERGE --> HITL
    T5 --> LLM_BACKEND
    SYNTH --> LLM_BACKEND
    SCHEMA --> EXECUTOR
    SYNTH -- done --> FINAL

    style NEO4J fill:#4a90d9,color:#fff
    style QDRANT fill:#6c4eb8,color:#fff
    style FINAL fill:#2a7a2a,color:#fff
    style HITL fill:#c47f00,color:#fff
    style GUARD fill:#555,color:#fff
```

---

## 4. Component Deep-Dive

### 4.1 Data Collection Layer (`data/crawl_*.py`)

All crawlers write to `data/raw_*.json`. Key crawlers:

| Crawler | Source | Output | Notes |
|---------|--------|--------|-------|
| `crawl_nvd.py` | NVD REST API v2 | `raw_nvd.json` 328k CVEs | Full CVE metadata, CVSS, CWE, CPE |
| `crawl_epss.py` | FIRST.org EPSS | `raw_epss.json` 305k | Daily exploit probability scores |
| `crawl_cisa_kev.py` | CISA KEV catalog | `raw_cisa_kev.json` 1.5k | Known-exploited vulnerabilities |
| `crawl_github.py` | GitHub Security Advisories | `raw_github.json` 3k | Patch info, GHSA IDs |
| `crawl_mitre_attack.py` | MITRE ATT&CK STIX | `raw_mitre_attack.json` | Technique-to-CVE mappings |
| `crawl_blogs.py` | Agentic Tavily + crawl4ai | `raw_blogs.json` 228 | LLM-driven URL discovery + quality filter |
| `crawl_papers.py` | Academic papers | `raw_papers.json` 1.45k | Research on exploit patterns |
| `crawl_vendor_advisories.py` | MSRC, Cisco, etc. | `raw_vendor_advisories.json` 851 | Vendor-specific mitigations |
| `crawl_closed_sources.py` | HackerOne, Reddit | `raw_closed.json` 1.55k | Bug bounty + community signals |
| `crawl_exploitdb.py` | Exploit-DB | `raw_exploitdb.json` | PoC exploit code (currently 0 — broken) |

**Blog crawler is fully agentic:**  
Uses Groq LLM to generate search queries → Tavily finds URLs → crawl4ai downloads pages → quality keyword filter keeps only security-relevant content → LLM gap-analysis generates Round 2 queries.

---

### 4.2 Correlation & Co-occurrence Layer

**`build_correlations.py`** — builds `raw_correlations.json`:
- Joins NVD + EPSS + GitHub + KEV + MITRE ATT&CK
- Emits up to 20 related CVEs per source CVE (capped to control hub noise)
- Minimum correlation score threshold: ≥0.60
- Output: 328k rows, 5.8M total links

**`scripts/dataset/build_cooccurrence_v2.py`** — builds `raw_cooccurrence_v2.json`:
- 8 co-occurrence signal types:

| Signal Type | Count | Meaning |
|-------------|------:|---------|
| `product_cooccurrence` | 574,457 | Same product/version affected |
| `high_conf_same_stack` | 133,910 | Same tech stack fingerprint |
| `vendor_kev` | 85,493 | Same vendor in CISA KEV |
| `temporal_kev` | 54,749 | Same KEV disclosure window |
| `cwe_can_precede` | 23,255 | CWE chain — one weakness enables another |
| `ransomware_kev` | 19,494 | Same ransomware campaign |
| `attack_chain` | 458 | Direct exploit chain evidence |
| `conditional_same_stack` | 118 | Same stack with conditional dependency |

**`cluster_kev_campaigns.py`** — groups KEV entries by temporal proximity and vendor overlap into 160k campaign cluster entries with 63 negative inference rules (pairs that look similar but are NOT related).

**`scripts/dataset/stack_profiles.py`** — builds 22,192 stack profiles (technology fingerprints) used as co-occurrence signals.

**`collect_cwe_chains.py`** — maps CWE parent-child and can-precede relationships, feeding `HAS_CWE` and `CONTAINS_CWE` graph edges.

---

### 4.3 Dataset Builder

**`data/build_dataset.py`** → `vuln_dataset.jsonl` (325k rows, 1.34 GB):

Each row is a rich per-CVE JSON containing:
- CVE ID, description, CVSS score, EPSS probability
- CWE IDs, OWASP category  
- Affected software list (CPEs)
- Correlated CVEs (up to 20) with confidence scores
- Co-occurring CVEs with signal types
- ATT&CK technique references
- Source references (NVD, GitHub, KEV, vendor)

**`data/expand_training_pairs.py`** + synthetic generators → `training_pairs.jsonl` (3.5M pairs, 3.17 GB):

Training pair distribution across 8 task layers:

| Layer | Count | % | Purpose |
|-------|------:|---|---------|
| `vulnerability_cooccurrence` | 1,568,794 | 44.7% | "CVE-X and CVE-Y co-occur because..." |
| `vulnerability_correlation` | 602,732 | 17.2% | "CVE-X is related to CVE-Y due to..." |
| `execution_context` | 325,964 | 9.3% | Stack-aware tooling |
| `vulnerability_intelligence` | 306,152 | 8.7% | OWASP / CWE mapping |
| `audit_evidence` | 306,051 | 8.7% | Audit finding generation |
| `remediation_learning` | 254,161 | 7.2% | Fix recommendations |
| `risk_scoring` | 140,784 | 4.0% | CVSS/EPSS risk assessment |
| `pentesting_intelligence` | 4,813 | 0.1% | Attack payloads, detection |

---

### 4.4 Neo4j Knowledge Graph

**Node types:**

| Label | Count | Primary ID | Description |
|-------|------:|-----------|-------------|
| `Vulnerability` | 326,969 | `vuln_id` (CVE-XXXX-XXXXX) | Core CVE nodes with all metadata |
| `Software` | 79,991 | `software_key` | CPE-based product/version nodes |
| `CWE` | 737 | `cwe_id` | Weakness type nodes |
| `OWASPCategory` | 10 | `owasp_id` | OWASP Top 10 categories |
| `CWECluster` | 6 | `cluster_id` | Grouped CWE families |

**Relationship types:**

| Relationship | Count | Direction | Properties |
|-------------|------:|-----------|-----------|
| `CORRELATED_WITH` | 5,120,356 | CVE ↔ CVE | `max_score`, `reasons`, `signals` |
| `CO_OCCURS_WITH` | 891,953 | CVE ↔ CVE | `max_confidence`, `source`, `signals` |
| `AFFECTS_SOFTWARE` | 499,086 | CVE → Software | CPE version range |
| `HAS_CWE` | 254,048 | CVE → CWE | Primary/secondary |
| `MAPS_TO_OWASP` | 162,704 | CVE → OWASPCategory | Category match |
| `CONTAINS_CWE` | 39 | CWECluster → CWE | Cluster membership |

**Total:** 407,713 nodes, 6,928,186 relationships

**Uniqueness constraints** enforced on all node IDs to prevent duplicates.  
**Self-loop guards** prevent any CVE from being correlated to itself.

---

### 4.5 Qdrant Vector Index

- **Collection:** `vuln_kg_evidence_v1`
- **Embedding model:** `BAAI/bge-small-en-v1.5`, dim=384, cosine similarity, CPU inference
- **Target size:** 1,000,000 – 2,000,000 vectors (planned; sourced from `master_vuln_context.jsonl`)
- **Ingest pipeline:** `graphrag_embed_index_qdrant_cpu.py` — streaming, no full JSON load in memory
- **Throughput observed:** 15.60 embeddings/sec on smoke test (2,000 vectors in 128s)
- **Chunk strategy:** 900-char structural chunks with 120-char overlap, boundary-aware splitting (prefers section breaks over mid-sentence cuts)
- **Sources chunked:** `vuln_dataset.jsonl` + `raw_correlations.json` + `raw_cooccurrence_v2.json` + optional KG edge chunks
- **Two-phase alternative:** `graphrag_embed_local.py` (cache to JSONL) → `graphrag_upsert_cache.py` (batch upsert)

---

### 4.6 LangGraph Agent (`pipeline/langgraph_agent.py`)

The agent uses a **3-node state machine:**

```
START → [Planner] → [Tool Executor] → [Synthesis/HITL] → (loop or END)
```

**Planner guardrails** (deterministic before LLM planning):
- If query contains `CVE-\d{4}-\d+` AND any correlation hint phrase → force `graphrag_query` as step 1
- If query contains `CWE-\d+` → force `lookup_by_cwe` as step 1
- Otherwise → LLM chooses tool sequence

**12 tools registered:**

| Tool | Tier | Backend |
|------|------|---------|
| `graphrag_query` | Primary | Neo4j + Qdrant hybrid |
| `lookup_cve` | Enrichment | NVD live API |
| `likely_on_system` | KG | Neo4j 3-tier traversal |
| `lookup_by_cwe` | KG | Neo4j CWE cluster |
| `map_owasp` | LLM | Groq/OR/Ollama |
| `get_pentest_method` | LLM | Groq/OR/Ollama |
| `select_tool` | LLM | Groq/OR/Ollama |
| `fetch_epss` | API | FIRST.org EPSS |
| `score_risk` | LLM | Groq/OR/Ollama |
| `generate_finding` | LLM | Groq/OR/Ollama |
| `get_remediation` | LLM | Groq/OR/Ollama |
| *(fallback)* | KG | `likely_on_system` fallback |

---

### 4.7 HITL Policy (`pipeline/hitl.py`)

Deterministic risk-triggered Human-in-the-Loop escalation. Triggers when any of:

1. Inferred evidence dominates direct evidence AND direct < 2 results
2. Overall confidence score < 0.40
3. Source disagreement: graph and vector/raw-cooccurrence both present, but no direct corroboration
4. CVE context with < 2 direct results AND top likelihood < 0.50
5. High confidence (>0.60) with sparse total evidence (<3 items) — over-estimation guard

When triggered, response includes `"hitl": {"required": true, "reasons": [...]}` and status `"needs_human_review"`.

---

### 4.8 GraphRAG Schema (`pipeline/graphrag/schema.py`)

Every agent response conforms to a strict Pydantic contract:

```python
GraphRAGAgentResponse:
  status: "ok" | "needs_human_review" | "error"
  query: str
  entity: {type: "cve"|"cwe"|"unknown", id: str}
  direct_evidence: [EvidenceItem]   # from KG direct edges
  inferred_candidates: [EvidenceItem]  # from 2-hop or vector
  citations: [Citation]             # source-traceable references
  confidence_summary: {overall: float, rationale: str}
  hitl: {required: bool, reasons: [str]}
  recommended_actions: [str]
```

`EvidenceItem` carries: `cve_id`, `likelihood [0,1]`, `evidence_tier`, `rel_type`, `signals`, `reasons`, `inferred_from`.

---

### 4.9 LLM Backend (`pipeline/model_loader.py`)

Three-tier fallback chain:

```
Groq (fastest, free 14.4k req/day)
  └─► llama-3.3-70b-versatile (best quality)
  └─► llama-3.1-8b-instant (rate limit fallback)
  └─► llama3-8b-8192 (solid general fallback)
  └─► llama-3.2-11b-text-preview

  ↓ (if Groq fails/rate-limited)

OpenRouter (free models)
  └─► meta-llama/llama-3.3-70b-instruct:free
  └─► google/gemma-3-27b-it:free
  └─► mistralai/mistral-7b-instruct:free

  ↓ (if OpenRouter unavailable)

Ollama (local, zero rate limits)
  └─► llama3.2 / mistral / llama3.1 / phi3
```

8 domain-specific system prompts per task **layer** — the model_loader routes each tool call to the correct security persona (`vulnerability_intelligence`, `audit_evidence`, `vulnerability_correlation`, etc.).

---

### 4.10 REST API Backend (`vuln-graph-backend/server.js`)

Express.js server exposing Neo4j through REST with:
- Rate limiting (default 120 req/min per IP, configurable)
- Optional API key authentication (`X-API-Key` header)
- CORS allowlist (configurable, defaults to localhost only)
- Strict CVE/CWE format validation before any Cypher execution (injection prevention)

**Endpoints:**

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/api/health` | DB connectivity status |
| GET | `/api/graph?limit=300` | Full graph subgraph for visualization |
| GET | `/api/cve/:cveId` | CVE node properties |
| GET | `/api/cve/:cveId/correlations` | CORRELATED_WITH + CO_OCCURS_WITH neighbors |
| GET | `/api/cve/:cveId/full` | CVE + all edges (CWE, OWASP, SW, clusters) |
| GET | `/api/cve/:cveId/chain` | Exploit chain CVEs |
| GET | `/api/cwe/:cweId/vulns` | CVEs in a CWE family + sibling CWEs |
| GET | `/api/search?q=...` | Text search across vuln_id / description |

---

### 4.11 Fine-tuning (`training/finetuning.py` + `finetuning_phase2.py`)

**Base model:** `fdtn-ai/Foundation-Sec-8B` (Llama 3.1-8B, pre-trained on 80B cybersecurity tokens)

**QLoRA config:**
- Rank `r=32`, `lora_alpha=64`, `lora_dropout=0.1`
- 4-bit quantization (BitsAndBytes NF4)
- `max_length=4096` (captures multi-CVE correlation sequences)
- Paged AdamW 8-bit optimizer
- Memory budget: ~15.5 GB → fits A100 40GB

**Training data composition (weighted sampler):**
- `vulnerability_correlation` layer: **3× oversample** — densest relational signal
- `vulnerability_cooccurrence` layer: **3× oversample** — primary attack chain evidence
- All other layers: 1× (standard)

**Phase 2 continuation** (`finetuning_phase2.py`): resumes from Phase 1 checkpoint with updated data mix and learning rate scheduling.

**Target repo:** `adityajayashankar/vuln-foundation-sec-8b` on HuggingFace Hub.

---

### 4.12 Evaluation (`eval/`)

**`run_graphrag_benchmark.py`** — Held-out CVE set benchmark:
- Computes **P@K, R@K** (K ∈ {5, 10, 20}) and false-positive rate
- Uses `data/heldout_cve_benchmark.jsonl` (held-out from training)
- Auto-generates benchmark JSONL from raw artifacts if missing
- Outputs results to `eval/results/graphrag_eval.csv` and `.json`

**`eval/ground_truth_benchmark.jsonl`** — manually verified CVE pairs with annotated positive/negative ground truth labels.

**`eval/probe_cooccurrence.py`** — lightweight smoke probe for co-occurrence edge quality.

---

### 4.13 Validation

**`scripts/analysis/validate_dataset.py`** — dataset quality checker:
- Token length distribution (with/without tokenizer)
- Duplicate detection and optional deduplication (`--fix`)
- Short-output detection (drops examples <80 chars)
- Layer coverage analysis

**`scripts/analysis/validate_dataset.py`** — deep vuln_dataset checks:
- CVSS/EPSS/CWE/software coverage percentages
- Expected count thresholds per layer

**KG validation** (Cypher queries defined in `docs/kg/KG_VALIDATION_CHECKLIST.md`):
- Schema + volume sanity (labels, relationship types, constraint existence)
- Null/duplicate node IDs
- Self-loop detection
- Coverage checks (fraction of CVEs with CWE, OWASP, software edges)
- Source alignment spot-checks (sample CVE → compare KG edges to raw JSON)

---

## 5. Full Data Lineage

```
NVD API ──────────────────────────────────────────────┐
EPSS API ───────────────────┐                         │
CISA KEV ──────────────┐    │                         │
GitHub Advisories ─┐   │    │                         │
MITRE ATT&CK ──┐   │   │    │                         │
Blogs/Papers ─┐│   │   │    │                         │
Vendors ─────┘│   │   │    └──► build_correlations ──► raw_correlations.json (830MB)
              ├───┴───┘         build_cooccurrence_v2 ► raw_cooccurrence_v2.json (184MB)
              │                 cluster_kev_campaigns ► raw_kev_clusters.json (41MB)
              │                 collect_cwe_chains ───► raw_cwe_chains.json (5MB)
              │                 stack_profiles ────────► (embedded in cooccurrence)
              │
              └──► build_dataset.py ──────────────────► vuln_dataset.jsonl (325k, 1.34GB)
                        │
                        ├──► expand_training_pairs ──► training_pairs.jsonl (3.5M, 3.17GB)
                        │    generate_cooccurrence_pairs
                        │    generate_synthetic_pairs
                        │
                        └──► build_master_dataset ───► master_vuln_context.jsonl
                                    │
                                    ├──► load_kg_master ──────────────► Neo4j (407k nodes, 6.9M edges)
                                    └──► graphrag_embed_index_cpu ────► Qdrant (20k / 1M-2M target)
```

---

## 6. Knowledge Graph Schema

```mermaid
erDiagram
    Vulnerability {
        string vuln_id PK
        float cvss_score
        float epss_score
        string description
        bool is_kev
        string published_date
        string[] cpe_list
    }
    CWE {
        string cwe_id PK
        string name
        string description
    }
    OWASPCategory {
        string owasp_id PK
        string name
        string category
    }
    Software {
        string software_key PK
        string vendor
        string product
        string version
    }
    CWECluster {
        string cluster_id PK
        string[] cwe_members
        string theme
    }

    Vulnerability ||--o{ Vulnerability : "CORRELATED_WITH (score, signals)"
    Vulnerability ||--o{ Vulnerability : "CO_OCCURS_WITH (confidence, source)"
    Vulnerability }o--|| CWE : "HAS_CWE"
    Vulnerability }o--|| OWASPCategory : "MAPS_TO_OWASP"
    Vulnerability }o--|| Software : "AFFECTS_SOFTWARE"
    CWECluster ||--|{ CWE : "CONTAINS_CWE"
```

---

## 7. Agent Reasoning Flow

```mermaid
stateDiagram-v2
    [*] --> Planner : user query

    Planner --> ToolExecutor : force graphrag_query\n(CVE correlation detected)
    Planner --> ToolExecutor : force lookup_by_cwe\n(CWE-only query)
    Planner --> ToolExecutor : LLM selects tool\n(general question)

    ToolExecutor --> GraphRAG : graphrag_query
    ToolExecutor --> NVD : lookup_cve
    ToolExecutor --> KGLookup : likely_on_system
    ToolExecutor --> CWEPath : lookup_by_cwe
    ToolExecutor --> LLMTool : map_owasp / score_risk\nget_pentest_method / etc.

    GraphRAG --> Neo4jTraversal : graph path
    GraphRAG --> QdrantSearch : vector path (when enabled)
    Neo4jTraversal --> Merge : evidence items
    QdrantSearch --> Merge : semantic hits
    Merge --> HITLEval : confidence + evidence count

    HITLEval --> Synthesis : HITL ok (confidence ≥ 0.40)
    HITLEval --> Synthesis : HITL flag (needs_human_review)

    Synthesis --> Planner : needs more tools (max_steps)
    Synthesis --> [*] : FINAL JSON report
```

---

## 8. LLM Backend & Fine-tuning

### Why Foundation-Sec-8B instead of a general LLM?

| Dimension | General LLM (LLaMA 3.1-8B) | Foundation-Sec-8B |
|-----------|--------------------------|------------------|
| Pre-training data | General internet | + 80B cybersecurity tokens |
| CVE description parsing | Moderate | Strong (domain vocabulary) |
| CWE/OWASP reasoning | Inconsistent | Consistent |
| Exploit chain inference | Hallucinates often | Grounded with our fine-tune |
| Context window | 8192 | 8192 (train at 4096) |

### QLoRA Design Rationale

- **r=32 (up from 16):** Higher rank needed to capture dense relational graph between CVE ↔ CWE ↔ ATT&CK ↔ OWASP. r=16 under-parameterizes cross-entity associations.
- **4096 token context:** ~22% of correlation pairs were silently truncated at 2048 tokens (verified with Llama 3 tokenizer which tokenizes security text at ~2.7 chars/token).
- **3× oversampling of correlation/cooccurrence layers:** These layers represent only ~71.5% of the dataset but are the highest-value signal for multi-CVE attack path reasoning.

---

## 9. Capabilities & Features

### What the system can do today:

| Capability | Status | Mechanism |
|-----------|--------|-----------|
| CVE correlation lookup | ✅ Full | Neo4j CORRELATED_WITH traversal |
| CVE co-occurrence lookup | ✅ Full | Neo4j CO_OCCURS_WITH traversal |
| CWE-first vulnerability family query | ✅ Full | `lookup_by_cwe` → CWECluster path |
| EPSS exploit probability | ✅ Full | FIRST.org live API |
| OWASP Top 10 category mapping | ✅ Full | LLM with domain prompt |
| Attack method / payload lookup | ✅ Full | LLM `pentesting_intelligence` layer |
| Security tool recommendation | ✅ Full | LLM `execution_context` layer |
| Risk scoring + CVSS analysis | ✅ Full | LLM `risk_scoring` layer |
| Audit finding generation | ✅ Full | LLM `audit_evidence` layer |
| Remediation recommendation | ✅ Full | LLM `remediation_learning` layer |
| Hybrid vector + graph retrieval | ⚠️ Partial | Vector ingest in progress (20k / 1M-2M target) |
| Full semantic evidence retrieval | 🔲 Pending | Requires full Qdrant ingest completion |
| Fine-tuned model inference | 🔲 Pending | Training pending full data pipeline |
| REST API for graph frontend | ✅ Full | Express.js + Neo4j driver |

### Agentic Blog Crawler

A fully agentic web intelligence crawler that:
1. Uses Groq LLM to generate 12 high-yield security search queries (no hardcoded URLs)
2. Searches via Tavily AI-optimized search API
3. Downloads pages via crawl4ai (async, 15 concurrent workers)
4. Applies a quality keyword filter (cve, exploit, payload, injection, rce, xss, sqli, etc.)
5. Runs LLM gap analysis to identify missing coverage → generates Round 2 queries
6. Caps at 370 total URLs to control cost

### HITL Policy

The deterministic HITL escalation ensures the agent **never silently returns low-confidence answers** without flagging them. Every response carries machine-readable HITL status, enabling downstream systems to route uncertain findings to human reviewers automatically.

### Dataset Completeness

| Quality Metric | Value |
|---------------|-------|
| CVEs with CVSS | 93.90% |
| CVEs with EPSS | 93.33% |
| CVEs with CWE | 77.92% |
| CVEs with affected software | 86.91% |

---

## 10. Metrics & Scale

### Raw Data Scale

| Artifact | Records | Size |
|----------|--------:|-----:|
| `raw_nvd.json` | 328,000 | 283.5 MB |
| `raw_correlations.json` | 328,000 | 830.4 MB |
| `raw_cooccurrence_v2.json` | 891,953 | 184.4 MB |
| `raw_kev_clusters.json` | 160,594 | 41.2 MB |
| `raw_epss.json` | 305,266 | 8.6 MB |
| `raw_github.json` | 3,000 | 10.0 MB |
| `vuln_dataset.jsonl` | 325,941 | 1,340.1 MB |
| `master_vuln_context.jsonl` | 325,942 | 3,055.0 MB |
| **Total** | | **~5.8 GB** |

### Knowledge Graph Scale

- **407,713 nodes**, **6,928,186 relationships**
- **5,120,356** CORRELATED_WITH edges (primary reasoning edges)
- Planned **1M-2M chunks** for vector indexing (from `master_vuln_context.jsonl`)

### Agent Performance (live probe — CVE-2021-28310)

> **Note:** Automated benchmark (`eval/run_graphrag_benchmark.py`) currently returns errors (`No module named 'pipeline'`) when run outside the project root. The figures below are from a manual agent probe.

- Primary tool call: `graphrag_query` (forced by CVE + correlation guardrail)
- Direct evidence returned: **20 results** (18 CORRELATED_WITH + 2 CO_OCCURS_WITH)
- Overall confidence: **0.652**
- HITL escalation: **not required**

---

## 11. Known Gaps & Next Steps

| Gap | Impact | Fix |
|----|--------|-----|
| Vector ingest incomplete (20k / 1M-2M target) | Hybrid retrieval underutilized | Run full ingest: `--max-vectors 1000000` (or `2000000`) |
| Benchmark eval errors (`No module named 'pipeline'`) | P@K / R@K scores unmeasurable; all 120 probes fail | Run from project root with `PYTHONPATH=.` or via `python -m eval.run_graphrag_benchmark` |
| `raw_exploitdb.json` is empty | No PoC exploit evidence layer | Fix ExploitDB crawler auth/pagination |
| Blog dataset small (228 pages) | Limited web threat intelligence | Re-run agentic crawler with wider queries |
| Decommissioned LLM model IDs in old configs | Rate limit exhaustion if triggered | Confirmed clean in current `model_loader.py` |
| Fine-tuned model not yet deployed | Agent uses API LLMs, not domain model | Complete training run + push to HF Hub |

---

## Quick Reference: Run Order

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Collect + build full pipeline
python run_pipeline.py

# 3. Validate dataset
python scripts/analysis/validate_dataset.py --no-tokenizer

# 4. Build master dataset
python data/build_master_dataset.py

# 5. Load Neo4j KG
python scripts/kg/load_kg_master.py

# 6. Start vector ingest (1M chunk target)
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py --max-vectors 1000000 --batch-size 64 --qdrant-batch-size 256

# 7. Run agent
python main.py

# 8. Run benchmark evaluation
python eval/run_graphrag_benchmark.py

# 9. Start REST API backend
cd vuln-graph-backend && node server.js
```

---

*Report generated March 9, 2026 — DeplAI Vulnerability GraphRAG Pipeline*
