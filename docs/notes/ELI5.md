# DeplAI — Explained Like You're 5

## What is this project?

**Imagine your computer has locks. Some locks are broken — those are called "vulnerabilities." Hackers find and exploit these broken locks.**

This project is a **giant brain that knows about ~326,000 broken locks** (CVEs), understands *which ones tend to appear together* on the same computer, and can answer questions like:

> *"If I know lock A is broken on this system, what other broken locks probably exist there too?"*

---

## The 4 Big Parts

### 1. The Data Collector (the "Library")

The system crawls **10+ real-world sources** and pulls in everything security-related:

| Source | What it gives |
|---|---|
| **NVD** (Govt database) | 328,000 CVE descriptions + severity scores |
| **CISA KEV** | 1,500 CVEs that are *actively being exploited right now* |
| **EPSS** | Daily probability scores: "how likely is this CVE to be exploited today?" |
| **GitHub Advisories** | 3,000 open-source library vulnerabilities |
| **MITRE ATT&CK** | How hackers chain vulnerabilities into attack techniques |
| **Security Blogs** | AI-driven web crawling — finds and reads blog posts about vulnerability chains |
| **Research Papers** | 1,450 academic papers on exploit patterns |
| **Vendor Advisories** | Microsoft, Cisco, Red Hat patch notices |

Total raw data: **~4.9 GB** across 10 JSON files.

---

### 2. The Knowledge Graph (the "Brain")

All that data gets loaded into **Neo4j** — a graph database — as:

- **407,000 nodes** (things: CVEs, software products, weakness types, etc.)
- **6.9 million edges** (relationships: "CVE-A is correlated with CVE-B", "CVE-X affects Windows Server 2019", etc.)

Think of it like a **city map of vulnerabilities**, where roads between cities represent "these two vulnerabilities tend to appear together." There are 8 different *types* of roads:

- Same product affected by both CVEs
- Same tech stack fingerprint
- Same vendor in active-exploit catalog
- Same ransomware campaign
- One weakness logically *enables* the other (CWE chains)
- Temporal clustering (patched in same window)
- and more...

---

### 3. The AI Agent (the "Detective")

When you ask a question like *"CVE-2021-28310 — what else might be on this system?"*, a **LangGraph state machine** (a 3-step AI loop) kicks in:

```
Planner → picks the right tool
    ↓
Tool Executor → queries Neo4j graph + Qdrant vector DB
    ↓
Synthesis → LLM writes a structured JSON answer with evidence, confidence, and citations
```

It uses a **fallback LLM chain**: tries Groq (cloud, fast) → OpenRouter (free tier) → local Ollama. So it works even offline.

The output is a **strict schema**: not free text, but a structured contract with `direct_evidence`, `inferred_candidates`, `confidence_summary`, and a `hitl` (human review flag) if confidence is low.

---

### 4. The Training Data + Fine-Tuning (the "Teacher")

The system also generates **2.6 million training pairs** from the graph, like:

> *Input:* "Given CVE-2021-28310, what other vulnerabilities commonly co-exist?"
> *Output:* "CVE-2021-44826 co-occurs due to product_cooccurrence in Windows Server 2016–2022"

These pairs are used to **fine-tune Foundation-Sec-8B** (a security-specialized LLM) with QLoRA — making a model that deeply understands vulnerability relationships from first principles, not just pattern matching.

---

## How Complex Is It?

This is a **full ML system**, not a script:

| Dimension | Scale |
|---|---|
| Data pipeline | 18-step orchestrated pipeline |
| Graph size | 407k nodes, 6.9M edges |
| Retrieval | Hybrid: graph traversal (Neo4j) + vector search (Qdrant, 1M-2M planned chunks) |
| Agent | LangGraph state machine with 12 tools, 15-step loop, HITL policy |
| Evaluation | Precision@K / Recall@K / FP-rate benchmark with held-out test set |
| Backend | Express.js REST API for graph visualization |
| Infra | 3 databases (Neo4j, Qdrant, vector cache), 3 LLM backends |

---

## Where Can This Be Used?

**1. Penetration Testing / Red Teams**
> "I found CVE-X on this box — what else should I check for?" — the agent gives you a ranked, evidence-backed list instantly.

**2. Vulnerability Management / Patch Prioritization**
> Instead of "patch all critical CVEs," teams can ask "which CVEs on our stack tend to cluster together?" and prioritize the ones that form attack chains.

**3. Security Auditing**
> The `generate_finding` tool produces structured audit findings from a CVE ID — ready to drop into a pentest report.

**4. Threat Intelligence Platforms**
> The REST API + graph data can power interactive dashboards showing CVE relationship networks visually.

**5. Fine-Tuned Security LLM**
> The trained `vuln-foundation-sec-8b` model can be deployed in any security chatbot, SIEM, or SOAR platform.

---

## How Valuable Is This?

**Very.** Here's why:

- **Commercial equivalents** (Recorded Future, Vulcan Cyber, Tenable One) charge **$50k–$200k/yr** for similar relationship intelligence.
- The **CISA KEV + EPSS + co-occurrence fusion** is something most teams do manually in spreadsheets today.
- The **2.6M training dataset** for security vulnerability reasoning doesn't publicly exist at this scale.
- The **hybrid GraphRAG** (graph + vector) approach is state-of-the-art — most RAG systems only do vector search, missing the structural multi-hop graph reasoning.
- It's **open and reproducible** — the full pipeline rebuilds from scratch with `python run_pipeline.py`.

The main limitation right now is scale of infrastructure: Neo4j + Qdrant + a GPU for fine-tuning is needed for the full setup, which makes it a research/enterprise-grade project rather than a quick install.
