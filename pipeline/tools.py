"""
tools.py
--------
Tool functions for the vulnerability agent pipeline.
Each tool corresponds to a specific dataset layer and agent role.

FIXES vs previous version:
  1. tool_likely_on_system was imported in langgraph_agent.py but never
     defined here — caused ImportError at runtime. Now fully implemented
     with a 3-tier lookup strategy:
       Tier 1 (direct):   CVE → CORRELATED_WITH / CO_OCCURS_WITH edges in Neo4j KG
       Tier 2 (2-hop):    CVE → CWE → CWECluster → other CVEs in same cluster
       Tier 3 (inferred): OWASP category co-occurrence fallback
     Returns structured JSON so the agent can distinguish direct vs inferred.

  2. tool_lookup_by_cwe added — CWE-first entry point the agent was missing.
     "Given CWE-89, what CVEs are in the same cluster and what co-occurs?"
     This was designed in the KG schema but had no tool wiring it up.

  3. Neo4j connection is read from env vars (NEO4J_URI, NEO4J_USER,
     NEO4J_PASSWORD) with no hardcoded password defaults.

  4. Both KG tools gracefully degrade to JSON-file fallback if Neo4j is
     unavailable, so the agent still works without a running DB.
"""

import os
import json
import requests
import re
from pathlib import Path

from pipeline.model_loader import ask_model

# ── Neo4j connection (lazy — only imported if KG tools are called) ─────────

_NEO4J_DRIVER = None
_CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_CWE_ID_RE = re.compile(r"^CWE-\d+$", re.IGNORECASE)


def _is_production_mode() -> bool:
    return os.getenv("APP_ENV", "").strip().lower() in {"prod", "production"}


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _normalize_likelihood(raw: float) -> float:
    """
    Convert arbitrary positive confidence/correlation score to [0,1].
    Uses monotonic compression to preserve ranking without exposing invalid probabilities.
    """
    if raw <= 0:
        return 0.0
    if raw <= 1:
        return round(float(raw), 3)
    return round(min(raw / (1.0 + raw), 0.999), 3)

def _get_neo4j_driver():
    """Lazily initialise the Neo4j driver. Returns None if unavailable."""
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER
    try:
        from neo4j import GraphDatabase
        uri  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "").strip()
        if not password:
            if _is_production_mode():
                raise RuntimeError("NEO4J_PASSWORD is required in production mode.")
            return None
        _NEO4J_DRIVER = GraphDatabase.driver(uri, auth=(user, password))
        # Quick connectivity test
        with _NEO4J_DRIVER.session() as s:
            s.run("RETURN 1")
        return _NEO4J_DRIVER
    except Exception as e:
        print(f"[tools] Neo4j unavailable ({e}) — KG tools will use JSON fallback.")
        return None


# ── JSON fallback paths (used when Neo4j is down) ─────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"

def _load_json_fallback(filename: str) -> dict | list:
    path = _DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Layer 1: Vulnerability Intelligence ───────────────────────────────────

def tool_lookup_cve(cve_id: str) -> str:
    """
    Fetch live CVE description + CWE + CVSS from NVD.
    Used by: OWASP Mapper Agent, Correlation Agent
    """
    try:
        url  = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.strip()}"
        data = requests.get(url, timeout=15).json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"CVE {cve_id} not found in NVD."

        cve    = vulns[0]["cve"]
        desc   = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "No description.")
        cwes   = [w["description"][0]["value"] for w in cve.get("weaknesses", []) if w.get("description")]
        metrics = cve.get("metrics", {})
        cvss   = ""
        for key in ["cvssMetricV31", "cvssMetricV30"]:
            if key in metrics:
                cvss = str(metrics[key][0]["cvssData"].get("baseScore", ""))
                break

        return (
            f"CVE ID: {cve_id}\n"
            f"Description: {desc}\n"
            f"CWE: {', '.join(cwes) if cwes else 'Unknown'}\n"
            f"CVSS Score: {cvss if cvss else 'Not available'}"
        )
    except Exception as e:
        return f"NVD lookup failed: {e}"


def tool_map_owasp(description: str) -> str:
    """
    Map vulnerability description to OWASP Top 10 category using the model.
    Used by: OWASP Mapper Agent
    """
    return ask_model(
        instruction="Identify the OWASP Top 10 category for the following vulnerability description.",
        context=description,
        layer="vulnerability_intelligence",
    )


# ── Layer 1b: Knowledge Graph (NEW / FIXED) ────────────────────────────────

def tool_graphrag_query(arg: str) -> str:
    """
    Hybrid GraphRAG retrieval entry point for agent-to-agent calls.

    Arg:
      JSON string preferred:
        {"query":"...", "entity":{"type":"cve|cwe","id":"..."}, "top_k":12, "max_hops":2, "use_vector":false}
      or raw text query.

    Returns:
      Strict JSON contract with evidence, citations, confidence, and HITL status.
    """
    raw = (arg or "").strip()
    req = {
        "query": raw,
        "entity": None,
        "top_k": 12,
        "max_hops": 2,
        "use_vector": os.getenv("GRAPHRAG_USE_VECTOR", "0").strip().lower() not in {"0", "false", "no"},
    }
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            req["query"] = str(parsed.get("query", raw)).strip()
            req["entity"] = parsed.get("entity")
            req["top_k"] = _clamp_int(int(parsed.get("top_k", 12)), 1, 25)
            req["max_hops"] = _clamp_int(int(parsed.get("max_hops", 2)), 1, 3)
            if "use_vector" in parsed:
                req["use_vector"] = str(parsed.get("use_vector", "")).strip().lower() not in {"0", "false", "no"}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    text = req["query"]
    if not req["entity"]:
        cve_match = re.search(r"CVE-\d{4}-\d+", text or "", re.IGNORECASE)
        cwe_match = re.search(r"CWE-\d+", text or "", re.IGNORECASE)
        if cve_match:
            req["entity"] = {"type": "cve", "id": cve_match.group(0).upper()}
        elif cwe_match:
            req["entity"] = {"type": "cwe", "id": cwe_match.group(0).upper()}

    if req["entity"]:
        ent_type = str(req["entity"].get("type", "")).lower().strip()
        ent_id = str(req["entity"].get("id", "")).upper().strip()
        if ent_type == "cve" and not _CVE_ID_RE.match(ent_id):
            return json.dumps(
                {
                    "status": "error",
                    "query": req["query"],
                    "entity": {"type": "cve", "id": ent_id},
                    "direct_evidence": [],
                    "inferred_candidates": [],
                    "citations": [],
                    "confidence_summary": {"overall": 0.0, "rationale": "Invalid CVE ID format."},
                    "hitl": {"required": False, "reasons": []},
                    "recommended_actions": [],
                    "error": "Invalid CVE format.",
                }
            )
        if ent_type == "cwe" and not _CWE_ID_RE.match(ent_id):
            return json.dumps(
                {
                    "status": "error",
                    "query": req["query"],
                    "entity": {"type": "cwe", "id": ent_id},
                    "direct_evidence": [],
                    "inferred_candidates": [],
                    "citations": [],
                    "confidence_summary": {"overall": 0.0, "rationale": "Invalid CWE ID format."},
                    "hitl": {"required": False, "reasons": []},
                    "recommended_actions": [],
                    "error": "Invalid CWE format.",
                }
            )
        req["entity"] = {"type": ent_type, "id": ent_id}

    try:
        from pipeline.graphrag.retriever import retrieve_hybrid
        from pipeline.hitl import evaluate_hitl_policy

        payload = retrieve_hybrid(
            query=req["query"],
            entity=req["entity"],
            top_k=req["top_k"],
            max_hops=req["max_hops"],
            use_vector=req["use_vector"],
        )
        hitl = evaluate_hitl_policy(payload)
        payload["hitl"] = hitl
        if hitl.get("required"):
            payload["status"] = "needs_human_review"
            payload["recommended_actions"] = []
        else:
            payload["status"] = "ok"
        return json.dumps(payload)
    except Exception as e:
        entity = req.get("entity") or {}
        ent_type = entity.get("type")
        ent_id = entity.get("id")

        if ent_type == "cve" and ent_id:
            raw_payload = json.loads(tool_likely_on_system(ent_id))
            results = raw_payload.get("results", [])
            direct = [r for r in results if str(r.get("evidence_tier", "")).startswith("direct")]
            inferred = [r for r in results if r not in direct]
            citations = [
                {
                    "citation_id": f"legacy-{i+1}",
                    "source_type": r.get("rel_type", "kg"),
                    "entity_id": r.get("cve_id", ""),
                    "snippet": ", ".join(r.get("signals", [])[:2]) or "legacy graph fallback",
                    "metadata": {"tier": r.get("evidence_tier", "unknown")},
                }
                for i, r in enumerate(results)
            ]
            return json.dumps(
                {
                    "status": "needs_human_review",
                    "query": req["query"],
                    "entity": {"type": "cve", "id": ent_id},
                    "direct_evidence": direct[: req["top_k"]],
                    "inferred_candidates": inferred[: req["top_k"]],
                    "citations": citations[: req["top_k"]],
                    "confidence_summary": {
                        "overall": round(sum(float(r.get("likelihood", 0.0)) for r in results[:5]) / max(len(results[:5]), 1), 3),
                        "rationale": f"Legacy fallback used because GraphRAG retrieval failed: {e}",
                    },
                    "hitl": {"required": True, "reasons": ["GraphRAG retrieval unavailable; using degraded fallback."]},
                    "recommended_actions": [],
                }
            )

        return json.dumps(
            {
                "status": "error",
                "query": req["query"],
                "entity": req.get("entity"),
                "direct_evidence": [],
                "inferred_candidates": [],
                "citations": [],
                "confidence_summary": {"overall": 0.0, "rationale": "GraphRAG retrieval failed."},
                "hitl": {"required": True, "reasons": ["GraphRAG retrieval failed."]},
                "recommended_actions": [],
                "error": str(e),
            }
        )


def tool_likely_on_system(arg: str) -> str:
    """
    FIX: This function was imported but never defined — caused ImportError.

    Given a CVE ID, returns likely co-present vulnerabilities using a 3-tier
    strategy against the Neo4j knowledge graph:

      Tier 1 (direct):   CORRELATED_WITH / CO_OCCURS_WITH edges  [evidence_tier=direct]
      Tier 2 (2-hop):    CVE → CWE → CWECluster → sibling CVEs   [evidence_tier=cluster]
      Tier 3 (inferred): OWASP category co-occurrence             [evidence_tier=inferred]

    Arg: CVE-ID string  OR  JSON {"cve_id": "CVE-...", "top_k": 15}

    Returns: JSON string with keys:
      query_cve, direct_count, inferred_count, results[]
        where each result has: cve_id, likelihood, evidence_tier, signals, inferred_from
    """
    # ── Parse argument ────────────────────────────────────────────────────
    cve_id = arg.strip()
    top_k  = 15
    if cve_id.startswith("{"):
        try:
            parsed = json.loads(cve_id)
            cve_id = parsed.get("cve_id", cve_id)
            top_k = _clamp_int(int(parsed.get("top_k", top_k)), 1, 50)
        except (json.JSONDecodeError, ValueError):
            pass
    cve_id = cve_id.upper().strip()
    if not _CVE_ID_RE.match(cve_id):
        return json.dumps(
            {
                "query_cve": cve_id,
                "direct_count": 0,
                "inferred_count": 0,
                "results": [],
                "error": "Invalid CVE format. Expected CVE-YYYY-NNNN.",
            }
        )

    results      = []
    direct_count = 0

    # ── Try Neo4j first ───────────────────────────────────────────────────
    driver = _get_neo4j_driver()
    if driver:
        try:
            with driver.session() as session:
                # Tier 1: direct graph edges
                tier1 = session.run(
                    """
                    MATCH (v:Vulnerability {vuln_id: $cve_id})
                          -[r:CORRELATED_WITH|CO_OCCURS_WITH]-(related:Vulnerability)
                    RETURN related.vuln_id        AS cve_id,
                           type(r)                AS rel_type,
                           coalesce(r.max_score, r.max_confidence, 0.0) AS confidence,
                           coalesce(r.signals, [])   AS signals,
                           coalesce(r.reasons, [])   AS reasons
                    ORDER BY confidence DESC
                    LIMIT $top_k
                    """,
                    cve_id=cve_id, top_k=top_k,
                ).data()

                for row in tier1:
                    raw_conf = float(row["confidence"])
                    results.append({
                        "cve_id":        row["cve_id"],
                        "likelihood":    _normalize_likelihood(raw_conf),
                        "raw_confidence": round(raw_conf, 3),
                        "evidence_tier": "direct",
                        "rel_type":      row["rel_type"],
                        "signals":       list(row["signals"])[:5],
                        "reasons":       list(row["reasons"])[:3],
                        "inferred_from": [],
                    })
                direct_count = len(results)

                # Tier 2: 2-hop via CWE cluster if direct results are sparse
                if len(results) < 5:
                    tier2 = session.run(
                        """
                        MATCH (v:Vulnerability {vuln_id: $cve_id})-[:HAS_CWE]->(w:CWE)
                              <-[:CONTAINS_CWE]-(cluster:CWECluster)
                              -[:CONTAINS_CWE]->(w2:CWE)
                              <-[:HAS_CWE]-(sibling:Vulnerability)
                        WHERE sibling.vuln_id <> $cve_id
                        RETURN DISTINCT sibling.vuln_id AS cve_id,
                               cluster.cluster_id       AS cluster_id,
                               w.cwe_id                 AS shared_cwe,
                               coalesce(sibling.epss_score, 0.0) AS epss
                        ORDER BY epss DESC
                        LIMIT $remaining
                        """,
                        cve_id=cve_id, remaining=(top_k - len(results)),
                    ).data()

                    seen = {r["cve_id"] for r in results}
                    for row in tier2:
                        if row["cve_id"] not in seen:
                            cluster_score = min(0.6, float(row["epss"]) * 2 + 0.3)
                            results.append({
                                "cve_id":        row["cve_id"],
                                "likelihood":    round(cluster_score, 3),
                                "raw_confidence": round(cluster_score, 3),
                                "evidence_tier": "cluster",
                                "rel_type":      "SAME_CWE_CLUSTER",
                                "signals":       [f"shared_cwe:{row['shared_cwe']}",
                                                  f"cluster:{row['cluster_id']}"],
                                "reasons":       [],
                                "inferred_from": [row["shared_cwe"], row["cluster_id"]],
                            })
                            seen.add(row["cve_id"])

                # Tier 3: OWASP co-occurrence if still sparse
                if len(results) < 3:
                    tier3 = session.run(
                        """
                        MATCH (v:Vulnerability {vuln_id: $cve_id})-[:MAPS_TO_OWASP]->(o:OWASPCategory)
                              <-[:MAPS_TO_OWASP]-(peer:Vulnerability)
                        WHERE peer.vuln_id <> $cve_id
                        RETURN DISTINCT peer.vuln_id AS cve_id,
                               o.owasp_id            AS owasp_id,
                               coalesce(peer.epss_score, 0.0) AS epss
                        ORDER BY epss DESC
                        LIMIT $remaining
                        """,
                        cve_id=cve_id, remaining=(top_k - len(results)),
                    ).data()

                    seen = {r["cve_id"] for r in results}
                    for row in tier3:
                        if row["cve_id"] not in seen:
                            inferred_score = min(0.4, float(row["epss"]) + 0.2)
                            results.append({
                                "cve_id":        row["cve_id"],
                                "likelihood":    round(inferred_score, 3),
                                "raw_confidence": round(inferred_score, 3),
                                "evidence_tier": "inferred",
                                "rel_type":      "SAME_OWASP",
                                "signals":       [f"owasp:{row['owasp_id']}"],
                                "reasons":       [],
                                "inferred_from": [row["owasp_id"]],
                            })
                            seen.add(row["cve_id"])

        except Exception as e:
            results = []
            direct_count = 0
            print(f"[tool_likely_on_system] Neo4j query failed: {e} — falling back to JSON")

    # ── JSON file fallback (Neo4j down or no results) ─────────────────────
    if not results:
        corr_data = _load_json_fallback("raw_correlations.json")
        cooc_data = _load_json_fallback("raw_cooccurrence.json")

        seen = set()

        # From correlations file
        if isinstance(corr_data, list):
            for rec in corr_data:
                if rec.get("cve_id", "").upper() == cve_id:
                    for rel in rec.get("related_vulnerabilities", [])[:top_k]:
                        rid = rel.get("cve_id", "").upper()
                        if rid and rid not in seen:
                            raw_score = float(rel.get("correlation_score", 0.5))
                            results.append({
                                "cve_id":        rid,
                                "likelihood":    _normalize_likelihood(raw_score),
                                "raw_correlation_score": round(raw_score, 3),
                                "evidence_tier": "direct_json",
                                "rel_type":      "CORRELATED_WITH",
                                "signals":       rel.get("signals", [])[:5],
                                "reasons":       [],
                                "inferred_from": [],
                            })
                            seen.add(rid)
                    direct_count = len(results)
                    break

        # From co-occurrence file
        pairs = []
        if isinstance(cooc_data, dict):
            pairs = cooc_data.get("cooccurrence_pairs", [])
        elif isinstance(cooc_data, list):
            pairs = cooc_data

        for pair in pairs:
            a = pair.get("cve_a", "").upper()
            b = pair.get("cve_b", "").upper()
            if a == cve_id and b not in seen:
                raw_conf = float(pair.get("confidence", 0.4))
                results.append({
                    "cve_id":        b,
                    "likelihood":    _normalize_likelihood(raw_conf),
                    "raw_confidence": round(raw_conf, 3),
                    "evidence_tier": "cooccurrence_json",
                    "rel_type":      "CO_OCCURS_WITH",
                    "signals":       [pair.get("source", "")],
                    "reasons":       [pair.get("reason", "")],
                    "inferred_from": [],
                })
                seen.add(b)
            elif b == cve_id and a not in seen:
                raw_conf = float(pair.get("confidence", 0.4))
                results.append({
                    "cve_id":        a,
                    "likelihood":    _normalize_likelihood(raw_conf),
                    "raw_confidence": round(raw_conf, 3),
                    "evidence_tier": "cooccurrence_json",
                    "rel_type":      "CO_OCCURS_WITH",
                    "signals":       [pair.get("source", "")],
                    "reasons":       [pair.get("reason", "")],
                    "inferred_from": [],
                })
                seen.add(a)

        results = sorted(results, key=lambda x: x["likelihood"], reverse=True)[:top_k]

    inferred_count = len(results) - direct_count

    payload = {
        "query_cve":      cve_id,
        "direct_count":   direct_count,
        "inferred_count": inferred_count,
        "results":        results,
    }
    return json.dumps(payload)


def tool_lookup_by_cwe(cwe_id: str) -> str:
    """
    NEW: CWE-first entry point into the knowledge graph.

    Given a CWE ID (e.g. "CWE-89"), returns:
      - CVEs in the dataset with this CWE
      - Their top co-occurring CVEs from the KG
      - The CWE cluster this weakness belongs to

    Arg: CWE-ID string (e.g. "CWE-89" or "89")

    Used by: Correlation Agent when the user asks:
      "Given CWE-89, what vulnerabilities are in the same family?"
      "What co-occurs with injection weaknesses?"
    """
    # Normalise input
    cwe_id = cwe_id.strip().upper()
    if not cwe_id.startswith("CWE-"):
        cwe_id = f"CWE-{cwe_id}"
    if not _CWE_ID_RE.match(cwe_id):
        return f"Invalid CWE format: {cwe_id}. Expected CWE-NNN."

    driver = _get_neo4j_driver()

    if driver:
        try:
            with driver.session() as session:
                # CVEs with this CWE, ordered by EPSS (exploitability)
                cve_rows = session.run(
                    """
                    MATCH (w:CWE {cwe_id: $cwe_id})<-[:HAS_CWE]-(v:Vulnerability)
                    OPTIONAL MATCH (v)-[:MAPS_TO_OWASP]->(o:OWASPCategory)
                    RETURN v.vuln_id                          AS cve_id,
                           coalesce(v.cvss_score, 0.0)        AS cvss,
                           coalesce(v.epss_score, 0.0)        AS epss,
                           coalesce(v.confirmed_exploited, false) AS kev,
                           o.owasp_id                         AS owasp
                    ORDER BY epss DESC, cvss DESC
                    LIMIT 20
                    """,
                    cwe_id=cwe_id,
                ).data()

                # CWE cluster siblings
                cluster_rows = session.run(
                    """
                    MATCH (w:CWE {cwe_id: $cwe_id})<-[:CONTAINS_CWE]-(cluster:CWECluster)
                          -[:CONTAINS_CWE]->(sibling:CWE)
                    WHERE sibling.cwe_id <> $cwe_id
                    RETURN sibling.cwe_id AS sibling_cwe,
                           cluster.cluster_id AS cluster_id
                    LIMIT 10
                    """,
                    cwe_id=cwe_id,
                ).data()

                if not cve_rows and not cluster_rows:
                    return f"No data found for {cwe_id} in the knowledge graph."

                lines = [f"Knowledge Graph results for {cwe_id}:\n"]

                if cluster_rows:
                    cluster_id = cluster_rows[0]["cluster_id"]
                    siblings   = [r["sibling_cwe"] for r in cluster_rows]
                    lines.append(f"Cluster: {cluster_id}")
                    lines.append(f"Related CWEs in same cluster: {', '.join(siblings)}\n")

                lines.append(f"Top CVEs with {cwe_id} (by EPSS score):")
                for r in cve_rows[:15]:
                    kev_flag = " [KEV]" if r["kev"] else ""
                    lines.append(
                        f"  {r['cve_id']}{kev_flag} | CVSS={r['cvss']:.1f} "
                        f"| EPSS={float(r['epss']):.3f} | {r['owasp'] or 'OWASP unknown'}"
                    )

                return "\n".join(lines)

        except Exception as e:
            print(f"[tool_lookup_by_cwe] Neo4j query failed: {e} — falling back to JSON")

    # JSON fallback
    corr_data = _load_json_fallback("raw_correlations.json")
    results = []
    if isinstance(corr_data, list):
        for rec in corr_data:
            if rec.get("cwe_id", "").upper() == cwe_id:
                results.append(
                    f"  {rec['cve_id']} | CVSS={rec.get('cvss_score', '?')} "
                    f"| {rec.get('owasp_category', 'OWASP unknown')}"
                )
    if results:
        return f"CVEs with {cwe_id} (JSON fallback):\n" + "\n".join(results[:20])
    return f"No data found for {cwe_id} (Neo4j unavailable, JSON fallback empty)."


# ── Layer 2: Pentesting Intelligence ──────────────────────────────────────

def tool_get_pentest_method(vuln_description: str) -> str:
    """
    Returns attack method, payload examples, and detection signals.
    Used by: Tool Selector Agent, Execution Planner Agent
    """
    return ask_model(
        instruction="Describe how to test for this vulnerability during a pentest. Include attack method, payload examples, and detection signals.",
        context=vuln_description,
        layer="pentesting_intelligence",
    )


def tool_select_tool(owasp_category: str, tech_stack: str = "") -> str:
    """
    Recommend the best security tool for a given vulnerability type.
    Used by: Tool Selector Agent
    """
    ctx = f"OWASP Category: {owasp_category}"
    if tech_stack:
        ctx += f"\nTech Stack: {tech_stack}"
    return ask_model(
        instruction="Which security testing tool should be used and why?",
        context=ctx,
        layer="execution_context",
    )


# ── Layer 3: Risk & Scoring ────────────────────────────────────────────────

def tool_fetch_epss(cve_id: str) -> str:
    """
    Fetch live EPSS exploit probability score from FIRST API.
    Used by: Base Scorer Agent, Severity Adjuster Agent
    """
    try:
        url  = f"https://api.first.org/data/v1/epss?cve={cve_id.strip()}"
        data = requests.get(url, timeout=15).json()
        items = data.get("data", [])
        if not items:
            return f"No EPSS data for {cve_id}"
        epss = items[0].get("epss", "N/A")
        pct  = items[0].get("percentile", "N/A")
        return (
            f"EPSS Score for {cve_id}: {epss}\n"
            f"Percentile: {pct} (higher = more likely to be exploited)"
        )
    except Exception as e:
        return f"EPSS lookup failed: {e}"


def tool_score_risk(cve_description: str, cvss: str = "", epss: str = "") -> str:
    """
    Generate full risk assessment using the model.
    Used by: Base Scorer Agent
    """
    ctx = cve_description
    if cvss:
        ctx += f"\nCVSS Score: {cvss}"
    if epss:
        ctx += f"\nEPSS Score: {epss}"
    return ask_model(
        instruction="Perform a risk assessment. Include risk level, business impact, and whether this should be treated as priority.",
        context=ctx,
        layer="risk_scoring",
    )


# ── Layer 4: Audit Evidence ────────────────────────────────────────────────

def tool_generate_finding(
    vuln_name:   str,
    cve_id:      str,
    description: str,
    cvss:        str,
    owasp:       str,
) -> str:
    """
    Generate a structured audit finding in report-ready format.
    Used by: Reporting Agent, Result Aggregator Agent
    """
    ctx = (
        f"Vulnerability Name: {vuln_name}\n"
        f"CVE ID: {cve_id}\n"
        f"Description: {description}\n"
        f"CVSS Score: {cvss}\n"
        f"OWASP Category: {owasp}"
    )
    return ask_model(
        instruction="Generate a formal audit finding summary including severity, evidence, and affected controls.",
        context=ctx,
        layer="audit_evidence",
    )


# ── Layer 5 & 6: Remediation Learning ─────────────────────────────────────

def tool_get_remediation(vuln_description: str) -> str:
    """
    Generate remediation advice and root cause analysis.
    Used by: Reflector Agent, Remediation Trainer Agent
    """
    return ask_model(
        instruction="Provide remediation steps and root cause analysis for this vulnerability.",
        context=vuln_description,
        layer="remediation_learning",
    )
