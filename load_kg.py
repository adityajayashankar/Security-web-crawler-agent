"""
load_kg.py
----------
Load vulnerability intelligence artifacts into Neo4j.

Usage:
  python load_kg.py

Required environment:
  NEO4J_PASSWORD

Optional environment:
  NEO4J_URI   (default: bolt://localhost:7687)
  NEO4J_USER  (default: neo4j)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", required=True)

DATA_DIR = Path("data")
BATCH_SIZE = int(os.getenv("KG_BATCH_SIZE", "2000"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def chunked(seq: list[dict], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_cwe_cluster_map() -> dict[str, set[str]]:
    """
    Build cluster -> set(CWE IDs) map.
    Priority:
      1) data/cwe_clusters.json (if present)
      2) fallback to data.build_dataset._CWE_CLUSTERS
    """
    explicit = DATA_DIR / "cwe_clusters.json"
    if explicit.exists():
        raw = load_json(explicit)
        out: dict[str, set[str]] = defaultdict(set)
        if isinstance(raw, dict):
            for cluster_id, payload in raw.items():
                members = payload.get("members", []) if isinstance(payload, dict) else []
                for cwe in members:
                    if isinstance(cwe, str) and cwe.upper().startswith("CWE-"):
                        out[cluster_id].add(cwe.upper())
        elif isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                cluster_id = str(row.get("cluster", "")).strip() or str(row.get("cluster_id", "")).strip()
                cwe_id = str(row.get("cwe_id", "")).upper().strip()
                if cluster_id and cwe_id.startswith("CWE-"):
                    out[cluster_id].add(cwe_id)
        return out

    try:
        from data.build_dataset import _CWE_CLUSTERS  # noqa: WPS437
    except Exception:
        return {}

    out: dict[str, set[str]] = defaultdict(set)
    for cwe, payload in _CWE_CLUSTERS.items():
        cluster = str(payload.get("cluster", "")).strip()
        if not cluster:
            continue
        cwe_norm = str(cwe).upper().strip()
        if cwe_norm.startswith("CWE-"):
            out[cluster].add(cwe_norm)
        for sibling in payload.get("co_present", []):
            sib = str(sibling).upper().strip()
            if sib.startswith("CWE-"):
                out[cluster].add(sib)
    return out


print("Creating constraints...")
with driver.session() as s:
    for q in [
        "CREATE CONSTRAINT vuln_id_unique IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.vuln_id IS UNIQUE",
        "CREATE CONSTRAINT cwe_id_unique IF NOT EXISTS FOR (n:CWE) REQUIRE n.cwe_id IS UNIQUE",
        "CREATE CONSTRAINT owasp_id_unique IF NOT EXISTS FOR (n:OWASPCategory) REQUIRE n.owasp_id IS UNIQUE",
        "CREATE CONSTRAINT software_key_unique IF NOT EXISTS FOR (n:Software) REQUIRE n.software_key IS UNIQUE",
        "CREATE CONSTRAINT cwe_cluster_id_unique IF NOT EXISTS FOR (n:CWECluster) REQUIRE n.cluster_id IS UNIQUE",
        "CREATE CONSTRAINT stack_profile_id_unique IF NOT EXISTS FOR (n:StackProfile) REQUIRE n.profile_id IS UNIQUE",
    ]:
        s.run(q)
print("  Done.")


# 1) Vulnerability records
print("Loading vuln_dataset.jsonl...")
records = load_jsonl(DATA_DIR / "vuln_dataset.jsonl")
print(f"  {len(records):,} records — ingesting...")

with driver.session() as s:
    vuln_rows: list[dict] = []
    cwe_edges: list[dict] = []
    owasp_edges: list[dict] = []
    sw_edges: list[dict] = []

    for row in records:
        vid = (row.get("cve_id") or row.get("ghsa_id") or "").strip().upper()
        if not vid:
            continue
        vuln_rows.append(
            {
                "vid": vid,
                "name": row.get("vulnerability_name", ""),
                "desc": (row.get("description") or "")[:500],
                "cvss": float(row["cvss_score"]) if row.get("cvss_score") else None,
                "epss": float(row["epss_score"]) if row.get("epss_score") else None,
                "kev": bool(row.get("confirmed_exploited")),
                "risk": row.get("risk_level", ""),
            }
        )

        cwe = (row.get("cwe_id") or "").strip().upper()
        if cwe.startswith("CWE-"):
            cwe_edges.append({"vid": vid, "cwe": cwe})

        owasp = (row.get("owasp_category") or "").strip()
        if owasp.startswith("A"):
            owasp_edges.append({"vid": vid, "owasp": owasp})

        for sw in (row.get("affected_software") or []):
            sw_norm = str(sw).strip().lower()
            if sw_norm:
                sw_edges.append({"vid": vid, "sw": sw_norm})

    processed = 0
    for batch in chunked(vuln_rows, BATCH_SIZE):
        s.run(
            """
            UNWIND $rows AS row
            MERGE (v:Vulnerability {vuln_id: row.vid})
            SET v.cve_id              = CASE WHEN row.vid STARTS WITH 'CVE-'  THEN row.vid ELSE v.cve_id END,
                v.ghsa_id             = CASE WHEN row.vid STARTS WITH 'GHSA-' THEN row.vid ELSE v.ghsa_id END,
                v.vulnerability_name  = coalesce(v.vulnerability_name, row.name),
                v.description         = coalesce(v.description, row.desc),
                v.cvss_score          = coalesce(v.cvss_score, row.cvss),
                v.epss_score          = coalesce(v.epss_score, row.epss),
                v.confirmed_exploited = coalesce(v.confirmed_exploited, row.kev),
                v.risk_level          = coalesce(v.risk_level, row.risk)
            """,
            rows=batch,
        )
        processed += len(batch)
        if processed % 10000 == 0:
            print(f"  {processed:,} / {len(vuln_rows):,}")

    for batch in chunked(cwe_edges, BATCH_SIZE):
        s.run(
            """
            UNWIND $rows AS row
            MERGE (w:CWE {cwe_id: row.cwe})
            MERGE (v:Vulnerability {vuln_id: row.vid})
            MERGE (v)-[:HAS_CWE]->(w)
            """,
            rows=batch,
        )

    for batch in chunked(owasp_edges, BATCH_SIZE):
        s.run(
            """
            UNWIND $rows AS row
            MERGE (o:OWASPCategory {owasp_id: row.owasp})
            MERGE (v:Vulnerability {vuln_id: row.vid})
            MERGE (v)-[:MAPS_TO_OWASP]->(o)
            """,
            rows=batch,
        )

    for batch in chunked(sw_edges, BATCH_SIZE):
        s.run(
            """
            UNWIND $rows AS row
            MERGE (p:Software {software_key: row.sw})
            SET p.name = row.sw
            MERGE (v:Vulnerability {vuln_id: row.vid})
            MERGE (v)-[:AFFECTS_SOFTWARE]->(p)
            """,
            rows=batch,
        )

print("  vuln_dataset done.")


# 2) CWE clusters for tier-2 graph traversals
print("Loading CWE clusters...")
cluster_map = load_cwe_cluster_map()
if cluster_map:
    with driver.session() as s:
        cluster_rows = []
        for cluster_id, cwes in cluster_map.items():
            for cwe in sorted(cwes):
                cluster_rows.append({"cluster_id": cluster_id, "cwe": cwe})
        for batch in chunked(cluster_rows, BATCH_SIZE):
            s.run(
                """
                UNWIND $rows AS row
                MERGE (w:CWE {cwe_id: row.cwe})
                MERGE (c:CWECluster {cluster_id: row.cluster_id})
                MERGE (c)-[:CONTAINS_CWE]->(w)
                """,
                rows=batch,
            )
    print(f"  CWE clusters done. cluster_count={len(cluster_map):,}")
else:
    print("  No CWE cluster source found — skipping.")


# 3) Correlations
print("Loading raw_correlations.json...")
corr_path = DATA_DIR / "raw_correlations.json"
if corr_path.exists():
    corr_data = load_json(corr_path)
    if not isinstance(corr_data, list):
        corr_data = []

    with driver.session() as s:
        edge_rows = []
        for row in corr_data:
            src = (row.get("cve_id") or "").strip().upper()
            if not src:
                continue
            for rel in (row.get("related_vulnerabilities") or []):
                tgt = (rel.get("cve_id") or "").strip().upper()
                if not tgt or tgt == src:
                    continue
                a, b = (src, tgt) if src < tgt else (tgt, src)
                edge_rows.append(
                    {
                        "a": a,
                        "b": b,
                        "score": float(rel.get("correlation_score", 0.5)),
                        "signals": rel.get("signals", []),
                    }
                )
        for batch in chunked(edge_rows, BATCH_SIZE):
            s.run(
                """
                UNWIND $rows AS row
                MERGE (a:Vulnerability {vuln_id: row.a})
                MERGE (b:Vulnerability {vuln_id: row.b})
                MERGE (a)-[r:CORRELATED_WITH]->(b)
                ON CREATE SET r.max_score = row.score,
                              r.signals   = row.signals
                ON MATCH SET  r.max_score = CASE WHEN row.score > r.max_score
                                            THEN row.score ELSE r.max_score END
                """,
                rows=batch,
            )
    print("  raw_correlations done.")
else:
    print("  raw_correlations.json not found — skipping.")


# 4) Co-occurrence
print("Loading raw_cooccurrence_v2.json...")
cooc_path = DATA_DIR / "raw_cooccurrence_v2.json"
if cooc_path.exists():
    cooc_data = load_json(cooc_path)
    pairs = cooc_data.get("cooccurrence_pairs", []) if isinstance(cooc_data, dict) else cooc_data

    with driver.session() as s:
        edge_rows = []
        for pair in pairs:
            a_id = (pair.get("cve_a") or "").strip().upper()
            b_id = (pair.get("cve_b") or "").strip().upper()
            if not a_id or not b_id or a_id == b_id:
                continue
            a, b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            edge_rows.append(
                {
                    "a": a,
                    "b": b,
                    "conf": float(pair.get("confidence", 0.5)),
                    "src": pair.get("source", "raw_cooccurrence_v2"),
                    "reason": pair.get("reason", ""),
                }
            )
        processed = 0
        for batch in chunked(edge_rows, BATCH_SIZE):
            s.run(
                """
                UNWIND $rows AS row
                MERGE (a:Vulnerability {vuln_id: row.a})
                MERGE (b:Vulnerability {vuln_id: row.b})
                MERGE (a)-[r:CO_OCCURS_WITH]->(b)
                ON CREATE SET r.max_confidence = row.conf,
                              r.sources        = [row.src],
                              r.reasons        = [row.reason]
                ON MATCH SET  r.max_confidence = CASE WHEN row.conf > r.max_confidence
                                                 THEN row.conf ELSE r.max_confidence END
                """,
                rows=batch,
            )
            processed += len(batch)
            if processed % 100000 == 0:
                print(f"  co-occ edges {processed:,} / {len(edge_rows):,}")
    print("  raw_cooccurrence_v2 done.")
else:
    print("  raw_cooccurrence_v2.json not found — skipping.")


print("\nKG load complete. Verifying...")
with driver.session() as s:
    v = s.run("MATCH (v:Vulnerability) RETURN count(v) AS c").single()["c"]
    cw = s.run("MATCH (w:CWE) RETURN count(w) AS c").single()["c"]
    cc = s.run("MATCH (c:CWECluster) RETURN count(c) AS c").single()["c"]
    co = s.run("MATCH ()-[r:CORRELATED_WITH]->() RETURN count(r) AS c").single()["c"]
    oc = s.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS c").single()["c"]
    print(f"  Vulnerability nodes : {v:,}")
    print(f"  CWE nodes           : {cw:,}")
    print(f"  CWECluster nodes    : {cc:,}")
    print(f"  CORRELATED_WITH     : {co:,}")
    print(f"  CO_OCCURS_WITH      : {oc:,}")

driver.close()
print("\nDone.")
