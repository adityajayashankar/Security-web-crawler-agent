"""
load_kg_master.py
-----------------
Load the Neo4j knowledge graph from a single combined master dataset file:
  data/master_vuln_context.jsonl

Usage:
  python scripts/kg/load_kg_master.py
  python scripts/kg/load_kg_master.py --master-file data/master_vuln_context.jsonl
  python scripts/kg/load_kg_master.py --max-rows 50000

Required environment:
  NEO4J_PASSWORD

Optional environment:
  NEO4J_URI   (default: bolt://localhost:7687)
  NEO4J_USER  (default: neo4j)
  KG_BATCH_SIZE (default: 2000)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

from neo4j import GraphDatabase


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_cve(value: Any) -> str:
    return _normalize(value).upper()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _normalize_negative_rule_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    profile = _normalize(obj.get("profile"))
    condition = _normalize(obj.get("condition"))
    if not profile or not condition:
        return None

    absent = []
    for c in obj.get("absent_cves", []) if isinstance(obj.get("absent_cves", []), list) else []:
        cv = _normalize_cve(c)
        if cv.startswith("CVE-"):
            absent.append(cv)

    still = []
    for c in obj.get("still_assess", []) if isinstance(obj.get("still_assess", []), list) else []:
        cv = _normalize_cve(c)
        if cv.startswith("CVE-"):
            still.append(cv)

    return {
        "profile": profile,
        "display": _normalize(obj.get("display")),
        "condition": condition,
        "reason": _normalize(obj.get("reason")),
        "absent_cves": list(dict.fromkeys(absent)),
        "still_assess": list(dict.fromkeys(still)),
    }


def _iter_named_array_stream(path: Path, key_name: str) -> Iterator[Any]:
    """
    Stream parse a named top-level array from a JSON object file.
    Example: key_name='negative_rules' for raw_cooccurrence_v2.json
    """
    if not path.exists():
        return

    decoder = json.JSONDecoder()
    key = f'"{key_name}"'
    in_array = False
    buf = ""

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            buf += chunk

            if not in_array:
                key_idx = buf.find(key)
                if key_idx < 0:
                    if len(buf) > len(key):
                        buf = buf[-len(key) :]
                    continue
                arr_idx = buf.find("[", key_idx)
                if arr_idx < 0:
                    continue
                buf = buf[arr_idx + 1 :]
                in_array = True

            while in_array:
                stripped = buf.lstrip()
                consumed = len(buf) - len(stripped)
                if consumed:
                    buf = stripped
                if not buf:
                    break
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break
                buf = buf[end:]
                yield obj


def _load_dotenv_if_present() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        # Non-fatal: shell env may already be configured.
        return


def _get_owasp_mapper():
    try:
        from data.owasp_mapper import get_owasp_category

        return get_owasp_category
    except Exception:
        return lambda _cwe: "Unknown"


def _flush_vuln_rows(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
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
        rows=rows,
    )


def _flush_cwe_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (w:CWE {cwe_id: row.cwe})
        MERGE (v:Vulnerability {vuln_id: row.vid})
        MERGE (v)-[:HAS_CWE]->(w)
        """,
        rows=rows,
    )


def _flush_owasp_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (o:OWASPCategory {owasp_id: row.owasp})
        MERGE (v:Vulnerability {vuln_id: row.vid})
        MERGE (v)-[:MAPS_TO_OWASP]->(o)
        """,
        rows=rows,
    )


def _flush_sw_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (p:Software {software_key: row.sw})
        SET p.name = row.sw
        MERGE (v:Vulnerability {vuln_id: row.vid})
        MERGE (v)-[:AFFECTS_SOFTWARE]->(p)
        """,
        rows=rows,
    )


def _flush_corr_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (a:Vulnerability {vuln_id: row.a})
        MERGE (b:Vulnerability {vuln_id: row.b})
        MERGE (a)-[r:CORRELATED_WITH]->(b)
        ON CREATE SET r.max_score = row.score,
                      r.signals   = row.signals
        ON MATCH SET  r.max_score = CASE WHEN row.score > coalesce(r.max_score, 0.0)
                                    THEN row.score ELSE r.max_score END,
                      r.signals = reduce(
                        acc = coalesce(r.signals, []),
                        s IN row.signals |
                        CASE WHEN s IN acc THEN acc ELSE acc + s END
                      )
        """,
        rows=rows,
    )


def _flush_cooc_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (a:Vulnerability {vuln_id: row.a})
        MERGE (b:Vulnerability {vuln_id: row.b})
        MERGE (a)-[r:CO_OCCURS_WITH]->(b)
        ON CREATE SET r.max_confidence = row.conf,
                      r.sources        = CASE WHEN row.src = '' THEN [] ELSE [row.src] END,
                      r.reasons        = CASE WHEN row.reason = '' THEN [] ELSE [row.reason] END
        ON MATCH SET  r.max_confidence = CASE WHEN row.conf > coalesce(r.max_confidence, 0.0)
                                         THEN row.conf ELSE r.max_confidence END,
                      r.sources = CASE
                        WHEN row.src = '' OR row.src IN coalesce(r.sources, [])
                        THEN coalesce(r.sources, [])
                        ELSE coalesce(r.sources, []) + row.src
                      END,
                      r.reasons = CASE
                        WHEN row.reason = '' OR row.reason IN coalesce(r.reasons, [])
                        THEN coalesce(r.reasons, [])
                        ELSE coalesce(r.reasons, []) + row.reason
                      END
        """,
        rows=rows,
    )


def _flush_profile_edges(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session.run(
        """
        UNWIND $rows AS row
        MERGE (sp:StackProfile {profile_id: row.profile})
        MERGE (a:Vulnerability {vuln_id: row.a})
        MERGE (b:Vulnerability {vuln_id: row.b})
        MERGE (sp)-[:PROFILE_INCLUDES]->(a)
        MERGE (sp)-[:PROFILE_INCLUDES]->(b)
        MERGE (a)-[ra:CO_OCCURS_IN_PROFILE]->(sp)
        ON CREATE SET ra.max_confidence = row.conf,
                      ra.sources = CASE WHEN row.src = '' THEN [] ELSE [row.src] END
        ON MATCH SET ra.max_confidence = CASE
                                           WHEN row.conf > coalesce(ra.max_confidence, 0.0)
                                           THEN row.conf ELSE ra.max_confidence END,
                     ra.sources = CASE
                                   WHEN row.src = '' OR row.src IN coalesce(ra.sources, [])
                                   THEN coalesce(ra.sources, [])
                                   ELSE coalesce(ra.sources, []) + row.src
                                 END
        MERGE (b)-[rb:CO_OCCURS_IN_PROFILE]->(sp)
        ON CREATE SET rb.max_confidence = row.conf,
                      rb.sources = CASE WHEN row.src = '' THEN [] ELSE [row.src] END
        ON MATCH SET rb.max_confidence = CASE
                                           WHEN row.conf > coalesce(rb.max_confidence, 0.0)
                                           THEN row.conf ELSE rb.max_confidence END,
                     rb.sources = CASE
                                   WHEN row.src = '' OR row.src IN coalesce(rb.sources, [])
                                   THEN coalesce(rb.sources, [])
                                   ELSE coalesce(rb.sources, []) + row.src
                                 END
        """,
        rows=rows,
    )


def _flush_negative_rules(session, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Core rule nodes + profile links.
    session.run(
        """
        UNWIND $rows AS row
        WITH row
        WHERE row.profile <> '' AND row.condition <> ''
        WITH row, toLower(row.profile + '|' + row.condition) AS rule_key
        MERGE (sp:StackProfile {profile_id: row.profile})
        MERGE (nr:NegativeRule {rule_key: rule_key})
        ON CREATE SET nr.profile_id = row.profile,
                      nr.condition = row.condition,
                      nr.reason = row.reason,
                      nr.display = row.display
        ON MATCH SET nr.reason = coalesce(nr.reason, row.reason),
                     nr.display = coalesce(nr.display, row.display)
        MERGE (sp)-[:HAS_NEGATIVE_RULE]->(nr)
        """,
        rows=rows,
    )

    # Absent-if edges.
    session.run(
        """
        UNWIND $rows AS row
        WITH row
        WHERE row.profile <> '' AND row.condition <> ''
        WITH row, toLower(row.profile + '|' + row.condition) AS rule_key
        MATCH (nr:NegativeRule {rule_key: rule_key})
        UNWIND row.absent_cves AS absent_raw
        WITH nr, toUpper(trim(absent_raw)) AS absent_cve
        WHERE absent_cve STARTS WITH 'CVE-'
        MERGE (v_abs:Vulnerability {vuln_id: absent_cve})
        ON CREATE SET v_abs.id_type = 'CVE', v_abs.cve_id = absent_cve
        MERGE (nr)-[:RULE_ABSENT_IF]->(v_abs)
        """,
        rows=rows,
    )

    # Still-assess edges.
    session.run(
        """
        UNWIND $rows AS row
        WITH row
        WHERE row.profile <> '' AND row.condition <> ''
        WITH row, toLower(row.profile + '|' + row.condition) AS rule_key
        MATCH (nr:NegativeRule {rule_key: rule_key})
        UNWIND row.still_assess AS still_raw
        WITH nr, toUpper(trim(still_raw)) AS still_cve
        WHERE still_cve STARTS WITH 'CVE-'
        MERGE (v_still:Vulnerability {vuln_id: still_cve})
        ON CREATE SET v_still.id_type = 'CVE', v_still.cve_id = still_cve
        MERGE (nr)-[:RULE_STILL_ASSESS]->(v_still)
        """,
        rows=rows,
    )


def _iter_negative_rules(path: Path) -> Iterator[dict[str, Any]]:
    for obj in _iter_named_array_stream(path, "negative_rules"):
        parsed = _normalize_negative_rule_obj(obj) if isinstance(obj, dict) else None
        if parsed:
            yield parsed


def _extract_sw_list(master_row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    direct = master_row.get("affected_software", [])
    if isinstance(direct, list):
        out.extend(str(x).strip().lower() for x in direct if str(x).strip())

    if out:
        return list(dict.fromkeys(out))

    raw_ctx = master_row.get("raw_source_context", {})
    if not isinstance(raw_ctx, dict):
        return []
    nvd = raw_ctx.get("nvd_records_raw", [])
    if not isinstance(nvd, list):
        return []
    for rec in nvd:
        if not isinstance(rec, dict):
            continue
        sws = rec.get("affected_software", [])
        if isinstance(sws, list):
            out.extend(str(x).strip().lower() for x in sws if str(x).strip())
    return list(dict.fromkeys(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-file", default="data/master_vuln_context.jsonl")
    parser.add_argument("--cooccurrence-file", default="data/raw_cooccurrence_v2.json")
    parser.add_argument("--skip-negative-rules", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    _load_dotenv_if_present()

    neo4j_uri = _env("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = _env("NEO4J_USER", "neo4j")
    neo4j_password = _env("NEO4J_PASSWORD", required=True)
    batch_size = int(os.getenv("KG_BATCH_SIZE", "2000"))

    master_path = Path(args.master_file)
    cooc_path = Path(args.cooccurrence_file)
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    get_owasp_category = _get_owasp_mapper()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    print("Creating constraints...")
    with driver.session() as s:
        for q in [
            "CREATE CONSTRAINT vuln_id_unique IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.vuln_id IS UNIQUE",
            "CREATE CONSTRAINT cwe_id_unique IF NOT EXISTS FOR (n:CWE) REQUIRE n.cwe_id IS UNIQUE",
            "CREATE CONSTRAINT owasp_id_unique IF NOT EXISTS FOR (n:OWASPCategory) REQUIRE n.owasp_id IS UNIQUE",
            "CREATE CONSTRAINT software_key_unique IF NOT EXISTS FOR (n:Software) REQUIRE n.software_key IS UNIQUE",
            "CREATE CONSTRAINT cwe_cluster_id_unique IF NOT EXISTS FOR (n:CWECluster) REQUIRE n.cluster_id IS UNIQUE",
            "CREATE CONSTRAINT stack_profile_id_unique IF NOT EXISTS FOR (n:StackProfile) REQUIRE n.profile_id IS UNIQUE",
            "CREATE CONSTRAINT negative_rule_key_unique IF NOT EXISTS FOR (n:NegativeRule) REQUIRE n.rule_key IS UNIQUE",
        ]:
            s.run(q)
    print("  Done.")

    print(f"Loading master dataset from {master_path} ...")
    vuln_rows: list[dict[str, Any]] = []
    cwe_edges: list[dict[str, Any]] = []
    owasp_edges: list[dict[str, Any]] = []
    sw_edges: list[dict[str, Any]] = []
    corr_edges: list[dict[str, Any]] = []
    cooc_edges: list[dict[str, Any]] = []
    profile_edges: list[dict[str, Any]] = []

    processed = 0
    embedded_negative_rules: list[dict[str, Any]] = []
    embedded_negative_rule_keys: set[str] = set()
    with driver.session() as s:
        for row in _iter_jsonl(master_path):
            record_type = _normalize(row.get("record_type")).lower()
            if record_type == "master_metadata":
                rules = row.get("negative_rules", [])
                if isinstance(rules, list):
                    for rule in rules:
                        parsed = _normalize_negative_rule_obj(rule) if isinstance(rule, dict) else None
                        if not parsed:
                            continue
                        k = (parsed["profile"] + "|" + parsed["condition"]).lower()
                        if k in embedded_negative_rule_keys:
                            continue
                        embedded_negative_rule_keys.add(k)
                        embedded_negative_rules.append(parsed)
                continue

            vid = _normalize_cve(row.get("cve_id"))
            if not vid:
                continue

            vuln_rows.append(
                {
                    "vid": vid,
                    "name": row.get("vulnerability_name", ""),
                    "desc": str(row.get("description", ""))[:2000],
                    "cvss": _to_float(row.get("cvss_score"), None),
                    "epss": _to_float(row.get("epss_score"), None),
                    "kev": bool(row.get("confirmed_exploited", False)),
                    "risk": row.get("risk_level", ""),
                }
            )

            cwe = _normalize(row.get("cwe_id")).upper()
            if cwe.startswith("CWE-"):
                cwe_edges.append({"vid": vid, "cwe": cwe})
                owasp = _normalize(row.get("owasp_category"))
                if not owasp:
                    desc_text = _normalize(row.get("description", ""))
                    try:
                        owasp = _normalize(get_owasp_category(cwe, desc_text))
                    except Exception:
                        owasp = ""
                if owasp.startswith("A"):
                    owasp_edges.append({"vid": vid, "owasp": owasp})

            for sw in _extract_sw_list(row):
                sw_edges.append({"vid": vid, "sw": sw})

            for rel in row.get("correlations", []) if isinstance(row.get("correlations", []), list) else []:
                tgt = _normalize_cve(rel.get("cve_id"))
                if not tgt or tgt == vid:
                    continue
                a, b = (vid, tgt) if vid < tgt else (tgt, vid)
                signals = rel.get("signals", [])
                if not isinstance(signals, list):
                    signals = []
                corr_edges.append(
                    {
                        "a": a,
                        "b": b,
                        "score": _to_float(rel.get("correlation_score"), 0.0),
                        "signals": [str(x) for x in signals[:20]],
                    }
                )

            for rel in row.get("cooccurrences", []) if isinstance(row.get("cooccurrences", []), list) else []:
                tgt = _normalize_cve(rel.get("cve_id"))
                if not tgt or tgt == vid:
                    continue
                a, b = (vid, tgt) if vid < tgt else (tgt, vid)
                cooc_edges.append(
                    {
                        "a": a,
                        "b": b,
                        "conf": _to_float(rel.get("confidence"), 0.0),
                        "src": _normalize(rel.get("source")),
                        "reason": _normalize(rel.get("reason")),
                    }
                )
                profile = _normalize(rel.get("profile"))
                if profile:
                    profile_edges.append(
                        {
                            "profile": profile,
                            "a": a,
                            "b": b,
                            "conf": _to_float(rel.get("confidence"), 0.0),
                            "src": _normalize(rel.get("source")),
                        }
                    )

            if len(vuln_rows) >= batch_size:
                _flush_vuln_rows(s, vuln_rows)
                vuln_rows.clear()
            if len(cwe_edges) >= batch_size:
                _flush_cwe_edges(s, cwe_edges)
                cwe_edges.clear()
            if len(owasp_edges) >= batch_size:
                _flush_owasp_edges(s, owasp_edges)
                owasp_edges.clear()
            if len(sw_edges) >= batch_size:
                _flush_sw_edges(s, sw_edges)
                sw_edges.clear()
            if len(corr_edges) >= batch_size:
                _flush_corr_edges(s, corr_edges)
                corr_edges.clear()
            if len(cooc_edges) >= batch_size:
                _flush_cooc_edges(s, cooc_edges)
                cooc_edges.clear()
            if len(profile_edges) >= batch_size:
                _flush_profile_edges(s, profile_edges)
                profile_edges.clear()

            processed += 1
            if processed % 10000 == 0:
                print(f"  processed {processed:,} rows...")
            if args.max_rows > 0 and processed >= args.max_rows:
                break

        _flush_vuln_rows(s, vuln_rows)
        _flush_cwe_edges(s, cwe_edges)
        _flush_owasp_edges(s, owasp_edges)
        _flush_sw_edges(s, sw_edges)
        _flush_corr_edges(s, corr_edges)
        _flush_cooc_edges(s, cooc_edges)
        _flush_profile_edges(s, profile_edges)

    # Optional pass: ingest negative rules directly from raw_cooccurrence_v2.
    negative_rules_ingested = 0
    if not args.skip_negative_rules:
        if embedded_negative_rules:
            print(f"Loading embedded negative rules from master dataset ...")
            with driver.session() as s:
                batch: list[dict[str, Any]] = []
                for rule in embedded_negative_rules:
                    batch.append(rule)
                    if len(batch) >= batch_size:
                        _flush_negative_rules(s, batch)
                        negative_rules_ingested += len(batch)
                        batch.clear()
                if batch:
                    _flush_negative_rules(s, batch)
                    negative_rules_ingested += len(batch)
            print(f"  Negative rules ingested: {negative_rules_ingested:,}")
        elif cooc_path.exists():
            print(f"Loading negative rules from fallback file {cooc_path} ...")
            with driver.session() as s:
                batch = []
                for rule in _iter_negative_rules(cooc_path):
                    batch.append(rule)
                    if len(batch) >= batch_size:
                        _flush_negative_rules(s, batch)
                        negative_rules_ingested += len(batch)
                        batch.clear()
                if batch:
                    _flush_negative_rules(s, batch)
                    negative_rules_ingested += len(batch)
            print(f"  Negative rules ingested: {negative_rules_ingested:,}")
        else:
            print(f"Skipping negative rule ingest: file not found ({cooc_path})")

    print("KG load from master dataset complete. Verifying...")
    with driver.session() as s:
        v = s.run("MATCH (v:Vulnerability) RETURN count(v) AS c").single()["c"]
        cw = s.run("MATCH (w:CWE) RETURN count(w) AS c").single()["c"]
        ow = s.run("MATCH (o:OWASPCategory) RETURN count(o) AS c").single()["c"]
        sw = s.run("MATCH (p:Software) RETURN count(p) AS c").single()["c"]
        sp = s.run("MATCH (sp:StackProfile) RETURN count(sp) AS c").single()["c"]
        nr = s.run("MATCH (nr:NegativeRule) RETURN count(nr) AS c").single()["c"]
        co = s.run("MATCH ()-[r:CORRELATED_WITH]->() RETURN count(r) AS c").single()["c"]
        oc = s.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS c").single()["c"]
        hp = s.run("MATCH ()-[r:HAS_NEGATIVE_RULE]->() RETURN count(r) AS c").single()["c"]
        ra = s.run("MATCH ()-[r:RULE_ABSENT_IF]->() RETURN count(r) AS c").single()["c"]
        rs = s.run("MATCH ()-[r:RULE_STILL_ASSESS]->() RETURN count(r) AS c").single()["c"]
        print(f"  Processed rows      : {processed:,}")
        print(f"  Vulnerability nodes : {v:,}")
        print(f"  CWE nodes           : {cw:,}")
        print(f"  OWASP nodes         : {ow:,}")
        print(f"  Software nodes      : {sw:,}")
        print(f"  StackProfile nodes  : {sp:,}")
        print(f"  NegativeRule nodes  : {nr:,}")
        print(f"  CORRELATED_WITH     : {co:,}")
        print(f"  CO_OCCURS_WITH      : {oc:,}")
        print(f"  HAS_NEGATIVE_RULE   : {hp:,}")
        print(f"  RULE_ABSENT_IF      : {ra:,}")
        print(f"  RULE_STILL_ASSESS   : {rs:,}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
