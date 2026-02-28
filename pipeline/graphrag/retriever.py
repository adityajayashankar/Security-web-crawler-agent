from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from pipeline.graphrag.schema import (
    Citation,
    ConfidenceSummary,
    EvidenceItem,
    GraphEntity,
    GraphRAGAgentResponse,
    dump_model,
)


_CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _normalize_graph_score(value: float) -> float:
    """
    Normalize arbitrary positive edge scores to [0,1] while preserving rank.
    Avoids hard saturation when raw graph scores are >1.
    """
    raw = float(value or 0.0)
    if raw <= 0:
        return 0.0
    if raw <= 1.0:
        return raw
    return raw / (1.0 + raw)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _normalize_entity(query: str, entity: dict[str, Any] | None) -> GraphEntity:
    if entity:
        ent_type = str(entity.get("type", "unknown")).lower().strip()
        ent_id = str(entity.get("id", "")).upper().strip()
        if ent_type in {"cve", "cwe"} and ent_id:
            return GraphEntity(type=ent_type, id=ent_id)
    cve = _CVE_RE.search(query or "")
    if cve:
        return GraphEntity(type="cve", id=cve.group(0).upper())
    cwe = _CWE_RE.search(query or "")
    if cwe:
        return GraphEntity(type="cwe", id=cwe.group(0).upper())
    return GraphEntity(type="unknown", id="")


def _get_neo4j_driver():
    try:
        from neo4j import GraphDatabase
    except Exception:
        return None
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if not password:
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as s:
            s.run("RETURN 1")
        return driver
    except Exception:
        return None


def _query_cve_neighbors(session, cve_id: str, top_k: int) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (v:Vulnerability {vuln_id: $cve_id})
              -[r:CORRELATED_WITH|CO_OCCURS_WITH]-(related:Vulnerability)
        RETURN related.vuln_id AS cve_id,
               coalesce(r.max_score, r.max_confidence, 0.0) AS score,
               type(r) AS rel_type,
               coalesce(r.signals, []) AS signals,
               coalesce(r.reasons, []) AS reasons
        ORDER BY score DESC
        LIMIT $top_k
        """,
        cve_id=cve_id,
        top_k=top_k,
    ).data()


def _query_cve_neighbors_by_type(session, cve_id: str, rel_type: str, top_k: int) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (v:Vulnerability {vuln_id: $cve_id})-[r]-(related:Vulnerability)
        WHERE type(r) = $rel_type
        RETURN related.vuln_id AS cve_id,
               coalesce(r.max_score, r.max_confidence, 0.0) AS score,
               type(r) AS rel_type,
               coalesce(r.signals, []) AS signals,
               coalesce(r.reasons, []) AS reasons
        ORDER BY score DESC
        LIMIT $top_k
        """,
        cve_id=cve_id,
        rel_type=rel_type,
        top_k=top_k,
    ).data()


def _mix_cve_neighbors(
    session,
    cve_id: str,
    top_k: int,
    min_cooccur: int = 0,
) -> list[dict[str, Any]]:
    if min_cooccur <= 0:
        return _query_cve_neighbors(session, cve_id, top_k)

    corr_rows = _query_cve_neighbors_by_type(session, cve_id, "CORRELATED_WITH", top_k=max(top_k, 1))
    cooc_rows = _query_cve_neighbors_by_type(
        session,
        cve_id,
        "CO_OCCURS_WITH",
        top_k=max(top_k, min_cooccur),
    )
    target_cooc = min(max(min_cooccur, 0), top_k)
    selected = cooc_rows[:target_cooc] + corr_rows[: max(0, top_k - min(len(cooc_rows), target_cooc))]

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in selected:
        key = str(row.get("cve_id", "")).upper().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if len(deduped) < top_k:
        remainder = sorted(corr_rows + cooc_rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        for row in remainder:
            key = str(row.get("cve_id", "")).upper().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= top_k:
                break
    return deduped[:top_k]


def _compute_second_hop_from_seed_map(
    origin_marker: str,
    hop1_rows: list[dict[str, Any]],
    seed_neighbors_map: dict[str, list[dict[str, Any]]],
    top_k: int,
    decay: float = 0.75,
) -> list[dict[str, Any]]:
    """
    Build second-order candidates from first-hop evidence.
    Example: A -> B (hop1), B -> G (neighbor) => candidate G with path A->B->G.
    """
    first_hop_ids = {
        str(r.get("cve_id", "")).upper().strip()
        for r in hop1_rows
        if str(r.get("cve_id", "")).strip()
    }
    seed_score_map = {
        str(r.get("cve_id", "")).upper().strip(): _normalize_graph_score(float(r.get("score", 0.0)))
        for r in hop1_rows
        if str(r.get("cve_id", "")).strip()
    }

    second_hop: dict[str, dict[str, Any]] = {}
    for seed, neighbors in seed_neighbors_map.items():
        seed_norm = str(seed).upper().strip()
        seed_score = seed_score_map.get(seed_norm, 0.0)
        if seed_score <= 0:
            continue

        for row in neighbors:
            target = str(row.get("cve_id", "")).upper().strip()
            if not target:
                continue
            if target == origin_marker or target in first_hop_ids:
                continue

            edge_score = _normalize_graph_score(float(row.get("score", 0.0)))
            combined = _clamp01(seed_score * edge_score * decay)
            if combined <= 0:
                continue

            candidate = {
                "cve_id": target,
                "score": combined,
                "rel_type": f"HOP2_{row.get('rel_type', 'UNKNOWN')}",
                "signals": list(dict.fromkeys(list(row.get("signals", []) or []) + [f"via:{seed_norm}", f"origin:{origin_marker}"]))[:6],
                "reasons": list(dict.fromkeys(list(row.get("reasons", []) or []) + [f"second_order_risk_from:{seed_norm}"]))[:4],
                "hop": 2,
                "via": seed_norm,
                "source_type": "graph_hop2",
                "path": [origin_marker, seed_norm, target],
            }
            prev = second_hop.get(target)
            if prev is None or combined > float(prev.get("score", 0.0)):
                second_hop[target] = candidate

    return sorted(second_hop.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k]


def _graph_candidates(entity: GraphEntity, top_k: int, max_hops: int = 2) -> list[dict[str, Any]]:
    driver = _get_neo4j_driver()
    if not driver:
        return []

    rows: list[dict[str, Any]] = []
    try:
        with driver.session() as session:
            if entity.type == "cve" and entity.id:
                min_cooccur = max(0, min(int(os.getenv("GRAPHRAG_MIN_COOCCUR_PER_CVE", "2")), top_k))
                hop1 = _mix_cve_neighbors(session, entity.id, top_k=top_k, min_cooccur=min_cooccur)
                for row in hop1:
                    row["hop"] = 1
                    row["source_type"] = "graph"
                rows.extend(hop1)

                if max_hops >= 2 and hop1:
                    branch_factor = min(max(3, top_k // 2), 8)
                    seeds = [str(r.get("cve_id", "")).upper().strip() for r in hop1[:branch_factor]]
                    seed_neighbors_map: dict[str, list[dict[str, Any]]] = {}
                    for seed in seeds:
                        if not seed:
                            continue
                        seed_neighbors_map[seed] = _query_cve_neighbors(
                            session,
                            seed,
                            top_k=max(top_k, branch_factor * 3),
                        )
                    rows.extend(
                        _compute_second_hop_from_seed_map(
                            origin_marker=entity.id,
                            hop1_rows=hop1,
                            seed_neighbors_map=seed_neighbors_map,
                            top_k=top_k,
                        )
                    )
            elif entity.type == "cwe" and entity.id:
                hop1 = session.run(
                    """
                    MATCH (w:CWE {cwe_id: $cwe_id})<-[:HAS_CWE]-(v:Vulnerability)
                    OPTIONAL MATCH (v)-[r:CORRELATED_WITH|CO_OCCURS_WITH]-(peer:Vulnerability)
                    RETURN coalesce(peer.vuln_id, v.vuln_id) AS cve_id,
                           coalesce(r.max_score, r.max_confidence, v.epss_score, 0.2) AS score,
                           coalesce(type(r), 'HAS_CWE') AS rel_type,
                           CASE WHEN r IS NULL THEN ['shared_cwe:' + $cwe_id] ELSE coalesce(r.signals, []) END AS signals,
                           coalesce(r.reasons, []) AS reasons
                    ORDER BY score DESC
                    LIMIT $top_k
                    """,
                    cwe_id=entity.id,
                    top_k=top_k,
                ).data()
                for row in hop1:
                    row["hop"] = 1
                    row["source_type"] = "graph"
                rows.extend(hop1)

                if max_hops >= 2 and hop1:
                    branch_factor = min(max(3, top_k // 2), 8)
                    seeds = [str(r.get("cve_id", "")).upper().strip() for r in hop1[:branch_factor]]
                    seed_neighbors_map: dict[str, list[dict[str, Any]]] = {}
                    for seed in seeds:
                        if not seed:
                            continue
                        seed_neighbors_map[seed] = _query_cve_neighbors(
                            session,
                            seed,
                            top_k=max(top_k, branch_factor * 3),
                        )
                    rows.extend(
                        _compute_second_hop_from_seed_map(
                            origin_marker=entity.id,
                            hop1_rows=hop1,
                            seed_neighbors_map=seed_neighbors_map,
                            top_k=top_k,
                        )
                    )
    except Exception:
        rows = []
    finally:
        driver.close()
    return rows


_EMBED_MODEL = None
_EMBED_BACKEND = ""


def _embed_query(text: str) -> list[float]:
    global _EMBED_MODEL, _EMBED_BACKEND
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    if _EMBED_MODEL is None:
        try:
            from fastembed import TextEmbedding

            _EMBED_MODEL = TextEmbedding(model_name=model_name)
            _EMBED_BACKEND = "fastembed"
        except Exception:
            from sentence_transformers import SentenceTransformer

            _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            _EMBED_BACKEND = "sentence-transformers"

    if _EMBED_BACKEND == "fastembed":
        vector = list(_EMBED_MODEL.embed([text]))[0]
        return [float(x) for x in vector]
    vector = _EMBED_MODEL.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vector.tolist()]


def _get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return None

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    qdrant_path = os.getenv("QDRANT_PATH", str(Path("data") / "qdrant"))
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=qdrant_path)


def _vector_candidates(query: str, entity: GraphEntity, top_k: int) -> list[dict[str, Any]]:
    client = _get_qdrant_client()
    if not client:
        return []
    try:
        query_vector = _embed_query(query)
    except Exception:
        return []

    collection = os.getenv("QDRANT_COLLECTION", "vuln_kg_evidence_v1")
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        return []

    out = []
    for h in hits:
        payload = h.payload or {}
        cve_id = str(payload.get("cve_id") or payload.get("target_cve") or "").upper().strip()
        if not cve_id:
            continue
        out.append(
            {
                "cve_id": cve_id,
                "score": _clamp01(float(h.score)),
                "rel_type": str(payload.get("rel_type", "VECTOR_SIMILARITY")),
                "signals": payload.get("signals", []),
                "reasons": payload.get("reasons", []),
                "source_type": str(payload.get("source_type", "vector")),
                "snippet": str(payload.get("text", ""))[:280],
            }
        )
    return out


def _fuse_candidates(
    graph_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[EvidenceItem], list[EvidenceItem], list[Citation]]:
    merged: dict[str, dict[str, Any]] = {}
    hop1_weight = _clamp01(_env_float("GRAPHRAG_GRAPH_WEIGHT_HOP1", 0.88))
    hop2_weight = _clamp01(_env_float("GRAPHRAG_GRAPH_WEIGHT_HOP2", 0.55))

    for row in graph_rows:
        cve_id = str(row.get("cve_id", "")).upper().strip()
        if not cve_id:
            continue
        g_score = _normalize_graph_score(float(row.get("score", 0.0)))
        hop = int(row.get("hop", 1))
        is_hop2 = hop >= 2
        graph_weight = hop2_weight if is_hop2 else hop1_weight
        row_payload = {
            "cve_id": cve_id,
            "likelihood": round(graph_weight * g_score, 3),
            "graph_score": g_score,
            "vector_score": 0.0,
            "evidence_tier": "inferred" if is_hop2 else "direct",
            "rel_type": row.get("rel_type", ""),
            "signals": list(row.get("signals", []) or [])[:5],
            "reasons": list(row.get("reasons", []) or [])[:3],
            "inferred_from": [f"via:{row.get('via')}"] if is_hop2 and row.get("via") else ([] if not is_hop2 else ["graph_hop2"]),
            "source_type": row.get("source_type", "graph"),
            "snippet": (
                " -> ".join(row.get("path", []))
                if row.get("path")
                else (", ".join(list(row.get("signals", [])[:2])) or "graph correlation evidence")
            ),
        }
        if cve_id in merged:
            # Keep the stronger graph path and preserve direct tier if any direct evidence exists.
            prev = merged[cve_id]
            if row_payload["graph_score"] > prev.get("graph_score", 0.0):
                merged[cve_id]["graph_score"] = row_payload["graph_score"]
                merged[cve_id]["likelihood"] = max(merged[cve_id]["likelihood"], row_payload["likelihood"])
                merged[cve_id]["rel_type"] = row_payload["rel_type"]
                merged[cve_id]["signals"] = row_payload["signals"]
                merged[cve_id]["reasons"] = row_payload["reasons"]
                merged[cve_id]["snippet"] = row_payload["snippet"]
            if prev.get("evidence_tier") == "direct" or row_payload["evidence_tier"] == "direct":
                merged[cve_id]["evidence_tier"] = "direct"
                merged[cve_id]["inferred_from"] = []
            else:
                merged[cve_id]["inferred_from"] = row_payload["inferred_from"]
        else:
            merged[cve_id] = row_payload

    for row in vector_rows:
        cve_id = str(row.get("cve_id", "")).upper().strip()
        if not cve_id:
            continue
        v_score = _clamp01(float(row.get("score", 0.0)))
        if cve_id in merged:
            merged[cve_id]["vector_score"] = max(merged[cve_id]["vector_score"], v_score)
            merged[cve_id]["likelihood"] = round(
                _clamp01(merged[cve_id]["likelihood"] + 0.35 * v_score), 3
            )
            if row.get("snippet"):
                merged[cve_id]["snippet"] = row["snippet"]
        else:
            merged[cve_id] = {
                "cve_id": cve_id,
                "likelihood": round(0.35 * v_score, 3),
                "graph_score": 0.0,
                "vector_score": v_score,
                "evidence_tier": "inferred",
                "rel_type": row.get("rel_type", "VECTOR_SIMILARITY"),
                "signals": list(row.get("signals", []) or [])[:5],
                "reasons": list(row.get("reasons", []) or [])[:3],
                "inferred_from": ["vector_similarity"],
                "source_type": row.get("source_type", "vector"),
                "snippet": row.get("snippet", "semantic similarity evidence"),
            }

    ordered = sorted(merged.values(), key=lambda x: x["likelihood"], reverse=True)[:top_k]
    direct: list[EvidenceItem] = []
    inferred: list[EvidenceItem] = []
    citations: list[Citation] = []

    for row in ordered:
        evidence = EvidenceItem(
            cve_id=row["cve_id"],
            likelihood=row["likelihood"],
            evidence_tier=row["evidence_tier"],
            rel_type=row["rel_type"],
            signals=row["signals"],
            reasons=row["reasons"],
            inferred_from=row["inferred_from"],
        )
        citation = Citation(
            citation_id=f"cit-{_hash(row['cve_id'] + row['rel_type'] + row['source_type'])}",
            source_type=row["source_type"],
            entity_id=row["cve_id"],
            snippet=row["snippet"],
            metadata={
                "graph_score": row["graph_score"],
                "vector_score": row["vector_score"],
                "tier": row["evidence_tier"],
            },
        )
        citations.append(citation)
        if row["evidence_tier"] == "direct":
            direct.append(evidence)
        else:
            inferred.append(evidence)

    return direct, inferred, citations


def retrieve_hybrid(
    query: str,
    entity: dict[str, Any] | None = None,
    top_k: int = 12,
    max_hops: int = 2,
    use_vector: bool = True,
) -> dict[str, Any]:
    top_k = max(1, min(int(top_k), 25))
    max_hops = max(1, min(int(max_hops), 3))
    ent = _normalize_entity(query, entity)

    graph_rows = _graph_candidates(ent, top_k=top_k, max_hops=max_hops)
    vector_rows = _vector_candidates(query, ent, top_k=top_k) if use_vector else []
    direct, inferred, citations = _fuse_candidates(graph_rows, vector_rows, top_k)

    top_scores = [e.likelihood for e in (direct + inferred)[:5]]
    overall = round(sum(top_scores) / max(len(top_scores), 1), 3) if top_scores else 0.0
    if use_vector:
        rationale = (
            f"Hybrid retrieval fused {len(graph_rows)} graph candidates and {len(vector_rows)} vector candidates (max_hops={max_hops})."
            if (graph_rows or vector_rows)
            else "No graph/vector evidence found."
        )
    else:
        rationale = (
            f"Graph-only retrieval produced {len(graph_rows)} graph candidates (max_hops={max_hops})."
            if graph_rows
            else "No graph evidence found."
        )

    response = GraphRAGAgentResponse(
        status="ok",
        query=query,
        entity=ent,
        direct_evidence=direct,
        inferred_candidates=inferred,
        citations=citations,
        confidence_summary=ConfidenceSummary(overall=overall, rationale=rationale),
        recommended_actions=[
            "Validate direct evidence against asset inventory and patch state.",
            "Actively test inferred candidates before remediation prioritization.",
        ],
    )
    return dump_model(response)
