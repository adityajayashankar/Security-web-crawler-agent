from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

CHUNK_SIZE = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "120"))
EMBED_BATCH_SIZE = int(os.getenv("GRAPHRAG_EMBED_BATCH_SIZE", "512"))
UPSERT_BATCH_SIZE = int(os.getenv("GRAPHRAG_UPSERT_BATCH_SIZE", "1000"))
MAX_DATASET_ROWS = int(os.getenv("GRAPHRAG_MAX_DATASET_ROWS", "0"))
MAX_CORR_ROWS = int(os.getenv("GRAPHRAG_MAX_CORR_ROWS", "0"))
MAX_COOC_ROWS = int(os.getenv("GRAPHRAG_MAX_COOC_ROWS", "0"))
MAX_TOTAL_CHUNKS = int(os.getenv("GRAPHRAG_MAX_TOTAL_CHUNKS", "0"))
LOG_EVERY = int(os.getenv("GRAPHRAG_LOG_EVERY", "20000"))
JSONL_LOG_EVERY = int(os.getenv("GRAPHRAG_JSONL_LOG_EVERY", "5000"))
ENABLE_CORR = os.getenv("GRAPHRAG_ENABLE_CORR", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_COOC = os.getenv("GRAPHRAG_ENABLE_COOC", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_DATASET = os.getenv("GRAPHRAG_ENABLE_DATASET", "1").strip().lower() not in {"0", "false", "no"}


def _log(msg: str) -> None:
    print(f"[graphrag-indexer] {msg}", flush=True)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Character-window chunking with overlap and whitespace boundary preference.
    """
    t = " ".join(str(text or "").split()).strip()
    if not t:
        return []
    if chunk_size <= 0:
        return [t]
    overlap = max(0, min(overlap, chunk_size // 2))
    if len(t) <= chunk_size:
        return [t]

    out: list[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            ws = t.rfind(" ", start + int(chunk_size * 0.65), end)
            if ws > start:
                end = ws
        chunk = t[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return out


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=model_name)
        return [[float(x) for x in vec] for vec in model.embed(texts)]
    except Exception:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row.tolist()] for row in vectors]


def _qdrant_client():
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    if url:
        return QdrantClient(url=url, api_key=api_key)
    path = os.getenv("QDRANT_PATH", str(Path("data") / "qdrant"))
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def _ensure_qdrant_collection_preflight() -> None:
    """
    For remote Qdrant, verify collection exists before expensive chunk building.
    If missing, create it using an embedding-dimension probe.
    """
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if not qdrant_url:
        return

    from qdrant_client.models import Distance, VectorParams

    collection = os.getenv("QDRANT_COLLECTION", "vuln_kg_evidence_v1")
    client = _qdrant_client()
    _log(f"Qdrant preflight: checking remote collection '{collection}' at {qdrant_url}")

    try:
        client.get_collection(collection_name=collection)
        _log("Qdrant preflight: collection exists and is reachable")
        return
    except Exception as err:
        _log(f"Qdrant preflight: collection missing/unavailable ({err}); creating")

    probe_vecs = _embed_texts(["qdrant collection preflight probe"])
    if not probe_vecs or not probe_vecs[0]:
        raise RuntimeError("Qdrant preflight failed: could not compute probe embedding.")
    vector_dim = len(probe_vecs[0])

    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        _log(f"Qdrant preflight: created collection '{collection}' (dim={vector_dim})")
    except Exception:
        # Handles race where another process created it after our first check.
        client.get_collection(collection_name=collection)
        _log("Qdrant preflight: collection already created by another process")


def _safe_json(path: Path) -> Any:
    started = time.perf_counter()
    if not path.exists():
        _log(f"Skipping missing file: {path}")
        return [] if path.suffix != ".jsonl" else []
    _log(f"Loading {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
    if path.suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        _log(f"Loaded {len(rows)} rows from {path} in {time.perf_counter() - started:.1f}s")
        return rows
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        extra = f"{len(data)} rows"
    elif isinstance(data, dict):
        extra = f"{len(data)} top-level keys"
    else:
        extra = f"type={type(data).__name__}"
    _log(f"Loaded {extra} from {path} in {time.perf_counter() - started:.1f}s")
    return data


def _read_jsonl_limited(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    started = time.perf_counter()
    if not path.exists():
        _log(f"Skipping missing file: {path}")
        return []
    _log(
        f"Streaming JSONL {path} ({path.stat().st_size / (1024 * 1024):.1f} MB)"
        + (f", cap={max_rows}" if max_rows > 0 else "")
    )
    rows: list[dict[str, Any]] = []
    lines_seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            lines_seen += 1
            if max_rows > 0 and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
            if JSONL_LOG_EVERY > 0 and lines_seen % JSONL_LOG_EVERY == 0:
                _log(
                    f"JSONL parse progress: lines={lines_seen}, rows={len(rows)}, elapsed={time.perf_counter() - started:.1f}s"
                )
    _log(f"Loaded {len(rows)} JSONL rows from {path} in {time.perf_counter() - started:.1f}s")
    return rows


def _kg_edges() -> list[dict[str, Any]]:
    try:
        from neo4j import GraphDatabase
    except Exception:
        return []

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if not password:
        return []

    out = []
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as s:
            rows = s.run(
                """
                MATCH (a:Vulnerability)-[r:CORRELATED_WITH|CO_OCCURS_WITH]->(b:Vulnerability)
                RETURN a.vuln_id AS a_id,
                       b.vuln_id AS b_id,
                       type(r) AS rel_type,
                       coalesce(r.max_score, r.max_confidence, 0.0) AS score
                LIMIT 20000
                """
            ).data()
            out.extend(rows)
        driver.close()
    except Exception:
        return []
    return out


def build_evidence_chunks(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    started = time.perf_counter()
    data_dir = Path(data_dir)
    _log(f"Building evidence chunks from {data_dir.resolve()}")
    dataset = (
        _read_jsonl_limited(data_dir / "vuln_dataset.jsonl", max_rows=MAX_DATASET_ROWS)
        if ENABLE_DATASET
        else []
    )
    corrs = _safe_json(data_dir / "raw_correlations.json") if ENABLE_CORR else []
    coocs = _safe_json(data_dir / "raw_cooccurrence_v2.json") if ENABLE_COOC else []
    kg_edges = _kg_edges()
    if kg_edges:
        _log(f"Loaded {len(kg_edges)} Neo4j graph edges")
    else:
        _log("No Neo4j graph edges loaded")

    chunks: list[dict[str, Any]] = []

    dataset_rows = dataset if isinstance(dataset, list) else []
    _log(f"Processing dataset rows: {len(dataset_rows)}")
    for row_idx, row in enumerate(dataset_rows, start=1):
        cve_id = str(row.get("cve_id") or row.get("ghsa_id") or "").upper().strip()
        if not cve_id:
            continue
        base_text = (
            f"{cve_id} {row.get('vulnerability_name', '')}. "
            f"CWE: {row.get('cwe_id', '')}. "
            f"OWASP: {row.get('owasp_category', '')}. "
            f"Description: {str(row.get('description', ''))}"
        ).strip()
        text_parts = _split_text(base_text)
        if not text_parts:
            continue
        for idx, text in enumerate(text_parts):
            chunks.append(
                {
                    "id": f"dataset-{_hash(cve_id + str(idx) + text[:120])}",
                    "text": text,
                    "cve_id": cve_id,
                    "source_type": "dataset",
                    "rel_type": "HAS_CONTEXT",
                    "signals": [row.get("cwe_id", ""), row.get("owasp_category", "")],
                    "reasons": [],
                    "chunk_index": idx,
                }
            )
        if LOG_EVERY > 0 and row_idx % LOG_EVERY == 0:
            _log(f"Dataset progress: {row_idx}/{len(dataset_rows)} rows, chunks={len(chunks)}")

    corr_rows = corrs if isinstance(corrs, list) else []
    if MAX_CORR_ROWS > 0:
        corr_rows = corr_rows[:MAX_CORR_ROWS]
        _log(f"Capping correlation rows to {len(corr_rows)} via GRAPHRAG_MAX_CORR_ROWS")
    _log(f"Processing correlation rows: {len(corr_rows)}")
    for row_idx, row in enumerate(corr_rows, start=1):
        src = str(row.get("cve_id", "")).upper().strip()
        for rel in row.get("related_vulnerabilities", [])[:20]:
            tgt = str(rel.get("cve_id", "")).upper().strip()
            if not src or not tgt:
                continue
            base_text = (
                f"{src} correlates with {tgt}. "
                f"Signals: {', '.join(rel.get('signals', [])[:5])}. "
                f"Score: {rel.get('correlation_score', 0.0)}."
            )
            for idx, text in enumerate(_split_text(base_text)):
                chunks.append(
                    {
                        "id": f"corr-{_hash(src + tgt + str(idx) + text)}",
                        "text": text,
                        "cve_id": tgt,
                        "target_cve": tgt,
                        "source_type": "raw_correlations",
                        "rel_type": "CORRELATED_WITH",
                        "signals": rel.get("signals", [])[:5],
                        "reasons": [],
                        "chunk_index": idx,
                    }
                )
        if LOG_EVERY > 0 and row_idx % LOG_EVERY == 0:
            _log(f"Correlations progress: {row_idx}/{len(corr_rows)} rows, chunks={len(chunks)}")

    pairs = []
    if isinstance(coocs, dict):
        pairs = coocs.get("cooccurrence_pairs", [])
    elif isinstance(coocs, list):
        pairs = coocs

    if MAX_COOC_ROWS > 0:
        pairs = pairs[:MAX_COOC_ROWS]
    _log(f"Processing cooccurrence pairs: {len(pairs)}")
    for pair_idx, pair in enumerate(pairs, start=1):
        a = str(pair.get("cve_a", "")).upper().strip()
        b = str(pair.get("cve_b", "")).upper().strip()
        if not a or not b:
            continue
        base_text = (
            f"{a} co-occurs with {b}. "
            f"Confidence: {pair.get('confidence', 0.0)}. "
            f"Source: {pair.get('source', '')}. "
            f"Reason: {pair.get('reason', '')}"
        )
        for idx, text in enumerate(_split_text(base_text)):
            chunks.append(
                {
                    "id": f"cooc-{_hash(a + b + str(idx) + text)}",
                    "text": text,
                    "cve_id": b,
                    "target_cve": b,
                    "source_type": "raw_cooccurrence_v2",
                    "rel_type": "CO_OCCURS_WITH",
                    "signals": [pair.get("source", "")],
                    "reasons": [pair.get("reason", "")],
                    "chunk_index": idx,
                }
            )
        if LOG_EVERY > 0 and pair_idx % LOG_EVERY == 0:
            _log(f"Cooccurrence progress: {pair_idx}/{len(pairs)} pairs, chunks={len(chunks)}")

    _log(f"Processing Neo4j rows: {len(kg_edges)}")
    for row in kg_edges:
        a = str(row.get("a_id", "")).upper().strip()
        b = str(row.get("b_id", "")).upper().strip()
        if not a or not b:
            continue
        base_text = (
            f"{a} has graph edge {row.get('rel_type', '')} with {b}. "
            f"Score: {row.get('score', 0.0)}."
        )
        for idx, text in enumerate(_split_text(base_text)):
            chunks.append(
                {
                    "id": f"kg-{_hash(a + b + str(idx) + text)}",
                    "text": text,
                    "cve_id": b,
                    "target_cve": b,
                    "source_type": "neo4j",
                    "rel_type": row.get("rel_type", "GRAPH_EDGE"),
                    "signals": [row.get("rel_type", "")],
                    "reasons": [],
                    "chunk_index": idx,
                }
            )

    seen = set()
    deduped = []
    for ch in chunks:
        key = ch["id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ch)
    _log(
        f"Chunk build complete: raw={len(chunks)} deduped={len(deduped)} in {time.perf_counter() - started:.1f}s"
    )
    if MAX_TOTAL_CHUNKS > 0 and len(deduped) > MAX_TOTAL_CHUNKS:
        _log(f"Capping total chunks to {MAX_TOTAL_CHUNKS} via GRAPHRAG_MAX_TOTAL_CHUNKS")
        deduped = deduped[:MAX_TOTAL_CHUNKS]
    return deduped


def upsert_qdrant(chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        _log("No chunks to upsert")
        return 0

    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "vuln_kg_evidence_v1")

    first_batch = chunks[: min(len(chunks), EMBED_BATCH_SIZE)]
    _log(
        f"Embedding first batch to discover vector size (batch={len(first_batch)}, model={os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')})"
    )
    vectors_first = _embed_texts([c["text"] for c in first_batch])
    if not vectors_first:
        return 0
    vector_dim = len(vectors_first[0])
    _log(f"Embedding dimension resolved: {vector_dim}")

    try:
        client.get_collection(collection_name=collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    total = 0
    started = time.perf_counter()
    _log(f"Upserting {len(chunks)} chunks into collection '{collection}'")
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        vectors = _embed_texts([c["text"] for c in batch])
        points = []
        for chunk, vector in zip(batch, vectors):
            payload = {k: v for k, v in chunk.items() if k not in {"id"}}
            payload["chunk_id"] = chunk["id"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        for j in range(0, len(points), UPSERT_BATCH_SIZE):
            pbatch = points[j : j + UPSERT_BATCH_SIZE]
            client.upsert(collection_name=collection, points=pbatch, wait=True)
            total += len(pbatch)
        done = min(i + len(batch), len(chunks))
        elapsed = max(1e-6, time.perf_counter() - started)
        rate = done / elapsed
        remaining = len(chunks) - done
        eta_sec = int(remaining / rate) if rate > 0 else 0
        _log(
            f"Embedded/upserted {done}/{len(chunks)} chunks "
            f"(rate={rate:.1f}/s, eta={eta_sec // 3600:02d}:{(eta_sec % 3600) // 60:02d}:{eta_sec % 60:02d})"
        )
    _log(f"Qdrant upsert complete: {total} points")
    return total


def build_and_index(data_dir: str | Path = "data") -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_qdrant_collection_preflight()
    chunks = build_evidence_chunks(data_dir=data_dir)
    count = upsert_qdrant(chunks)
    result = {"indexed_points": count, "collection": os.getenv("QDRANT_COLLECTION", "vuln_kg_evidence_v1")}
    _log(f"Indexing complete in {time.perf_counter() - started:.1f}s")
    return result


if __name__ == "__main__":
    result = build_and_index()
    print(json.dumps(result, indent=2))
