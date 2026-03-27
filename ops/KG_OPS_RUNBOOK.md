# KG Ops Runbook

## Current State (as of March 9, 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| Neo4j KG | ✅ Loaded | 407,713 nodes / 6,928,186 relationships |
| Qdrant vector index | ⚠️ Partial | 20,038 / 1,000,000–2,000,000 target vectors ingested |
| Benchmark (`run_graphrag_benchmark.py`) | ❌ Blocked | All 120 probes error with `No module named 'pipeline'` — run from project root with `PYTHONPATH=.` |
| Ground-truth benchmark | ❌ Blocked | Same import error; 20/20 probes fail |
| training_pairs.jsonl | ✅ Current | 3,509,451 rows / 3.17 GB |
| vuln_dataset.jsonl | ✅ Current | 325,941 rows / 1.34 GB |
| Fine-tuned model | 🔲 Pending | Training not yet run |
| Releases manifest | 🔲 Empty | No promoted releases yet |

**Benchmark fix:** Run the benchmark from the project root so that `pipeline` is importable:
```powershell
cd C:\Users\adity\dataset-deplai
$env:PYTHONPATH="."
python eval/run_graphrag_benchmark.py --benchmark-file eval/heldout_cve_benchmark.jsonl --ground-truth eval/ground_truth_benchmark.jsonl --max-probes 120 --top-k 20 --max-hops 2 --strict --output-json eval/results/graphrag_eval.json --output-csv eval/results/graphrag_eval.csv
```

---

## Release Model
- Cadence: on-demand manual.
- Release id format: `kg_YYYYMMDD_HHMM`.
- Promote only after validation + benchmark pass.

## Preflight
1. Ensure `.env` has valid `NEO4J_*` and `QDRANT_*`.
2. Ensure Neo4j is reachable on `NEO4J_URI`.
3. Ensure Qdrant target is reachable.

## Build + Validate + Evaluate + Index (On-Demand)
```powershell
$release = "kg_$(Get-Date -Format 'yyyyMMdd_HHmm')"

# 1) Build master dataset
python data/build_master_dataset.py --output data/master_vuln_context_$release.jsonl

# 2) Load KG (single-file path)
python scripts/kg/load_kg_master.py --master-file data/master_vuln_context_$release.jsonl

# 3) KG validation gate
python scripts/maintenance/validate_kg.py --strict --sample-cves 50 --seed 42 --output-json eval/results/kg_validation_$release.json --output-md eval/results/kg_validation_$release.md

# 4) Held-out retrieval benchmark gate
python eval/run_graphrag_benchmark.py --benchmark-file eval/heldout_cve_benchmark.jsonl --ground-truth eval/ground_truth_benchmark.jsonl --max-probes 120 --top-k 20 --max-hops 2 --strict --output-json eval/results/graphrag_eval_$release.json --output-csv eval/results/graphrag_eval_$release.csv

# 5) Vector indexing (versioned collection)
python scripts/maintenance/graphrag_embed_index_qdrant_cpu.py --collection-name ("vuln_kg_evidence_v1__" + $release) --max-vectors 225000 --batch-size 64 --qdrant-batch-size 256
```

## Promote
1. Choose versioned collection: `vuln_kg_evidence_v1__<release>`.
2. Set runtime to promoted collection:
```powershell
$env:QDRANT_COLLECTION="vuln_kg_evidence_v1__<release>"
```
3. Keep previous collection unchanged for rollback.

## Versioned Snapshot Requirements
For each release, store:
1. Master dataset path (`data/master_vuln_context_<release>.jsonl`).
2. Neo4j dump path (`backups/neo4j_<release>.dump`).
3. Qdrant collection name (`vuln_kg_evidence_v1__<release>`).
4. Validation artifact paths.
5. Benchmark artifact paths.

## Rollback
Rollback target: previous stable `<release_prev>`.

1. Restore Neo4j from prior dump:
```powershell
# Example command varies by Neo4j setup; run against your DBMS service:
# neo4j-admin database load neo4j --from-path backups --overwrite-destination=true
```
2. Repoint active Qdrant collection:
```powershell
$env:QDRANT_COLLECTION="vuln_kg_evidence_v1__<release_prev>"
```
3. Restore previous master dataset artifact pointer in ops manifest.
4. Smoke verify:
```powershell
python scripts/maintenance/validate_kg.py --sample-cves 20 --seed 42
python eval/run_graphrag_benchmark.py --benchmark-file eval/heldout_cve_benchmark.jsonl --ground-truth eval/ground_truth_benchmark.jsonl --max-probes 20 --top-k 20 --max-hops 2
```

## Release Manifest Update
Append one object per release in `ops/releases_manifest.json` with:
- `release_id`
- `master_dataset`
- `neo4j_dump`
- `qdrant_collection`
- `validation_report_json`
- `benchmark_report_json`
- `status` (`candidate`, `promoted`, `rolled_back`)
