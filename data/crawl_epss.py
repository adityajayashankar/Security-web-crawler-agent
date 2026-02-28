"""
crawl_epss.py
-------------
Fetches EPSS (Exploit Prediction Scoring System) scores from FIRST API.
Output: raw_epss.json  →  { "CVE-XXXX-YYYY": 0.87, ... }

Scaling improvements:
  - Resume mode: reuse existing raw_epss.json and fetch only missing CVEs
  - Retries with exponential backoff for 429/5xx transient failures
  - Optional concurrent workers for large CVE lists
  - Periodic checkpoint writes to avoid losing progress on long runs
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

EPSS_API_URL = "https://api.first.org/data/v1/epss"
EPSS_BATCH_SIZE = int(os.getenv("EPSS_BATCH_SIZE", "100"))  # API supports up to 100 CVEs/query
EPSS_WORKERS = max(1, int(os.getenv("EPSS_WORKERS", "4")))
EPSS_TIMEOUT = int(os.getenv("EPSS_TIMEOUT", "40"))
EPSS_MAX_RETRIES = int(os.getenv("EPSS_MAX_RETRIES", "5"))
EPSS_CHECKPOINT_EVERY = int(os.getenv("EPSS_CHECKPOINT_EVERY", "200"))


def _load_existing_epss(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in raw.items():
            try:
                out[str(k).upper()] = float(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _save_epss(path: Path, epss_map: dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(epss_map, f, indent=2, ensure_ascii=False)


def _fetch_batch(batch: list[str], max_retries: int = EPSS_MAX_RETRIES) -> tuple[dict[str, float], str | None]:
    params = {"cve": ",".join(batch)}
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(EPSS_API_URL, params=params, timeout=EPSS_TIMEOUT)
            # Retry transient server/rate-limit errors.
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_error = f"{resp.status_code} {resp.reason}"
                sleep_s = min(25.0, (1.5 ** attempt) + random.uniform(0.05, 0.35))
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json().get("data", [])
            out = {}
            for entry in data:
                cve = str(entry.get("cve", "")).upper().strip()
                if not cve:
                    continue
                try:
                    out[cve] = float(entry.get("epss", 0.0))
                except Exception:
                    continue
            return out, None
        except Exception as e:
            last_error = str(e)
            sleep_s = min(25.0, (1.5 ** attempt) + random.uniform(0.05, 0.35))
            time.sleep(sleep_s)
    return {}, last_error or "unknown error"


def fetch_epss_for_cves(cve_ids: list[str], existing: dict[str, float] | None = None) -> tuple[dict[str, float], int]:
    """
    Given CVE IDs, return updated epss map and failed batch count.
    """
    epss_map = dict(existing or {})
    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for c in cve_ids:
        cid = str(c).upper().strip()
        if cid and cid not in seen:
            seen.add(cid)
            deduped.append(cid)

    missing = [c for c in deduped if c not in epss_map]
    if not missing:
        return epss_map, 0

    batches = [missing[i : i + EPSS_BATCH_SIZE] for i in range(0, len(missing), EPSS_BATCH_SIZE)]
    failures = 0
    done = 0

    with ThreadPoolExecutor(max_workers=EPSS_WORKERS) as ex:
        futures = [ex.submit(_fetch_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EPSS lookup"):
            batch_map, err = fut.result()
            if err:
                failures += 1
            epss_map.update(batch_map)
            done += 1
            if done % EPSS_CHECKPOINT_EVERY == 0:
                # Caller saves final file; checkpoint to local temp snapshot for long runs.
                _save_epss(Path("data/raw_epss.partial.json"), epss_map)

    return epss_map, failures


def run(nvd_path="data/raw_nvd.json", out="data/raw_epss.json"):
    with open(nvd_path, encoding="utf-8") as f:
        nvd = json.load(f)

    cve_ids = [r["cve_id"] for r in nvd if r.get("cve_id")]
    out_path = Path(out)
    existing = _load_existing_epss(out_path)

    print(f"Fetching EPSS for {len(cve_ids)} CVEs...")
    if existing:
        print(f"  Resume mode: {len(existing):,} CVEs already cached in {out_path.name}")

    start = time.time()
    epss_map, failed_batches = fetch_epss_for_cves(cve_ids, existing=existing)
    _save_epss(out_path, epss_map)
    elapsed = time.time() - start

    covered = sum(1 for c in cve_ids if c in epss_map)
    print(f"\n✅ Saved EPSS scores for {covered:,} CVEs → {out}")
    print(f"   (CVEs without EPSS data: {len(cve_ids) - covered:,})")
    print(f"   Failed batches after retries: {failed_batches}")
    print(f"   Elapsed: {elapsed/60:.1f} min | workers={EPSS_WORKERS} batch_size={EPSS_BATCH_SIZE}")


if __name__ == "__main__":
    run()
