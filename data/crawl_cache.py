"""
data/crawler_cache.py
---------------------
Shared cache-staleness utilities for all crawlers.

PROBLEM: Every run_pipeline.py invocation re-fetches everything from scratch.
  - NVD full fetch:       ~45 min, 5000+ API calls
  - Blog crawl:           ~15 min, 200+ HTTP requests
  - STIX bundle:          5 MB download every time
  - CISA KEV:             fine (tiny), but still unnecessary on --correlate runs

SOLUTION: is_stale() — check file mtime before any network call.
  Each crawler wraps its main fetch with:
      if is_stale(out_path, max_age_hours=24):
          ... do fetch ...
      else:
          print("  ✅ Cache fresh — skipping fetch")

INCREMENTAL NVD: The NVD API supports pubStartDate / pubEndDate query params.
  nvd_start_date() returns the ISO 8601 date of the most recent record in
  raw_nvd.json so crawl_nvd.py can fetch ONLY new CVEs added since last run.
  A full pipeline re-run for a correlation fix (--correlate) goes from
  45 min → <5 min this way.

Usage in a crawler:
    from data.crawler_cache import is_stale, nvd_start_date, cache_summary

    def run(out: str = "data/raw_nvd.json"):
        out_path = Path(out)
        if not is_stale(out_path, max_age_hours=24):
            print(f"  ✅ {out_path.name} is fresh — skipping NVD fetch")
            return
        # ... do fetch ...
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Per-source default max ages ────────────────────────────────────────────────
# Tune these based on how frequently each source actually updates.
DEFAULT_MAX_AGE: dict[str, int] = {
    "raw_nvd.json":               24,    # NVD updates every few hours but daily is fine
    "raw_epss.json":              24,    # EPSS scores update daily
    "raw_github.json":            24,
    "raw_blogs.json":             72,    # blogs don't change that fast
    "raw_exploitdb.json":         48,
    "raw_cisa_kev.json":          12,    # KEV is high-value, check twice daily
    "raw_papers.json":            72,
    "raw_mitre_attack.json":      168,   # ATT&CK releases quarterly; weekly check fine
    "raw_vendor_advisories.json": 24,
    "raw_closed.json":            24,
    "raw_correlations.json":      1,     # always rebuild after crawl changes
    "raw_cooccurrence.json":      1,
}


def is_stale(
    path:         Path | str,
    max_age_hours: int | None = None,
) -> bool:
    """
    Return True if the file doesn't exist or is older than max_age_hours.

    Args:
        path:          Path to the cache file.
        max_age_hours: Max acceptable age in hours. If None, looks up the
                       filename in DEFAULT_MAX_AGE; defaults to 24h if unknown.

    Returns:
        True  → file is missing or stale  → crawler should re-fetch
        False → file is fresh             → crawler can skip
    """
    p = Path(path)

    if not p.exists():
        return True  # always fetch if missing

    if not p.stat().st_size:
        return True  # empty file is as good as missing

    if max_age_hours is None:
        max_age_hours = DEFAULT_MAX_AGE.get(p.name, 24)

    age_hours = (time.time() - p.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def cache_age_str(path: Path | str) -> str:
    """Human-readable age string for a cache file, e.g. '2h 14m' or 'MISSING'."""
    p = Path(path)
    if not p.exists():
        return "MISSING"
    age_secs = time.time() - p.stat().st_mtime
    hours    = int(age_secs // 3600)
    minutes  = int((age_secs % 3600) // 60)
    if hours >= 24:
        return f"{hours // 24}d {hours % 24}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def cache_summary(data_dir: Path | str = "data") -> None:
    """
    Print a table of all raw_*.json files with their age and staleness status.
    Useful at the start of run_pipeline.py to see what will be skipped.
    """
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob("raw_*.json"))

    if not files:
        print("  No cached files found in data/")
        return

    print(f"\n  {'File':<36} {'Age':>8}  {'Status'}")
    print(f"  {'─'*36}  {'─'*8}  {'─'*10}")
    for f in files:
        max_age = DEFAULT_MAX_AGE.get(f.name, 24)
        age_str = cache_age_str(f)
        stale   = is_stale(f, max_age_hours=max_age)
        status  = "STALE — will re-fetch" if stale else f"fresh (max {max_age}h)"
        flag    = "⚠️ " if stale else "✅ "
        print(f"  {f.name:<36} {age_str:>8}  {flag}{status}")
    print()


# ── NVD incremental fetch helpers ─────────────────────────────────────────────

def nvd_start_date(
    nvd_path: Path | str = "data/raw_nvd.json",
    lookback_days: int   = 7,
) -> str | None:
    """
    Return the ISO 8601 pubStartDate for an incremental NVD fetch.

    Reads raw_nvd.json, finds the most recent 'published' field, then
    subtracts lookback_days as a safety buffer (so we don't miss CVEs
    that were added to NVD with a slight delay).

    Returns None if raw_nvd.json doesn't exist or has no 'published' fields
    (callers should fall back to a full fetch in that case).

    Example return value: "2025-01-15T00:00:00.000"
    """
    p = Path(nvd_path)
    if not p.exists():
        return None

    try:
        with open(p, encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return None

    dates = [
        r.get("published", "")
        for r in records
        if r.get("published", "")
    ]
    if not dates:
        return None

    # Sort ISO dates lexicographically (they're zero-padded so this works)
    latest = sorted(dates)[-1]

    try:
        dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
    except ValueError:
        return None

    # Subtract lookback buffer
    from datetime import timedelta
    start_dt = dt - timedelta(days=lookback_days)
    return start_dt.strftime("%Y-%m-%dT00:00:00.000")


def nvd_record_count(nvd_path: Path | str = "data/raw_nvd.json") -> int:
    """Return number of records in raw_nvd.json, or 0 if missing."""
    p = Path(nvd_path)
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:
        return 0