"""
crawl_epss.py
-------------
Fetches EPSS (Exploit Prediction Scoring System) scores from FIRST API.
Covers: epss_score (exploit probability 0-1), percentile ranking
Output: raw_epss.json  →  { "CVE-XXXX-YYYY": 0.87, ... }

EPSS API docs: https://api.first.org/data/v1/epss
Free, no auth needed.
"""

import requests
import json
import time
from tqdm import tqdm

EPSS_API_URL = "https://api.first.org/data/v1/epss"

def fetch_epss_page(offset=0, limit=100):
    params = {"offset": offset, "limit": limit, "order": "!epss"}
    resp = requests.get(EPSS_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_epss_for_cves(cve_ids: list[str]) -> dict:
    """
    Given a list of CVE IDs, return a dict {cve_id: epss_score}.
    Batches 100 at a time (API limit).
    """
    result = {}
    batch_size = 100

    for i in tqdm(range(0, len(cve_ids), batch_size), desc="EPSS lookup"):
        batch = cve_ids[i : i + batch_size]
        cve_param = ",".join(batch)
        try:
            resp = requests.get(
                EPSS_API_URL,
                params={"cve": cve_param},
                timeout=30
            )
            resp.raise_for_status()
            for entry in resp.json().get("data", []):
                result[entry["cve"]] = float(entry["epss"])
            time.sleep(0.3)
        except Exception as e:
            print(f"  ⚠️  EPSS batch {i} failed: {e}")

    return result

def run(nvd_path="data/raw_nvd.json", out="data/raw_epss.json"):
    # Load CVE IDs from NVD data
    with open(nvd_path) as f:
        nvd = json.load(f)

    cve_ids = [r["cve_id"] for r in nvd if r.get("cve_id")]
    print(f"Fetching EPSS for {len(cve_ids)} CVEs...")

    epss_map = fetch_epss_for_cves(cve_ids)

    with open(out, "w") as f:
        json.dump(epss_map, f, indent=2)

    print(f"\n✅ Saved EPSS scores for {len(epss_map)} CVEs → {out}")
    print(f"   (CVEs without EPSS data: {len(cve_ids) - len(epss_map)})")

if __name__ == "__main__":
    run()