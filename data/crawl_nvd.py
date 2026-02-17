"""
crawl_nvd.py
------------
Fetches CVE records from NVD API v2.0
Covers: vulnerability_name, cve_id, cwe_id, description,
        cvss_score, cvss_severity, affected software
Output: raw_nvd.json
"""

import requests
import json
import time
from tqdm import tqdm

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

SEVERITY_MAP = {
    "CRITICAL": "Critical",
    "HIGH":     "High",
    "MEDIUM":   "Medium",
    "LOW":      "Low"
}

def fetch_nvd_batch(start_index=0, results_per_page=2000):
    params = {
        "startIndex":      start_index,
        "resultsPerPage":  results_per_page
    }
    resp = requests.get(NVD_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def extract_cwe(cve_data):
    """Pull CWE IDs from the weaknesses block."""
    cwe_ids = []
    for w in cve_data.get("weaknesses", []):
        for d in w.get("description", []):
            if d["lang"] == "en" and d["value"].startswith("CWE-"):
                cwe_ids.append(d["value"])
    return cwe_ids[0] if cwe_ids else ""          # return first CWE

def extract_cvss(cve_data):
    """Return (score, severity) preferring CVSSv3.1 then v3.0 then v2."""
    metrics = cve_data.get("metrics", {})
    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        if key in metrics:
            data = metrics[key][0]["cvssData"]
            score    = data.get("baseScore", "")
            severity = data.get("baseSeverity", "")
            return score, SEVERITY_MAP.get(severity.upper(), severity)
    return "", ""

def extract_cpe_tech(cve_data):
    """Best-effort extraction of affected software from CPE strings."""
    tech = []
    for config in cve_data.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                cpe = match.get("criteria", "")
                parts = cpe.split(":")
                if len(parts) > 4:
                    tech.append(parts[4])        # product name from CPE
    return list(set(tech))[:5]                   # top 5 unique products

def parse_record(item):
    cve       = item["cve"]
    cve_id    = cve["id"]
    desc      = next(
        (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
        ""
    )
    cwe_id             = extract_cwe(cve)
    cvss_score, sev    = extract_cvss(cve)
    affected_software  = extract_cpe_tech(cve)
    references         = [r["url"] for r in cve.get("references", [])]

    return {
        "cve_id":            cve_id,
        "vulnerability_name": f"{cwe_id} vulnerability" if cwe_id else cve_id,
        "cwe_id":            cwe_id,
        "description":       desc,
        "cvss_score":        cvss_score,
        "cvss_severity":     sev,
        "affected_software": affected_software,
        "published":         cve.get("published", ""),
        "references":        references[:5]
    }

def run(total=10000, batch=2000, out="data/raw_nvd.json"):
    all_records = []
    for start in tqdm(range(0, total, batch), desc="NVD batches"):
        try:
            data = fetch_nvd_batch(start, batch)
            for item in data.get("vulnerabilities", []):
                all_records.append(parse_record(item))
            time.sleep(0.6)             # respect NVD rate limit
        except Exception as e:
            print(f"  ⚠️  Batch {start} failed: {e}")
            time.sleep(2)

    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n✅ Saved {len(all_records)} NVD records → {out}")

if __name__ == "__main__":
    run()