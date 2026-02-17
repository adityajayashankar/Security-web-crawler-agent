"""
crawl_github.py
---------------
Fetches GitHub Security Advisories via REST API.
Covers: fix_recommendation, affected packages, patched versions,
        cve_id cross-reference, vulnerability descriptions
Output: raw_github.json

Docs: https://docs.github.com/en/rest/security-advisories
Free — but set GITHUB_TOKEN env var for higher rate limits (5000 req/hr vs 60).
"""

import requests
import json
import os
import time
from tqdm import tqdm

GITHUB_API = "https://api.github.com/advisories"

def get_headers():
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def fetch_page(page=1, per_page=100):
    params = {
        "per_page": per_page,
        "page":     page,
        "type":     "reviewed"          # only reviewed (high quality) advisories
    }
    resp = requests.get(GITHUB_API, headers=get_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def parse_advisory(adv):
    # Extract affected package info
    affected = adv.get("vulnerabilities", [])
    packages    = []
    fix_versions = []
    languages   = []

    for v in affected:
        pkg = v.get("package", {})
        if pkg.get("name"):
            packages.append(pkg["name"])
        if pkg.get("ecosystem"):
            languages.append(pkg["ecosystem"])
        pv = v.get("patched_versions", "")
        if pv:
            fix_versions.append(pv)

    # Build a fix recommendation string
    fix = ""
    if fix_versions:
        fix = f"Update to patched version(s): {', '.join(fix_versions)}"
    elif adv.get("summary"):
        fix = f"Refer to advisory: {adv['summary']}"

    return {
        "source":           "github_advisory",
        "ghsa_id":          adv.get("ghsa_id", ""),
        "cve_id":           adv.get("cve_id", ""),
        "vulnerability_name": adv.get("summary", ""),
        "description":      adv.get("description", ""),
        "cvss_score":       adv.get("cvss", {}).get("score", ""),
        "cvss_severity":    adv.get("severity", "").capitalize(),
        "cwe_ids":          [c["cwe_id"] for c in adv.get("cwes", [])],
        "affected_packages": packages,
        "languages":        list(set(languages)),
        "fix_recommendation": fix,
        "published":        adv.get("published_at", ""),
        "references":       adv.get("references", [])[:5]
    }

def run(max_pages=30, out="data/raw_github.json"):
    all_advisories = []

    for page in tqdm(range(1, max_pages + 1), desc="GitHub advisories"):
        try:
            items = fetch_page(page)
            if not items:
                break
            for item in items:
                all_advisories.append(parse_advisory(item))
            time.sleep(0.5)
        except requests.exceptions.HTTPError as e:
            if "rate limit" in str(e).lower():
                print("  ⚠️  Rate limited. Set GITHUB_TOKEN env var for higher limits.")
                break
            print(f"  ⚠️  Page {page} failed: {e}")

    with open(out, "w") as f:
        json.dump(all_advisories, f, indent=2)
    print(f"\n✅ Saved {len(all_advisories)} GitHub advisories → {out}")

if __name__ == "__main__":
    run()