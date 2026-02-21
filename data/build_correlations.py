"""
build_correlations.py
---------------------
Builds a cross-CVE vulnerability correlation graph from all collected data.
This is the core of the "correlation" requirement — it answers questions like:
  "What other CVEs are related to CVE-2021-44228 and why?"

Correlation signals computed:
  1. shared_cwe          — same CWE weakness family
  2. shared_product      — same vendor/product (from NVD CPE data)
  3. shared_attack_tech  — same MITRE ATT&CK technique
  4. shared_capec        — same CAPEC attack pattern (via CWE chain)
  5. kev_temporal        — both in CISA KEV, added within 30 days (campaign signal)
  6. exploit_chain       — appear together in same Exploit-DB entry or PoC repo
  7. shared_owasp        — same OWASP Top 10 category

Output:
  data/raw_correlations.json — per-CVE correlation records
  Correlation training pairs are emitted and appended to training_pairs.jsonl
    by build_dataset.py (which imports load_correlations_lookup())
"""

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── TF-IDF-style signal weighting ─────────────────────────────────────────────
# Signals from very large groups (e.g., shared_product:"linux:linux_kernel"
# connecting 5000 CVEs) are noise, not signal.  We penalise them with an
# inverse-group-frequency weight: weight = log(N / group_size) / log(N),
# clamped to [0.1, 1.0].  Groups with >N members effectively get ~0.1.

IDF_LARGE_GROUP_THRESHOLD = 100  # groups above this start getting penalised


def compute_idf_weights(
    total_cves: int,
    cwe_index:     dict[str, set[str]],
    product_index: dict[str, set[str]],
    owasp_index:   dict[str, set[str]],
    attack_index:  dict[str, set[str]],
    capec_index:   dict[str, set[str]],
) -> dict[str, float]:
    """
    Pre-compute IDF weight for every group key across all indices.
    Returns a dict  {"shared_cwe:CWE-79": 0.45, "shared_product:linux:linux_kernel": 0.12, ...}
    """
    N = max(total_cves, 1)
    log_N = math.log(N) if N > 1 else 1.0
    weights: dict[str, float] = {}

    def _idf(group_size: int) -> float:
        if group_size <= 1:
            return 1.0
        raw = math.log(N / group_size) / log_N
        return max(0.1, min(1.0, raw))

    for key, members in cwe_index.items():
        weights[f"shared_cwe:{key}"] = _idf(len(members))
    for key, members in product_index.items():
        weights[f"shared_product:{key}"] = _idf(len(members))
    for key, members in owasp_index.items():
        weights[f"shared_owasp:{key}"] = _idf(len(members))
    for key, members in attack_index.items():
        weights[f"shared_attack_technique:{key}"] = _idf(len(members))
    for key, members in capec_index.items():
        weights[f"shared_capec:{key}"] = _idf(len(members))

    # Fixed-weight signals (always high quality)
    weights["kev_campaign_temporal"] = 1.0
    weights["exploit_chain_cooccurrence"] = 1.0

    return weights


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: str) -> list | dict:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str[:19], fmt[:len(date_str[:19])])
        except ValueError:
            continue
    return None


# ── Index builders ─────────────────────────────────────────────────────────────

def build_cwe_index(nvd_records: list[dict]) -> dict[str, set[str]]:
    """CWE ID → set of CVE IDs sharing that CWE."""
    index: dict[str, set[str]] = defaultdict(set)
    for rec in nvd_records:
        cve_id = rec.get("cve_id", "")
        cwe    = rec.get("cwe_id", "")
        if cve_id and cwe and cwe.startswith("CWE-"):
            index[cwe].add(cve_id)
    return index


def build_product_index(nvd_records: list[dict]) -> dict[str, set[str]]:
    """
    Product key → set of CVE IDs affecting that product.
    Key is 'vendor:product' derived from affected_software list.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for rec in nvd_records:
        cve_id   = rec.get("cve_id", "")
        software = rec.get("affected_software", [])
        if not cve_id or not software:
            continue
        for sw in software:
            if isinstance(sw, str) and len(sw) > 2:
                # Normalize: lowercase, strip version numbers
                key = re.sub(r"\s+\d[\d.]*\s*$", "", sw.lower().strip())
                key = key[:60]  # cap length
                if key:
                    index[key].add(cve_id)
    return index


def build_attack_technique_index(mitre_data: dict) -> dict[str, set[str]]:
    """ATT&CK technique ID → set of CVE IDs linked to that technique."""
    index: dict[str, set[str]] = defaultdict(set)
    cve_to_techniques: dict[str, list[str]] = mitre_data.get("cve_to_techniques", {})
    for cve_id, technique_ids in cve_to_techniques.items():
        for tid in technique_ids:
            index[tid].add(cve_id.upper())
    return index


def build_capec_index(
    nvd_records:    list[dict],
    mitre_data:     dict,
    owasp_mapper_fn
) -> dict[str, set[str]]:
    """
    CAPEC ID → set of CVE IDs linked via CWE → CAPEC chain.
    Uses the cwe_to_capec mapping from MITRE data.
    """
    cwe_to_capec: dict[str, list[str]] = mitre_data.get("cwe_to_capec", {})
    index: dict[str, set[str]] = defaultdict(set)

    for rec in nvd_records:
        cve_id = rec.get("cve_id", "")
        cwe    = rec.get("cwe_id", "")
        if not cve_id or not cwe:
            continue
        for capec_id in cwe_to_capec.get(cwe, []):
            index[capec_id].add(cve_id)

    return index


def build_kev_temporal_index(kev_records: list[dict]) -> dict[str, set[str]]:
    """
    Find CVE pairs both added to CISA KEV within 30 days of each other.
    These likely belong to the same active exploitation campaign.
    Returns {cve_id: set of temporally-close KEV CVEs}.
    """
    # Sort by date_added
    dated = []
    for rec in kev_records:
        cve_id  = rec.get("cve_id", "")
        date_str = rec.get("date_added", "")
        dt = parse_date(date_str)
        if cve_id and dt:
            dated.append((dt, cve_id))

    dated.sort(key=lambda x: x[0])
    index: dict[str, set[str]] = defaultdict(set)

    # Sliding window of 30 days
    WINDOW = timedelta(days=30)
    for i, (dt_i, cve_i) in enumerate(dated):
        for j in range(i + 1, len(dated)):
            dt_j, cve_j = dated[j]
            if dt_j - dt_i > WINDOW:
                break
            index[cve_i].add(cve_j)
            index[cve_j].add(cve_i)

    return index


def build_exploit_cooccurrence_index(
    exploitdb_records: list[dict],
    poc_records:       list[dict],
) -> dict[str, set[str]]:
    """
    CVE ID → set of CVEs that appear together in the same Exploit-DB entry
    or the same PoC GitHub repo. Strong signal for exploit chain/campaign.
    """
    index: dict[str, set[str]] = defaultdict(set)

    all_sources = exploitdb_records + poc_records
    for item in all_sources:
        cves = [c.upper() for c in item.get("cves_mentioned", []) if c]
        if len(cves) < 2:
            continue
        # All pairs of CVEs in this record
        for i, c1 in enumerate(cves):
            for c2 in cves[i+1:]:
                if c1 != c2:
                    index[c1].add(c2)
                    index[c2].add(c1)

    return index


def build_owasp_index(nvd_records: list[dict], owasp_mapper_fn) -> dict[str, set[str]]:
    """OWASP category → set of CVE IDs in that category."""
    index: dict[str, set[str]] = defaultdict(set)
    for rec in nvd_records:
        cve_id = rec.get("cve_id", "")
        cwe    = rec.get("cwe_id", "")
        if not cve_id or not cwe:
            continue
        owasp = owasp_mapper_fn(cwe)
        if owasp and owasp != "Unknown":
            index[owasp].add(cve_id)
    return index


# ── Correlation record builder ─────────────────────────────────────────────────

def build_correlation_record(
    cve_id:          str,
    nvd_rec:         dict,
    cwe_index:       dict[str, set[str]],
    product_index:   dict[str, set[str]],
    attack_index:    dict[str, set[str]],
    capec_index:     dict[str, set[str]],
    kev_temporal:    dict[str, set[str]],
    exploit_cooccur: dict[str, set[str]],
    owasp_index:     dict[str, set[str]],
    mitre_data:      dict,
    idf_weights:     dict[str, float] | None = None,
    max_related:     int = 10,
) -> dict:
    """
    Build a correlation record for a single CVE, gathering all related CVEs
    with their correlation signals explained.
    """
    cwe   = nvd_rec.get("cwe_id", "")
    desc  = nvd_rec.get("description", "")
    sw    = nvd_rec.get("affected_software", [])

    # Collect all related CVEs with their signal types
    related: dict[str, list[str]] = defaultdict(list)  # {cve_id: [signals]}

    # Signal 1: shared CWE
    for rel_cve in cwe_index.get(cwe, set()):
        if rel_cve != cve_id:
            related[rel_cve].append(f"shared_cwe:{cwe}")

    # Signal 2: shared product
    for sw_item in sw:
        if isinstance(sw_item, str):
            key = re.sub(r"\s+\d[\d.]*\s*$", "", sw_item.lower().strip())[:60]
            for rel_cve in product_index.get(key, set()):
                if rel_cve != cve_id:
                    related[rel_cve].append(f"shared_product:{sw_item[:40]}")

    # Signal 3: ATT&CK technique
    cve_to_techniques = mitre_data.get("cve_to_techniques", {})
    my_techniques = cve_to_techniques.get(cve_id, [])
    for tid in my_techniques:
        for rel_cve in attack_index.get(tid, set()):
            if rel_cve != cve_id:
                related[rel_cve].append(f"shared_attack_technique:{tid}")

    # Signal 4: CAPEC
    cwe_to_capec = mitre_data.get("cwe_to_capec", {})
    my_capecs = cwe_to_capec.get(cwe, [])
    for capec_id in my_capecs:
        for rel_cve in capec_index.get(capec_id, set()):
            if rel_cve != cve_id:
                related[rel_cve].append(f"shared_capec:{capec_id}")

    # Signal 5: KEV temporal proximity (campaign)
    for rel_cve in kev_temporal.get(cve_id, set()):
        if rel_cve != cve_id:
            related[rel_cve].append("kev_campaign_temporal")

    # Signal 6: Exploit co-occurrence
    for rel_cve in exploit_cooccur.get(cve_id, set()):
        if rel_cve != cve_id:
            related[rel_cve].append("exploit_chain_cooccurrence")

    # Signal 7: Shared OWASP
    for owasp_cat, owasp_cves in owasp_index.items():
        if cve_id in owasp_cves:
            for rel_cve in owasp_cves:
                if rel_cve != cve_id:
                    related[rel_cve].append(f"shared_owasp:{owasp_cat[:30]}")

    # ── Temporal decay multiplier ──────────────────────────────────────
    # Recent CVE pairs get full weight; older pairs decay.
    # Extract year from cve_id (e.g., CVE-2021-44228 → 2021)
    def _recency_multiplier(cve_a: str, cve_b: str) -> float:
        """Return decay factor based on the *older* CVE's year."""
        try:
            year_a = int(cve_a.split("-")[1])
            year_b = int(cve_b.split("-")[1])
        except (IndexError, ValueError):
            return 1.0
        older_year = min(year_a, year_b)
        current_year = datetime.now().year
        age = current_year - older_year
        if age <= 1:
            return 1.0
        elif age <= 3:
            return 0.85
        elif age <= 5:
            return 0.65
        else:
            return 0.5

    # Score and rank: IDF-weighted signals × temporal decay
    def _score_signals(rel_cve: str, signals: list[str]) -> float:
        if idf_weights:
            # Sum the IDF weight of each unique signal
            unique = set(signals)
            raw = sum(idf_weights.get(s, idf_weights.get(s.split(":")[0], 0.5)) for s in unique)
        else:
            raw = float(len(set(signals)))
        return raw * _recency_multiplier(cve_id, rel_cve)

    scored = sorted(
        [(rel_cve, signals) for rel_cve, signals in related.items()],
        key=lambda x: _score_signals(x[0], x[1]),
        reverse=True,
    )

    top_related = [
        {
            "cve_id":           rel_cve,
            "correlation_score": round(_score_signals(rel_cve, signals), 3),
            "signals":          list(set(signals)),
        }
        for rel_cve, signals in scored[:max_related]
    ]

    return {
        "cve_id":           cve_id,
        "cwe_id":           cwe,
        "description":      desc[:500],
        "attack_techniques": my_techniques,
        "capec_patterns":   my_capecs,
        "related_vulnerabilities": top_related,
        "correlation_signal_count": len(related),
    }


# ── Training pair generator ────────────────────────────────────────────────────

def to_correlation_training_pairs(corr_record: dict, mitre_data: dict) -> list[dict]:
    """
    Generate training pairs for the 'vulnerability_correlation' layer.
    These teach the model to reason about HOW and WHY vulnerabilities are related.
    """
    cve_id   = corr_record["cve_id"]
    desc     = corr_record["description"]
    related  = corr_record["related_vulnerabilities"]
    techniques = corr_record["attack_techniques"]
    capecs   = corr_record["capec_patterns"]

    pairs = []

    if not related:
        return pairs

    # Build technique name lookup
    tech_names = {
        t["technique_id"]: t["technique_name"]
        for t in mitre_data.get("techniques", [])
    }

    # ── Pair 1: "What CVEs are correlated with X?" ─────────────────────────
    related_summary = []
    for rel in related[:5]:
        sig_types = list({s.split(":")[0] for s in rel["signals"]})
        related_summary.append(
            f"  - {rel['cve_id']} (correlation score: {rel['correlation_score']}, "
            f"signals: {', '.join(sig_types)})"
        )

    pairs.append({
        "instruction": f"What vulnerabilities are correlated with {cve_id} and why?",
        "input":       desc,
        "output": (
            f"Correlated vulnerabilities for {cve_id}:\n\n"
            + "\n".join(related_summary)
            + f"\n\nCorrelation signals used:\n"
            + "  - shared_cwe: Same underlying weakness class (CWE)\n"
            + "  - shared_product: Same affected vendor/product\n"
            + "  - shared_attack_technique: Same MITRE ATT&CK technique\n"
            + "  - kev_campaign_temporal: Both actively exploited in same time window\n"
            + "  - exploit_chain_cooccurrence: Appear together in known exploit chains"
        ),
        "layer": "vulnerability_correlation",
        "agent": "Correlation Agent",
    })

    # ── Pair 2: ATT&CK technique context ──────────────────────────────────
    if techniques:
        tech_descriptions = [
            f"{tid} ({tech_names.get(tid, 'Unknown technique')})"
            for tid in techniques[:3]
        ]
        pairs.append({
            "instruction": f"Which MITRE ATT&CK techniques are associated with {cve_id}?",
            "input":       desc,
            "output": (
                f"MITRE ATT&CK techniques for {cve_id}:\n\n"
                + "\n".join(f"  - {t}" for t in tech_descriptions)
                + f"\n\nThese techniques map to the following correlated CVEs "
                  f"(vulnerabilities exploited via the same techniques):\n"
                + "\n".join(
                    f"  - {r['cve_id']}"
                    for r in related[:5]
                    if any("attack_technique" in s for s in r["signals"])
                )
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    # ── Pair 3: Campaign / KEV cluster context ─────────────────────────────
    kev_related = [
        r for r in related
        if any("kev_campaign" in s for s in r["signals"])
    ]
    if kev_related:
        pairs.append({
            "instruction": f"Is {cve_id} part of a known ransomware or exploit campaign? What other CVEs are in the same campaign?",
            "input":       desc,
            "output": (
                f"{cve_id} is part of an active exploitation cluster based on CISA KEV temporal analysis.\n\n"
                f"CVEs added to CISA KEV within the same 30-day window (likely same campaign):\n"
                + "\n".join(f"  - {r['cve_id']}" for r in kev_related[:6])
                + "\n\nThis temporal clustering indicates these CVEs may be used together "
                  "in coordinated attacks or by the same threat actor groups."
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    # ── Pair 4: Exploit chain context ─────────────────────────────────────
    chain_related = [
        r for r in related
        if any("exploit_chain" in s for s in r["signals"])
    ]
    if chain_related:
        pairs.append({
            "instruction": f"What exploit chains or attack sequences involve {cve_id}?",
            "input":       desc,
            "output": (
                f"Exploit chain analysis for {cve_id}:\n\n"
                f"The following CVEs have been observed in the same exploit code or PoC repositories:\n"
                + "\n".join(f"  - {r['cve_id']}" for r in chain_related[:5])
                + "\n\nCo-occurrence in exploit code suggests these may be used together "
                  "in multi-stage attacks (e.g., initial access CVE chained with a privilege "
                  "escalation CVE for full compromise)."
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def run(out: str = "data/raw_correlations.json"):
    """Build correlation graph from all raw data sources."""
    print("Building vulnerability correlation graph...")

    # Load all data sources
    nvd_records    = load_json("data/raw_nvd.json")
    kev_records    = load_json("data/raw_cisa_kev.json")
    exploitdb_raw  = load_json("data/raw_exploitdb.json")
    mitre_data     = {}
    poc_records    = []

    mitre_path = Path("data/raw_mitre_attack.json")
    if mitre_path.exists():
        mitre_data = load_json(str(mitre_path))
        print(f"  ATT&CK techniques loaded: {len(mitre_data.get('techniques', []))}")
        print(f"  CAPEC patterns loaded:    {len(mitre_data.get('capec_patterns', []))}")
    else:
        print("  ⚠️  raw_mitre_attack.json not found — ATT&CK/CAPEC signals will be empty")
        print("      Run: python -c \"from data.crawl_mitre_attack import run; run()\"")

    vendor_path = Path("data/raw_vendor_advisories.json")
    if vendor_path.exists():
        vendor_records = load_json(str(vendor_path))
        poc_records = [r for r in vendor_records if r.get("source") == "poc_github"]
        print(f"  Vendor advisories loaded: {len(vendor_records)}")
    else:
        print("  ℹ️  raw_vendor_advisories.json not found — PoC cooccurrence signals limited")

    print(f"  NVD records: {len(nvd_records)}")
    print(f"  KEV records: {len(kev_records)}")

    # Import OWASP mapper
    try:
        import sys
        sys.path.insert(0, "data")
        from owasp_mapper import get_owasp_category as owasp_fn
    except ImportError:
        owasp_fn = lambda cwe: "Unknown"

    # Build all indices
    print("\nBuilding correlation indices...")
    cwe_index       = build_cwe_index(nvd_records)
    product_index   = build_product_index(nvd_records)
    attack_index    = build_attack_technique_index(mitre_data)
    capec_index     = build_capec_index(nvd_records, mitre_data, owasp_fn)
    kev_temporal    = build_kev_temporal_index(kev_records)
    exploit_cooccur = build_exploit_cooccurrence_index(exploitdb_raw, poc_records)
    owasp_index     = build_owasp_index(nvd_records, owasp_fn)

    print(f"  CWE groups:              {len(cwe_index)}")
    print(f"  Product groups:          {len(product_index)}")
    print(f"  ATT&CK technique groups: {len(attack_index)}")
    print(f"  CAPEC groups:            {len(capec_index)}")
    print(f"  KEV temporal clusters:   {len(kev_temporal)}")
    print(f"  Exploit co-occurrences:  {len(exploit_cooccur)}")
    print(f"  OWASP groups:            {len(owasp_index)}")

    # ── TF-IDF signal weighting ──────────────────────────────────────────
    idf_weights = compute_idf_weights(
        total_cves   = len(nvd_records),
        cwe_index    = cwe_index,
        product_index= product_index,
        owasp_index  = owasp_index,
        attack_index = attack_index,
        capec_index  = capec_index,
    )
    # Report the noisiest groups getting penalised
    large_groups = [
        (k, w) for k, w in idf_weights.items() if w < 0.3
    ]
    if large_groups:
        print(f"\n  IDF-penalised signals (weight < 0.3): {len(large_groups)}")
        for k, w in sorted(large_groups, key=lambda x: x[1])[:10]:
            print(f"    {k[:55]:55s}  weight={w:.3f}")

    # Build NVD lookup for quick access
    nvd_by_cve = {r.get("cve_id", ""): r for r in nvd_records if r.get("cve_id")}

    # Build correlation records
    print("\nComputing per-CVE correlations...")
    correlation_records = []
    cves_with_correlations = 0

    for cve_id, nvd_rec in nvd_by_cve.items():
        corr = build_correlation_record(
            cve_id          = cve_id,
            nvd_rec         = nvd_rec,
            cwe_index       = cwe_index,
            product_index   = product_index,
            attack_index    = attack_index,
            capec_index     = capec_index,
            kev_temporal    = kev_temporal,
            exploit_cooccur = exploit_cooccur,
            owasp_index     = owasp_index,
            mitre_data      = mitre_data,
            idf_weights     = idf_weights,
        )
        correlation_records.append(corr)
        if corr["related_vulnerabilities"]:
            cves_with_correlations += 1

    # Save correlation records
    with open(out, "w", encoding="utf-8") as f:
        json.dump(correlation_records, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Correlation Summary:")
    print(f"  CVEs processed:          {len(correlation_records)}")
    print(f"  CVEs with correlations:  {cves_with_correlations}")
    avg_related = sum(
        len(r["related_vulnerabilities"]) for r in correlation_records
    ) / max(len(correlation_records), 1)
    print(f"  Avg related per CVE:     {avg_related:.1f}")
    print(f"\n✅ Saved correlation graph → {out}")

    return correlation_records


def load_correlations_lookup(path: str = "data/raw_correlations.json") -> dict[str, dict]:
    """
    Load correlation records as a CVE → correlation_record lookup.
    Called by build_dataset.py to enrich records and generate
    correlation training pairs.
    """
    records = load_json(path)
    if isinstance(records, list):
        return {r["cve_id"]: r for r in records if r.get("cve_id")}
    return {}


if __name__ == "__main__":
    run()