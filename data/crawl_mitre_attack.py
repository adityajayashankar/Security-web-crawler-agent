"""
crawl_mitre_attack.py
---------------------
Downloads MITRE ATT&CK (Enterprise) STIX bundles and CAPEC attack patterns.

FIX: ATT&CK→CVE community mapping 404
  The center-for-threat-informed-defense/attack_to_cve repo moved / renamed.
  fetch_attack_cve_mapping() now tries 4 fallback URLs in order, then falls
  back to mining CVE references directly from the STIX bundle descriptions.
  Only 16 of 691 techniques had CVE links before — the STIX mining + NVD
  cross-reference now retrieves a much richer set.

Data sources (all authoritative MITRE feeds):
  - MITRE ATT&CK Enterprise STIX 2.1 bundle (GitHub: mitre-attack/attack-stix-data)
  - MITRE CAPEC XML (capec.mitre.org) → attack pattern → CWE relationships
  - NVD CVE → CWE mappings (already in raw_nvd.json, reused for chain building)

Output: data/raw_mitre_attack.json
"""

import requests
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# ── MITRE ATT&CK STIX bundle ──────────────────────────────────────────────────
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)

# ── MITRE CAPEC XML ────────────────────────────────────────────────────────────
CAPEC_XML_URL = "https://capec.mitre.org/data/xml/capec_latest.xml"

# ── ATT&CK → CVE community mapping — multiple fallback URLs ───────────────────
# The original URL (attack_to_cve/main/data/attack-to-cve.json) returns 404.
# We try these in order; if all fail we fall back to STIX description mining.
ATTACK_CVE_MAPPING_URLS = [
    # Current canonical location (repo was renamed)
    "https://raw.githubusercontent.com/center-for-threat-informed-defense/"
    "attack-to-cve/main/data/attack-to-cve.json",
    # Alternate branch
    "https://raw.githubusercontent.com/center-for-threat-informed-defense/"
    "attack-to-cve/master/data/attack-to-cve.json",
    # Old repo name (underscore)
    "https://raw.githubusercontent.com/center-for-threat-informed-defense/"
    "attack_to_cve/main/data/attack-to-cve.json",
    # Old repo name, master branch
    "https://raw.githubusercontent.com/center-for-threat-informed-defense/"
    "attack_to_cve/master/data/attack-to-cve.json",
    # Backup: MITRE CTI GitHub (contains technique→CVE cross-refs in STIX objects)
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/"
    "enterprise-attack.json",
]

# ── NVD cross-reference path (used for extra CVE→technique enrichment) ─────────
NVD_PATH = Path("data") / "raw_nvd.json"


# ── ATT&CK STIX Parser ────────────────────────────────────────────────────────

def fetch_attack_stix() -> list[dict]:
    """
    Download and parse the full MITRE ATT&CK Enterprise STIX 2.1 bundle.
    Extracts techniques with CVE references, CAPEC mappings, tactics,
    mitigations, and platform coverage.
    """
    print("Fetching MITRE ATT&CK Enterprise STIX bundle...")
    try:
        resp = requests.get(ATTACK_STIX_URL, timeout=120)
        resp.raise_for_status()
        bundle = resp.json()
    except Exception as e:
        print(f"  ❌ ATT&CK STIX fetch failed: {e}")
        return []

    objects = bundle.get("objects", [])
    print(f"  STIX bundle: {len(objects)} objects")

    # ── Build mitigation lookup first ─────────────────────────────────────
    technique_mitigations: dict[str, list[str]] = {}
    mitigation_names: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") == "course-of-action":
            mitigation_names[obj["id"]] = obj.get("name", "")
    for obj in objects:
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
            mid  = obj.get("source_ref", "")
            tid  = obj.get("target_ref", "")
            name = mitigation_names.get(mid, "")
            if name:
                technique_mitigations.setdefault(tid, []).append(name)

    techniques = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("x_mitre_deprecated") or obj.get("revoked"):
            continue
        # Only keep techniques that have an external_id like T\d{4}
        ext_refs  = obj.get("external_references", [])
        tech_id   = ""
        cve_refs  = []
        capec_ids = []

        for ref in ext_refs:
            src_name = ref.get("source_name", "")
            if src_name == "mitre-attack":
                tech_id = ref.get("external_id", "")
            elif src_name == "cve":
                cve_refs.append(ref.get("external_id", "").upper())
            elif src_name == "capec":
                capec_ids.append(ref.get("external_id", ""))

            # Mine CVE IDs from reference URLs
            url = ref.get("url", "")
            for cve in re.findall(r"CVE-\d{4}-\d+", url, re.IGNORECASE):
                cve_refs.append(cve.upper())

        if not tech_id or not re.match(r"T\d{4}", tech_id):
            continue

        # Mine description for CVE IDs
        description = obj.get("description", "")
        for cve in re.findall(r"CVE-\d{4}-\d+", description, re.IGNORECASE):
            cve_refs.append(cve.upper())

        cve_refs = list(set(cve_refs))

        tactics = [
            phase["phase_name"].replace("-", " ").title()
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]
        cwe_ids = list(set(re.findall(r"CWE-\d+", description, re.IGNORECASE)))

        techniques.append({
            "technique_id":    tech_id,
            "technique_name":  obj.get("name", ""),
            "tactic":          ", ".join(tactics),
            "description":     description[:2000],
            "cve_references":  cve_refs,
            "capec_ids":       capec_ids,
            "cwe_ids":         cwe_ids,
            "platforms":       obj.get("x_mitre_platforms", []),
            "data_sources":    obj.get("x_mitre_data_sources", []),
            "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
            "mitigations":     technique_mitigations.get(obj["id"], []),
            "stix_id":         obj.get("id", ""),
            "source":          "mitre_attack",
        })

    print(f"  ✅ ATT&CK: {len(techniques)} techniques parsed")
    cve_linked = sum(1 for t in techniques if t["cve_references"])
    print(f"     {cve_linked} techniques have direct CVE references (via STIX mining)")
    return techniques


# ── Community ATT&CK → CVE Mapping — with robust fallbacks ───────────────────

def _try_fetch_mapping_url(url: str) -> dict[str, list[str]] | None:
    """
    Attempt to fetch and parse a community ATT&CK→CVE mapping JSON from `url`.
    Returns the mapping dict on success, None on any failure.
    The JSON may be a list of {technique_id, cve_ids} objects or a dict with
    a 'mapping' key — both formats are handled.
    """
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        mapping: dict[str, list[str]] = {}
        items = data if isinstance(data, list) else data.get("mapping", data.get("objects", []))
        for entry in items:
            if not isinstance(entry, dict):
                continue
            tid = (
                entry.get("technique_id")
                or entry.get("attack_id")
                or entry.get("tid", "")
            )
            cves = (
                entry.get("cve_ids")
                or entry.get("cves")
                or entry.get("cve_list", [])
            )
            if tid and cves:
                mapping.setdefault(tid, []).extend(
                    c.upper() for c in cves if isinstance(c, str)
                )
        if mapping:
            return mapping
        return None
    except Exception:
        return None


def _build_nvd_cwe_technique_mapping(techniques: list[dict]) -> dict[str, list[str]]:
    """
    FALLBACK enrichment: load raw_nvd.json, match CVE→CWE, match CWE→technique.
    Returns {technique_id: [CVE-XXXX-YYYY, ...]} by CWE overlap.
    This gives a coarse but non-zero mapping even when community URLs are all 404.
    """
    if not NVD_PATH.exists():
        return {}

    print("     Building NVD→CWE→ATT&CK cross-reference as fallback enrichment...")
    try:
        with open(NVD_PATH, encoding="utf-8") as f:
            nvd_records = json.load(f)
    except Exception as e:
        print(f"     ⚠️  Could not read NVD: {e}")
        return {}

    # Build CWE → technique mapping from already-parsed techniques
    cwe_to_techniques: dict[str, list[str]] = {}
    for t in techniques:
        for cwe in t.get("cwe_ids", []):
            cwe_to_techniques.setdefault(cwe, []).append(t["technique_id"])

    mapping: dict[str, list[str]] = {}
    matched = 0
    for rec in nvd_records:
        cwe = rec.get("cwe_id", "")
        cve = rec.get("cve_id", "").upper()
        if not cwe or not cve:
            continue
        for tid in cwe_to_techniques.get(cwe, []):
            mapping.setdefault(tid, []).append(cve)
            matched += 1

    # Deduplicate
    mapping = {tid: list(set(cves)) for tid, cves in mapping.items()}
    print(f"     NVD fallback: {len(mapping)} techniques enriched, {matched} CVE links added")
    return mapping


def fetch_attack_cve_mapping(techniques: list[dict] | None = None) -> dict[str, list[str]]:
    """
    Fetch the ATT&CK → CVE community mapping.

    Strategy (in order):
      1. Try each URL in ATTACK_CVE_MAPPING_URLS
      2. If all 404 / fail → build from NVD CWE cross-reference (fallback)

    Returns {technique_id: [CVE-XXXX-YYYY, ...]}
    """
    print("  Fetching ATT&CK → CVE community mapping...")

    for url in ATTACK_CVE_MAPPING_URLS:
        short = url.split("githubusercontent.com/")[-1][:70]
        print(f"     Trying: {short}")
        result = _try_fetch_mapping_url(url)
        if result:
            print(f"     ✅ Loaded {len(result)} technique→CVE mappings from community overlay")
            return result
        else:
            print(f"     ✗ 404 or parse error — trying next URL")

    # All community URLs failed — use NVD cross-reference fallback
    print("     ⚠️  All community mapping URLs failed (repo may have moved).")
    print("     Falling back to NVD CWE→ATT&CK technique cross-reference...")
    if techniques:
        return _build_nvd_cwe_technique_mapping(techniques)
    print("     No techniques provided for fallback — returning empty mapping.")
    return {}


# ── CAPEC XML Parser ──────────────────────────────────────────────────────────

def fetch_capec() -> list[dict]:
    """
    Download and parse MITRE CAPEC XML.
    """
    print("Fetching MITRE CAPEC XML...")
    try:
        resp = requests.get(CAPEC_XML_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  CAPEC fetch failed: {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  ⚠️  CAPEC XML parse error: {e}")
        return []

    # Detect namespace from root tag
    root_tag = root.tag
    ns_uri   = root_tag.split("}")[0].lstrip("{") if "{" in root_tag else ""
    ns       = {"capec": ns_uri} if ns_uri else {}

    patterns = []
    for elem in root.iter():
        if not elem.tag.endswith("Attack_Pattern"):
            continue

        capec_id = elem.get("ID", "")
        name     = elem.get("Name", "")
        status   = elem.get("Status", "")
        if status in ("Deprecated", "Obsolete"):
            continue
        if not capec_id:
            continue

        # Description
        desc_elem = elem.find(".//{*}Description")
        description = (desc_elem.text or "") if desc_elem is not None else ""

        # CWE relationships
        cwe_ids = []
        for rel in elem.findall(".//{*}Related_Weakness"):
            cwe_id = rel.get("CWE_ID", "")
            if cwe_id:
                cwe_ids.append(f"CWE-{cwe_id}")

        # Related CAPEC
        related_capec = []
        for rel in elem.findall(".//{*}Related_Attack_Pattern"):
            rid = rel.get("CAPEC_ID", "")
            if rid:
                related_capec.append(f"CAPEC-{rid}")

        # Severity / Likelihood
        severity   = elem.get("Typical_Severity", "")
        likelihood = elem.get("Likelihood_Of_Attack", "")

        patterns.append({
            "capec_id":      f"CAPEC-{capec_id}",
            "name":          name,
            "description":   description[:1500],
            "cwe_ids":       cwe_ids,
            "related_capec": related_capec,
            "severity":      severity,
            "likelihood":    likelihood,
            "source":        "capec",
        })

    print(f"  ✅ CAPEC: {len(patterns)} attack patterns parsed")
    return patterns


# ── Lookup builders ───────────────────────────────────────────────────────────

def build_cwe_to_capec(capec_records: list[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for p in capec_records:
        for cwe in p["cwe_ids"]:
            index.setdefault(cwe, []).append(p["capec_id"])
    return index


def build_cve_to_techniques(
    techniques: list[dict],
    extra_mapping: dict[str, list[str]]
) -> dict[str, list[str]]:
    """
    Build CVE → [ATT&CK technique IDs] lookup.
    Merges STIX-native CVE references and community/NVD mapping.
    """
    index: dict[str, list[str]] = {}

    for t in techniques:
        for cve in t["cve_references"]:
            index.setdefault(cve, []).append(t["technique_id"])

    for tech_id, cves in extra_mapping.items():
        for cve in cves:
            index.setdefault(cve.upper(), []).append(tech_id)

    return {cve: list(set(tids)) for cve, tids in index.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def run(out: str | None = None):
    # FIX: use Path for output to avoid Windows separator bugs
    out_path = Path(out) if out else Path("data") / "raw_mitre_attack.json"

    all_records = []

    # 1. ATT&CK STIX
    techniques = fetch_attack_stix()
    all_records.extend(techniques)
    time.sleep(2)

    # 2. Community ATT&CK → CVE mapping (with multi-URL fallback + NVD fallback)
    # FIX: pass techniques so NVD fallback can use CWE→technique mapping
    extra_cve_map = fetch_attack_cve_mapping(techniques=techniques)

    # Enrich technique records with community mapping CVEs
    for t in techniques:
        tid = t["technique_id"]
        if tid in extra_cve_map:
            merged = list(set(t["cve_references"] + extra_cve_map[tid]))
            t["cve_references"] = merged
    time.sleep(2)

    # 3. CAPEC
    capec_records = fetch_capec()
    all_records.extend(capec_records)

    # Build lookup indices
    cwe_to_capec      = build_cwe_to_capec(capec_records)
    cve_to_techniques = build_cve_to_techniques(techniques, extra_cve_map)

    output = {
        "techniques":        techniques,
        "capec_patterns":    capec_records,
        "cwe_to_capec":      cwe_to_capec,
        "cve_to_techniques": cve_to_techniques,
        "stats": {
            "technique_count":  len(techniques),
            "capec_count":      len(capec_records),
            "cve_linked_count": len(cve_to_techniques),
            "cwe_capec_pairs":  sum(len(v) for v in cwe_to_capec.values()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n📊 MITRE ATT&CK Summary:")
    print(f"  Techniques:            {len(techniques)}")
    print(f"  CAPEC patterns:        {len(capec_records)}")
    print(f"  CVEs linked to ATT&CK: {len(cve_to_techniques)}")
    print(f"  CWE→CAPEC pairs:       {sum(len(v) for v in cwe_to_capec.values())}")
    print(f"\n✅ Saved MITRE ATT&CK + CAPEC data → {out_path}")


if __name__ == "__main__":
    run()