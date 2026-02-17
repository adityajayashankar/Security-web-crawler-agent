"""
build_dataset.py
----------------
Merges all raw source files into the full 6-layer schema and generates
instruction-response training pairs for each layer.

Layers built:
  1. Vulnerability Intelligence  (OWASP Mapper + Correlation agents)
  2. Pentesting Intelligence      (Tool Selector + Scanner agents)
  3. Risk & Scoring               (Base Scorer + Severity Adjuster agents)
  4. Execution Context            (Tech Stack Filter + Spawn Decision agents)
  5. Audit Evidence               (Result Aggregator + Reporting agents)
  6. Remediation Learning         (Reflector + Memory agents)

Output:
  data/vuln_dataset.jsonl         — full schema records (one per line)
  data/training_pairs.jsonl       — instruction-response pairs for fine-tuning
"""

import json
import re
import uuid
from pathlib import Path
from owasp_mapper import get_owasp_category, get_pentest_intel

# ── Helpers ────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def risk_level(cvss_score) -> str:
    if not cvss_score:
        return "Unknown"
    try:
        s = float(cvss_score)
        if s >= 9.0: return "Critical"
        if s >= 7.0: return "High"
        if s >= 4.0: return "Medium"
        return "Low"
    except ValueError:
        return "Unknown"

def business_impact(owasp_cat: str) -> str:
    impacts = {
        "A01:2021-Broken Access Control":           "Unauthorized data access, privilege escalation",
        "A02:2021-Cryptographic Failures":          "Sensitive data exposure, credential theft",
        "A03:2021-Injection":                       "Database compromise, remote code execution",
        "A04:2021-Insecure Design":                 "Systematic security bypass, reputational damage",
        "A05:2021-Security Misconfiguration":       "System compromise via exposed attack surface",
        "A06:2021-Vulnerable and Outdated Components": "Full system takeover via known exploits",
        "A07:2021-Identification and Authentication Failures": "Account takeover, session hijacking",
        "A08:2021-Software and Data Integrity Failures": "Supply chain compromise, malicious updates",
        "A09:2021-Security Logging and Monitoring Failures": "Undetected breaches, delayed incident response",
        "A10:2021-Server-Side Request Forgery":     "Internal network access, cloud metadata theft",
    }
    return impacts.get(owasp_cat, "Security breach, data loss")

def infer_exploitability(desc: str) -> str:
    d = desc.lower()
    if any(w in d for w in ["remote", "network", "unauthenticated", "internet"]):
        return "Remotely exploitable — no authentication required based on description."
    if any(w in d for w in ["local", "physical", "adjacent"]):
        return "Requires local or adjacent network access to exploit."
    if any(w in d for w in ["authenticated", "requires login", "privilege"]):
        return "Requires authenticated access or specific privileges to exploit."
    return "Exploitability context unclear — manual review recommended."

def infer_security_control_missing(owasp_cat: str) -> str:
    controls = {
        "A03:2021-Injection":                       "Input validation and parameterized queries",
        "A02:2021-Cryptographic Failures":          "Strong encryption and secure key management",
        "A01:2021-Broken Access Control":           "Authorization checks and role-based access control",
        "A07:2021-Identification and Authentication Failures": "MFA and strong session management",
        "A05:2021-Security Misconfiguration":       "Secure configuration baseline and hardening",
        "A06:2021-Vulnerable and Outdated Components": "Dependency scanning and patch management",
        "A10:2021-Server-Side Request Forgery":     "URL allowlist validation and network segmentation",
    }
    return controls.get(owasp_cat, "Security control review required")

# ── Load raw sources ───────────────────────────────────────────────────────

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return []
    with open(p) as f:
        return json.load(f)

def build_epss_lookup(epss_path: str) -> dict:
    raw = load_json(epss_path)
    if isinstance(raw, dict):
        return raw           # already {cve_id: score}
    return {}

def build_github_lookup(github_path: str) -> dict:
    """Returns {cve_id: advisory_record}"""
    raw = load_json(github_path)
    lookup = {}
    for item in raw:
        cve = item.get("cve_id", "")
        if cve:
            lookup[cve] = item
    return lookup

# ── Build full schema record ───────────────────────────────────────────────

def build_record(nvd_rec: dict, epss_map: dict, github_map: dict) -> dict:
    cve_id   = nvd_rec.get("cve_id", "")
    cwe_id   = nvd_rec.get("cwe_id", "")
    desc     = clean(nvd_rec.get("description", ""))
    cvss     = nvd_rec.get("cvss_score", "")
    sev      = nvd_rec.get("cvss_severity", "")

    owasp_cat   = get_owasp_category(cwe_id)
    pentest     = get_pentest_intel(owasp_cat)
    epss_score  = epss_map.get(cve_id, "")
    gh_advisory = github_map.get(cve_id, {})

    fix_rec = gh_advisory.get("fix_recommendation", "")
    if not fix_rec:
        fix_rec = "Apply vendor-supplied patches. Implement input validation and follow secure coding practices."

    return {
        # ── IDs ──────────────────────────────────────────
        "id":                    f"VULN_{str(uuid.uuid4())[:8].upper()}",

        # ── Layer 1: Vulnerability Intelligence ──────────
        "vulnerability_name":    nvd_rec.get("vulnerability_name", cve_id),
        "cve_id":                cve_id,
        "cwe_id":                cwe_id,
        "owasp_category":        owasp_cat,
        "description":           desc,
        "root_cause":            infer_security_control_missing(owasp_cat),

        # ── Layer 2: Pentesting Intelligence ─────────────
        "attack_method":         pentest.get("attack_method", ""),
        "payload_example":       pentest.get("payload_example", ""),
        "detection_signals":     pentest.get("detection_signals", []),
        "code_pattern":          pentest.get("code_pattern", ""),

        # ── Layer 3: Risk & Scoring ───────────────────────
        "cvss_score":            cvss,
        "cvss_severity":         sev,
        "epss_score":            epss_score,
        "risk_level":            risk_level(cvss),
        "business_impact":       business_impact(owasp_cat),

        # ── Layer 4: Execution Context ────────────────────
        "asset_type":            "Web Application",          # enriched at scan-time
        "environment":           "Unknown",                  # enriched at scan-time
        "internet_facing":       True,                       # conservative default
        "tech_stack":            {
            "language":  "",
            "framework": "",
            "database":  ""
        },

        # ── Layer 5: Audit Evidence ───────────────────────
        "tool_used":             pentest.get("tool_used", "Manual review"),
        "evidence_type":         "vulnerability_research",
        "evidence_summary":      f"Identified via CVE database. CVSS: {cvss}. {desc[:120]}...",
        "security_control_missing": infer_security_control_missing(owasp_cat),
        "control_type":          "Technical",

        # ── Layer 6: Remediation Learning ────────────────
        "fix_recommendation":    fix_rec,
        "status":                "Open",
        "related_vulnerabilities": [],

        # ── Source tracking ───────────────────────────────
        "source":                "NVD + OWASP + FIRST EPSS" + (
            " + GitHub Advisories" if gh_advisory else ""
        ),
    }

# ── Generate training pairs ────────────────────────────────────────────────

def to_training_pairs(record: dict) -> list[dict]:
    """
    Generate instruction-response pairs covering all 6 dataset layers.
    Each pair maps to a specific agent's expected use case.
    """
    cve    = record["cve_id"]
    desc   = record["description"]
    owasp  = record["owasp_category"]
    cvss   = record["cvss_score"]
    risk   = record["risk_level"]
    sev    = record["cvss_severity"]
    epss   = record["epss_score"]
    fix    = record["fix_recommendation"]
    method = record["attack_method"]
    sigs   = ", ".join(record["detection_signals"])
    biz    = record["business_impact"]
    ctrl   = record["security_control_missing"]
    tool   = record["tool_used"]
    cwe    = record["cwe_id"]

    pairs = []

    # ── L1: Vulnerability Intelligence (OWASP Mapper, Correlation agents) ─
    if desc:
        pairs.append({
            "instruction": f"Explain the vulnerability {cve} and map it to its OWASP category.",
            "input":       "",
            "output":      f"{desc}\n\nOWASP Category: {owasp}\nCWE: {cwe}",
            "layer":       "vulnerability_intelligence",
            "agent":       "OWASP Mapper Agent"
        })

    if owasp != "Unknown" and desc:
        pairs.append({
            "instruction": "Identify the OWASP Top 10 category for the following vulnerability description.",
            "input":       desc,
            "output":      f"This vulnerability maps to {owasp}.\nCWE classification: {cwe}.",
            "layer":       "vulnerability_intelligence",
            "agent":       "OWASP Mapper Agent"
        })

    # ── L2: Pentesting Intelligence (Tool Selector, Scanner agents) ────────
    if method:
        pairs.append({
            "instruction": "Describe how to test for this vulnerability during a pentest.",
            "input":       desc,
            "output":      (
                f"Attack Method: {method}\n\n"
                f"Detection Signals: {sigs}\n\n"
                f"Recommended Tool: {tool}"
            ),
            "layer":       "pentesting_intelligence",
            "agent":       "Tool Selector Agent"
        })

    if sigs:
        pairs.append({
            "instruction": "What code patterns or signals indicate this vulnerability is present?",
            "input":       desc,
            "output":      f"Detection signals to look for:\n- " + "\n- ".join(record["detection_signals"]),
            "layer":       "pentesting_intelligence",
            "agent":       "Scanner Agent"
        })

    # ── L3: Risk & Scoring (Base Scorer, Severity Adjuster agents) ─────────
    if cvss:
        pairs.append({
            "instruction": "Perform a risk assessment for this vulnerability.",
            "input":       desc,
            "output":      (
                f"CVSS Score: {cvss} ({sev})\n"
                f"Risk Level: {risk}\n"
                f"EPSS Score: {epss if epss else 'Not available'}\n"
                f"Business Impact: {biz}"
            ),
            "layer":       "risk_scoring",
            "agent":       "Base Scorer Agent"
        })

    if cvss:
        pairs.append({
            "instruction": "Should this vulnerability be treated as a priority? Explain based on risk scoring.",
            "input":       f"Vulnerability: {desc}\nCVSS: {cvss}\nEPSS: {epss}",
            "output":      (
                f"Priority: {'YES — immediate action required' if risk in ['Critical','High'] else 'Moderate — schedule remediation'}.\n"
                f"CVSS base score {cvss} classifies this as {sev} severity. "
                f"{'EPSS score of ' + str(epss) + ' indicates high exploit probability in the wild.' if epss else ''} "
                f"Business impact: {biz}."
            ),
            "layer":       "risk_scoring",
            "agent":       "Severity Adjuster Agent"
        })

    # ── L4: Execution Context (Tech Stack Filter, Spawn Decision agents) ───
    if owasp != "Unknown":
        pairs.append({
            "instruction": "Which security tool should be used to test this vulnerability, and why?",
            "input":       f"Vulnerability type: {owasp}\nDescription: {desc}",
            "output":      (
                f"Recommended tool: {tool}\n"
                f"Reason: This is a {owasp} class vulnerability. "
                f"The attack method involves: {method}"
            ),
            "layer":       "execution_context",
            "agent":       "Tool Selector Agent"
        })

    # ── L5: Audit Evidence (Result Aggregator, Reporting agents) ───────────
    if cvss:
        pairs.append({
            "instruction": "Generate an audit finding summary for this vulnerability.",
            "input":       desc,
            "output":      (
                f"Finding: {record['vulnerability_name']}\n"
                f"CVE: {cve} | CWE: {cwe} | OWASP: {owasp}\n"
                f"Severity: {sev} (CVSS {cvss})\n"
                f"Security Control Missing: {ctrl}\n"
                f"Evidence: Confirmed via vulnerability research and CVE database.\n"
                f"Tool: {tool}"
            ),
            "layer":       "audit_evidence",
            "agent":       "Reporting Agent"
        })

    # ── L6: Remediation Learning (Reflector, Memory agents) ───────────────
    if fix:
        pairs.append({
            "instruction": "What is the recommended remediation for this vulnerability?",
            "input":       desc,
            "output":      (
                f"Remediation: {fix}\n\n"
                f"Root Cause: {ctrl}\n"
                f"Control Type: Technical"
            ),
            "layer":       "remediation_learning",
            "agent":       "Reflector Agent"
        })

    if desc and fix:
        pairs.append({
            "instruction": "Explain the root cause of this vulnerability and how to prevent it.",
            "input":       desc,
            "output":      (
                f"Root Cause: {ctrl} is missing.\n"
                f"Prevention: {fix}\n"
                f"OWASP Category: {owasp}"
            ),
            "layer":       "remediation_learning",
            "agent":       "Learning Agent"
        })

    return pairs

# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("Loading raw data sources...")
    nvd_records = load_json("data/raw_nvd.json")
    epss_map    = build_epss_lookup("data/raw_epss.json")
    github_map  = build_github_lookup("data/raw_github.json")

    print(f"  NVD records:   {len(nvd_records)}")
    print(f"  EPSS entries:  {len(epss_map)}")
    print(f"  GitHub entries:{len(github_map)}")

    # ── Build full records ─────────────────────────────────────────────────
    seen_cves = set()
    full_records  = []
    training_pairs = []

    for nvd_rec in nvd_records:
        cve_id = nvd_rec.get("cve_id", "")
        desc   = nvd_rec.get("description", "")

        # Skip empty or duplicate
        if not desc or len(desc) < 50:
            continue
        if cve_id in seen_cves:
            continue
        seen_cves.add(cve_id)

        record = build_record(nvd_rec, epss_map, github_map)
        full_records.append(record)
        training_pairs.extend(to_training_pairs(record))

    # ── Save full schema records ───────────────────────────────────────────
    with open("data/vuln_dataset.jsonl", "w") as f:
        for r in full_records:
            f.write(json.dumps(r) + "\n")

    # ── Save training pairs ────────────────────────────────────────────────
    with open("data/training_pairs.jsonl", "w") as f:
        for p in training_pairs:
            f.write(json.dumps(p) + "\n")

    # ── Stats ──────────────────────────────────────────────────────────────
    layer_counts = {}
    for p in training_pairs:
        l = p.get("layer", "unknown")
        layer_counts[l] = layer_counts.get(l, 0) + 1

    print(f"\n✅ Full schema records:  {len(full_records)} → data/vuln_dataset.jsonl")
    print(f"✅ Training pairs total: {len(training_pairs)} → data/training_pairs.jsonl")
    print("\nTraining pairs per layer:")
    for layer, count in layer_counts.items():
        print(f"  {layer:<30} {count:>6} examples")

if __name__ == "__main__":
    run()