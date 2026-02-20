"""
dataset_enrichment.py
---------------------
Correlation and vendor enrichment functions imported by build_dataset.py.
Keeps build_dataset.py clean — just import and call these functions.

Usage in build_dataset.py run():
    from data.dataset_enrichment import (
        load_corr_lookup,
        load_vendor_lookup,
        enrich_with_correlations,
        enrich_with_vendors,
        infer_security_control_missing,   # replaces the old stub
        correlation_pairs,
    )
"""

import json
from pathlib import Path
from collections import defaultdict


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_corr_lookup(path: str = "data/raw_correlations.json") -> dict:
    """Load CVE → correlation_record lookup."""
    p = Path(path)
    if not p.exists():
        print("  ℹ️  No correlations file. Run: python run_pipeline.py --correlate")
        return {}
    try:
        records = json.loads(p.read_text(encoding="utf-8"))
        return {r["cve_id"]: r for r in records if r.get("cve_id")}
    except Exception as e:
        print(f"  ⚠️  Failed to load correlations: {e}")
        return {}


def load_vendor_lookup(path: str = "data/raw_vendor_advisories.json") -> dict:
    """Load CVE → [vendor advisory records] lookup."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        lookup: dict[str, list] = defaultdict(list)
        for item in raw:
            for cve in item.get("cves_mentioned", []):
                lookup[cve.upper()].append(item)
        return dict(lookup)
    except Exception as e:
        print(f"  ⚠️  Failed to load vendor advisories: {e}")
        return {}


def load_mitre_data(path: str = "data/raw_mitre_attack.json") -> dict:
    """Load MITRE ATT&CK + CAPEC data."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Security control mapping (replaces the generic stub) ──────────────────────

_OWASP_CONTROLS = {
    "A01:2021-Broken Access Control":               "Implement RBAC, enforce object-level authorization, validate ownership before serving resources",
    "A02:2021-Cryptographic Failures":              "Use AES-256/TLS 1.2+, replace MD5/SHA1 with bcrypt/argon2, enforce HTTPS, avoid hardcoded keys",
    "A03:2021-Injection":                           "Use parameterized queries and prepared statements, apply allowlist input validation, avoid dynamic query construction",
    "A04:2021-Insecure Design":                     "Add rate limiting, remove verbose errors, implement threat modeling, use anti-automation controls",
    "A05:2021-Security Misconfiguration":           "Harden configurations, disable unnecessary features, apply security headers (CSP, HSTS, X-Frame-Options)",
    "A06:2021-Vulnerable and Outdated Components":  "Maintain SBOM, use automated dependency scanning (Snyk/Dependabot), apply patches promptly",
    "A07:2021-Identification and Authentication":   "Enforce MFA, implement brute-force protection, use secure session tokens, invalidate sessions on logout",
    "A08:2021-Software and Data Integrity":         "Verify supply chain integrity (SRI, Sigstore), implement CI/CD security, validate deserialized inputs",
    "A09:2021-Security Logging and Monitoring":     "Implement SIEM, alert on anomalies, ensure logs are tamper-evident and include authentication events",
    "A10:2021-Server-Side Request Forgery":         "Allowlist permitted URLs, block internal IP ranges, disable unnecessary URL-fetching functionality",
}

_CWE_CONTROLS = {
    "CWE-79":  "Implement context-aware output encoding (HTML/JS/CSS), enforce Content Security Policy (CSP)",
    "CWE-89":  "Use parameterized queries or ORM — never concatenate user input into SQL statements",
    "CWE-78":  "Use subprocess with argument lists (not shell=True), validate and sanitize all command inputs",
    "CWE-22":  "Canonicalize paths before validation, reject traversal sequences (../), use chroot jails",
    "CWE-94":  "Disable eval()/exec() on user-supplied data, use sandboxed execution environments",
    "CWE-502": "Avoid deserializing untrusted data; if necessary, use safe libraries and validate before deserialization",
    "CWE-287": "Enforce MFA, implement account lockout and brute-force throttling, use secure session management",
    "CWE-798": "Remove all hardcoded credentials, use secrets management (Vault, AWS Secrets Manager, env vars)",
    "CWE-476": "Initialize all pointers, use memory-safe languages, add null-pointer checks before dereference",
    "CWE-190": "Validate integer ranges, use safe arithmetic with overflow detection, check bounds before operations",
    "CWE-416": "Use memory-safe languages; audit manual memory management; deploy AddressSanitizer in testing",
    "CWE-125": "Enable compiler bounds checking, validate input lengths, use safe buffer access patterns",
    "CWE-434": "Validate file type by content (magic bytes), store uploads outside webroot, restrict execute permissions",
    "CWE-611": "Disable XML external entity (XXE) processing in parser configuration",
    "CWE-918": "Allowlist permitted outbound URLs/IPs, enforce egress firewall rules to block SSRF",
    "CWE-352": "Implement anti-CSRF tokens (Synchronizer Token Pattern), use SameSite cookie attribute",
    "CWE-601": "Validate redirect targets against an allowlist of permitted destinations",
    "CWE-400": "Implement resource quotas, rate limiting, and input size validation to prevent resource exhaustion",
    "CWE-770": "Set explicit limits on resource allocation; add request timeouts and concurrency caps",
    "CWE-862": "Add explicit authorization checks at every sensitive function, not just at the route level",
    "CWE-863": "Implement fine-grained permission checks based on user roles and data ownership",
}


def infer_security_control_missing(owasp_category: str, cwe_id: str = "") -> str:
    """
    Return a specific, actionable missing security control.
    Replaces the old stub that returned 'Security control review required' for everything.
    """
    if cwe_id and cwe_id in _CWE_CONTROLS:
        return _CWE_CONTROLS[cwe_id]
    return _OWASP_CONTROLS.get(owasp_category, "Apply vendor patches, enforce least privilege, validate all inputs")


# ── Record enrichment ──────────────────────────────────────────────────────────

def enrich_with_correlations(record: dict, corr_lookup: dict) -> dict:
    """
    Populate related_vulnerabilities, attack_techniques, capec_patterns
    from the correlation graph. Fixes the always-empty related_vulnerabilities.
    """
    cve_id = record.get("cve_id", "")
    corr   = corr_lookup.get(cve_id, {})
    if corr:
        record["related_vulnerabilities"] = corr.get("related_vulnerabilities", [])
        record["attack_techniques"]       = corr.get("attack_techniques", [])
        record["capec_patterns"]          = corr.get("capec_patterns", [])
        raw_count = corr.get("correlation_signal_count", 0)
        signal_breakdown = corr.get("signal_type_counts", {})
        if signal_breakdown:
            record["correlation_signals"] = signal_breakdown
        else:
            record["correlation_signals"] = raw_count
    return record


def enrich_with_vendors(record: dict, vendor_lookup: dict) -> dict:
    """
    Add vendor-specific advisory context (Red Hat, Ubuntu, Debian, Cisco, PoC repos).
    """
    cve_id     = record.get("cve_id", "")
    advisories = vendor_lookup.get(cve_id, [])
    if not advisories:
        return record

    parts           = []
    affected_distros = set()

    for adv in advisories[:5]:
        source = adv.get("source", "")
        if source == "redhat_security":
            sev  = adv.get("severity", "")
            pkgs = adv.get("affected_packages", [])
            parts.append(f"Red Hat: severity={sev}, packages={', '.join(pkgs[:3])}")
            affected_distros.add("RHEL")
        elif source == "ubuntu_usn":
            pkgs = adv.get("affected_packages", [])
            if pkgs:
                parts.append(f"Ubuntu USN: {', '.join(pkgs[:3])}")
            affected_distros.add("Ubuntu")
        elif source == "debian_security":
            pkg   = adv.get("package", "")
            fixed = adv.get("releases_fixed", {})
            fixed_str = ", ".join(f"{r}:{v}" for r, v in list(fixed.items())[:2])
            parts.append(f"Debian package={pkg}, fixed_in={fixed_str or 'pending'}")
            affected_distros.add("Debian")
        elif source == "cisco_psirt":
            adv_id = adv.get("advisory_id", "")
            wk     = adv.get("workarounds", "")[:100]
            parts.append(f"Cisco {adv_id}: {wk or 'see advisory'}")
            affected_distros.add("Cisco")
        elif source == "poc_github":
            repo  = adv.get("repo", "")
            stars = adv.get("stars", 0)
            lang  = adv.get("language", "")
            parts.append(f"Public PoC exploit: {repo} ({stars}⭐, {lang})")

    if parts:
        vendor_block = "Vendor Advisory Context:\n" + "\n".join(f"  • {p}" for p in parts)
        existing = record.get("real_world_exploit", "")
        record["real_world_exploit"] = (existing + "\n\n" + vendor_block).strip() if existing else vendor_block

    if affected_distros:
        sw = list(set(record.get("affected_software", []) + list(affected_distros)))
        record["affected_software"] = sw[:20]

    return record


# ── Correlation training pairs ─────────────────────────────────────────────────

_mitre_data_cache = None

def _get_mitre_data() -> dict:
    global _mitre_data_cache
    if _mitre_data_cache is None:
        _mitre_data_cache = load_mitre_data()
    return _mitre_data_cache


def correlation_pairs(record: dict) -> list[dict]:
    """
    Generate vulnerability_correlation training pairs for a CVE record.
    Returns [] if no correlation data available.
    """
    cve_id  = record.get("cve_id", "")
    related = record.get("related_vulnerabilities", [])

    if not related or not cve_id:
        return []

    desc       = record.get("description", "")[:500]
    techniques = record.get("attack_techniques", [])
    capecs     = record.get("capec_patterns", [])
    mitre_data = _get_mitre_data()

    # Build technique name lookup
    tech_names = {
        t["technique_id"]: t["technique_name"]
        for t in mitre_data.get("techniques", [])
    }

    pairs = []

    # ── Pair 1: General correlation question ──────────────────────────────
    related_lines = []
    for r in related[:5]:
        sig_types = list({s.split(":")[0] for s in r.get("signals", [])})
        related_lines.append(
            f"  • {r['cve_id']} (score:{r.get('correlation_score',0)}, via: {', '.join(sig_types)})"
        )

    pairs.append({
        "instruction": f"What vulnerabilities are correlated with {cve_id} and why?",
        "input":       desc,
        "output": (
            f"Correlated vulnerabilities for {cve_id}:\n\n"
            + "\n".join(related_lines)
            + "\n\nCorrelation signal types:\n"
            "  • shared_cwe: Same underlying weakness class\n"
            "  • shared_product: Same affected software/vendor\n"
            "  • shared_attack_technique: Same MITRE ATT&CK technique\n"
            "  • kev_campaign_temporal: Co-listed in CISA KEV within 30 days (active campaign)\n"
            "  • exploit_chain_cooccurrence: Appear together in known exploit code"
        ),
        "layer": "vulnerability_correlation",
        "agent": "Correlation Agent",
    })

    # ── Pair 2: ATT&CK technique question ─────────────────────────────────
    if techniques:
        tech_lines = [
            f"  • {tid}: {tech_names.get(tid, 'Unknown')}"
            for tid in techniques[:4]
        ]
        att_related = [
            r["cve_id"] for r in related[:5]
            if any("attack_technique" in s for s in r.get("signals", []))
        ]
        pairs.append({
            "instruction": f"Which MITRE ATT&CK techniques are linked to {cve_id}, and what other CVEs use the same techniques?",
            "input":       desc,
            "output": (
                f"ATT&CK techniques for {cve_id}:\n"
                + "\n".join(tech_lines)
                + ("\n\nOther CVEs exploited via the same techniques:\n"
                   + "\n".join(f"  • {c}" for c in att_related)
                   if att_related else "")
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    # ── Pair 3: KEV campaign cluster ──────────────────────────────────────
    kev_cluster = [
        r["cve_id"] for r in related
        if any("kev_campaign" in s for s in r.get("signals", []))
    ]
    if kev_cluster:
        pairs.append({
            "instruction": f"Is {cve_id} part of a known active exploitation campaign? What other CVEs are clustered with it?",
            "input":       desc,
            "output": (
                f"{cve_id} is part of an active exploitation cluster identified via CISA KEV temporal analysis.\n\n"
                f"CVEs added to CISA KEV within the same 30-day window (probable shared campaign):\n"
                + "\n".join(f"  • {c}" for c in kev_cluster[:6])
                + "\n\nTemporal clustering of KEV entries strongly suggests coordinated use by the same threat actor group or ransomware operation."
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    # ── Pair 4: Exploit chain ─────────────────────────────────────────────
    chain = [
        r["cve_id"] for r in related
        if any("exploit_chain" in s for s in r.get("signals", []))
    ]
    if chain:
        pairs.append({
            "instruction": f"What exploit chains involve {cve_id}? Which CVEs are typically chained with it?",
            "input":       desc,
            "output": (
                f"Exploit chain analysis for {cve_id}:\n\n"
                f"The following CVEs co-occur in exploit code or PoC repositories:\n"
                + "\n".join(f"  • {c}" for c in chain[:5])
                + "\n\nCo-occurrence in exploit code indicates potential multi-stage attack patterns "
                  "(e.g., initial access via one CVE, privilege escalation via another for full compromise)."
            ),
            "layer": "vulnerability_correlation",
            "agent": "Correlation Agent",
        })

    return pairs