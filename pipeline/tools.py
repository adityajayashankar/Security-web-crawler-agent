"""
tools.py
--------
Tool functions for the vulnerability agent pipeline.
Each tool corresponds to a specific dataset layer and agent role.

Tools are imported by agent.py and registered in the TOOLS dict.
"""

import requests
import json
from pipeline.model_loader import ask_model

# ── Layer 1: Vulnerability Intelligence ───────────────────────────────────

def tool_lookup_cve(cve_id: str) -> str:
    """
    Fetch live CVE description + CWE + CVSS from NVD.
    Used by: OWASP Mapper Agent, Correlation Agent
    """
    try:
        url  = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.strip()}"
        data = requests.get(url, timeout=15).json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"CVE {cve_id} not found in NVD."

        cve    = vulns[0]["cve"]
        desc   = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "No description.")
        cwes   = [w["description"][0]["value"] for w in cve.get("weaknesses", []) if w.get("description")]
        metrics = cve.get("metrics", {})
        cvss   = ""
        for key in ["cvssMetricV31", "cvssMetricV30"]:
            if key in metrics:
                cvss = str(metrics[key][0]["cvssData"].get("baseScore", ""))
                break

        return (
            f"CVE ID: {cve_id}\n"
            f"Description: {desc}\n"
            f"CWE: {', '.join(cwes) if cwes else 'Unknown'}\n"
            f"CVSS Score: {cvss if cvss else 'Not available'}"
        )
    except Exception as e:
        return f"NVD lookup failed: {e}"


def tool_map_owasp(description: str) -> str:
    """
    Map vulnerability description to OWASP Top 10 category using the model.
    Used by: OWASP Mapper Agent
    """
    return ask_model(
        instruction = "Identify the OWASP Top 10 category for the following vulnerability description.",
        context     = description,
        layer       = "vulnerability_intelligence"
    )


# ── Layer 2: Pentesting Intelligence ──────────────────────────────────────

def tool_get_pentest_method(vuln_description: str) -> str:
    """
    Returns attack method, payload examples, and detection signals.
    Used by: Tool Selector Agent, Execution Planner Agent
    """
    return ask_model(
        instruction = "Describe how to test for this vulnerability during a pentest. Include attack method, payload examples, and detection signals.",
        context     = vuln_description,
        layer       = "pentesting_intelligence"
    )


def tool_select_tool(owasp_category: str, tech_stack: str = "") -> str:
    """
    Recommend the best security tool for a given vulnerability type.
    Used by: Tool Selector Agent
    """
    ctx = f"OWASP Category: {owasp_category}"
    if tech_stack:
        ctx += f"\nTech Stack: {tech_stack}"
    return ask_model(
        instruction = "Which security testing tool should be used and why?",
        context     = ctx,
        layer       = "execution_context"
    )


# ── Layer 3: Risk & Scoring ────────────────────────────────────────────────

def tool_fetch_epss(cve_id: str) -> str:
    """
    Fetch live EPSS exploit probability score from FIRST API.
    Used by: Base Scorer Agent, Severity Adjuster Agent
    """
    try:
        url  = f"https://api.first.org/data/v1/epss?cve={cve_id.strip()}"
        data = requests.get(url, timeout=15).json()
        items = data.get("data", [])
        if not items:
            return f"No EPSS data for {cve_id}"
        epss  = items[0].get("epss", "N/A")
        pct   = items[0].get("percentile", "N/A")
        return (
            f"EPSS Score for {cve_id}: {epss}\n"
            f"Percentile: {pct} (higher = more likely to be exploited)"
        )
    except Exception as e:
        return f"EPSS lookup failed: {e}"


def tool_score_risk(cve_description: str, cvss: str = "", epss: str = "") -> str:
    """
    Generate full risk assessment using the model.
    Used by: Base Scorer Agent
    """
    ctx = cve_description
    if cvss:
        ctx += f"\nCVSS Score: {cvss}"
    if epss:
        ctx += f"\nEPSS Score: {epss}"
    return ask_model(
        instruction = "Perform a risk assessment. Include risk level, business impact, and whether this should be treated as priority.",
        context     = ctx,
        layer       = "risk_scoring"
    )


# ── Layer 4: Audit Evidence ────────────────────────────────────────────────

def tool_generate_finding(
    vuln_name:   str,
    cve_id:      str,
    description: str,
    cvss:        str,
    owasp:       str
) -> str:
    """
    Generate a structured audit finding in report-ready format.
    Used by: Reporting Agent, Result Aggregator Agent
    """
    ctx = (
        f"Vulnerability Name: {vuln_name}\n"
        f"CVE ID: {cve_id}\n"
        f"Description: {description}\n"
        f"CVSS Score: {cvss}\n"
        f"OWASP Category: {owasp}"
    )
    return ask_model(
        instruction = "Generate a formal audit finding summary including severity, evidence, and affected controls.",
        context     = ctx,
        layer       = "audit_evidence"
    )


# ── Layer 5 & 6: Remediation Learning ─────────────────────────────────────

def tool_get_remediation(vuln_description: str) -> str:
    """
    Generate remediation advice and root cause analysis.
    Used by: Reflector Agent, Learning Agent
    """
    return ask_model(
        instruction = "What is the recommended remediation for this vulnerability? Include root cause and prevention.",
        context     = vuln_description,
        layer       = "remediation_learning"
    )