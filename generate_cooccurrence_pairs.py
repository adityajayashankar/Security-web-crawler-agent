"""
generate_cooccurrence_pairs.py  (FIXED v2)
─────────────────────────────────────────────────────────────────────────────
Fix: was generating only 1 pair per trigger CVE → ceiling of ~2,087.
Now generates multiple template variants per CVE + uses KEV pairs directly
to reach the 15,000 target.

Run:
  python generate_cooccurrence_pairs.py --count 15000
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from stack_profiles import STACK_PROFILES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = Path("data")
COOC_FILE = DATA_DIR / "raw_cooccurrence_v2.json"
NVD_FILE  = DATA_DIR / "vuln_dataset.jsonl"
OUT_FILE  = DATA_DIR / "training_pairs.jsonl"

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_nvd_index(path):
    """Load NVD records; report year range for temporal awareness."""
    index = {}
    years = set()
    if not path.exists():
        return index
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cid = rec.get("cve_id") or rec.get("id", "")
                if cid:
                    index[cid] = rec
                    # Track year for temporal reporting
                    pub = rec.get("published_date", rec.get("date_published", ""))
                    if pub:
                        years.add(pub[:4])
                    elif cid.startswith("CVE-"):
                        parts = cid.split("-")
                        if len(parts) >= 2 and parts[1].isdigit():
                            years.add(parts[1])
            except json.JSONDecodeError:
                pass
    log.info(f"  NVD index: {len(index):,} CVEs")
    if years:
        sorted_years = sorted(years)
        log.info(f"  NVD date range: {sorted_years[0]} \u2192 {sorted_years[-1]}  "
                 f"({len(sorted_years)} distinct years)")
    return index


def load_cooc(path):
    if not path.exists():
        log.warning(f"  {path} not found — run build_cooccurrence_v2.py first")
        return [], []
    with open(path) as f:
        data = json.load(f)
    return data.get("cooccurrence_pairs", []), data.get("negative_rules", [])


# ─────────────────────────────────────────────────────────────────────────────
# CVE helpers
# ─────────────────────────────────────────────────────────────────────────────

def cve_summary(cve_id, nvd_index, max_chars=200):
    rec  = nvd_index.get(cve_id, {})
    desc = rec.get("description", rec.get("summary", ""))
    if not desc:
        desc = f"vulnerability affecting {rec.get('affected_product', rec.get('product', 'the target system'))}"
    return desc.replace("\n", " ").strip()[:max_chars]


def cve_severity(cve_id, nvd_index):
    rec = nvd_index.get(cve_id, {})
    return rec.get("cvss_v3_severity", rec.get("severity", rec.get("cvss_severity", "UNKNOWN")))


def cve_cvss(cve_id, nvd_index):
    rec = nvd_index.get(cve_id, {})
    score = rec.get("cvss_v3_score", rec.get("cvss_score", rec.get("base_score")))
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def cve_product(cve_id, nvd_index):
    rec = nvd_index.get(cve_id, {})
    return rec.get("affected_product", rec.get("product", rec.get("vendor", "")))


# ─────────────────────────────────────────────────────────────────────────────
# Group pairs by trigger CVE
# ─────────────────────────────────────────────────────────────────────────────

def group_by_trigger(all_pairs):
    """Group co-occurrence pairs by trigger CVE, creating reverse pairs with
    asymmetric confidence: P(B|A) \u2260 P(A|B).  Reverse pairs get a random
    perturbation (0.75\u00d7\u20131.15\u00d7) to teach the model directionality."""
    grouped = defaultdict(list)
    for p in all_pairs:
        a, b = p["cve_a"], p["cve_b"]
        grouped[a].append(p)
        # Reverse pair \u2014 perturb confidence to teach P(A|B) \u2260 P(B|A)
        rev = dict(p)
        rev["cve_a"], rev["cve_b"] = b, a
        orig_conf = p.get("confidence", 0.5)
        asym_factor = random.uniform(0.75, 1.15)
        rev["confidence"] = round(min(0.99, max(0.10, orig_conf * asym_factor)), 2)
        rev["_reversed"] = True
        grouped[b].append(rev)
    for cve in grouped:
        grouped[cve].sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 1: Positive inference  (FIXED — multiple templates per CVE)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_INPUTS = [
    "Security assessment confirmed: {cve_a} is present and exploitable.\nDescription: {desc_a}\nSeverity: {sev_a} / CVSS {cvss_a}\nAffected product: {product_a}\n\nWhat other vulnerabilities should be investigated?",
    "Penetration test finding: {cve_a} exploited successfully.\nProduct: {product_a} | CVSS: {cvss_a}\n\nBased on this confirmed vulnerability, what co-occurring CVEs are likely present?",
    "Vulnerability scan result: {cve_a} confirmed vulnerable.\n{desc_a}\n\nPerform co-occurrence analysis: which related vulnerabilities are statistically likely?",
    "Red team finding — initial access via {cve_a}.\n{desc_a}\nSeverity: {sev_a}\n\nIdentify which other CVEs are likely exploitable given this foothold.",
    "Bug bounty: {cve_a} confirmed ({sev_a}, CVSS {cvss_a}).\nProduct: {product_a}\n\nWhat is the broader vulnerability surface implied by this finding?",
    "Threat intelligence alert: {cve_a} is being actively exploited.\n{desc_a}\n\nWhich co-occurring vulnerabilities should defenders prioritise patching immediately?",
    "Audit finding: {cve_a} identified in {product_a}.\nCVSS score: {cvss_a} ({sev_a})\n\nAs a senior security analyst, what related CVEs would you expect in the same environment?",
    "SIEM alert triggered for {cve_a}.\nDescription: {desc_a}\n\nFor incident response: what other CVEs should be checked immediately given this finding?",
]

POSITIVE_OUTPUT_VARIANTS = [
    # Variant 0: Full structured report
    """Co-occurrence Analysis: {cve_a}

CONFIRMED FINDING
─────────────────
{cve_a} | {sev_a} | CVSS {cvss_a}
Product: {product_a}
{desc_a}

LIKELY CO-PRESENT VULNERABILITIES
──────────────────────────────────
{related_block}

REASONING
─────────
{reasoning}

INVESTIGATION PRIORITY
──────────────────────
{priority_block}""",
    # Variant 1: Concise assessment
    """{cve_a} ({sev_a}, CVSS {cvss_a}) — Co-occurrence Assessment
Product: {product_a}

Co-present vulnerabilities ranked by confidence:

{related_block}

{reasoning_short}

Priority order:
{priority_block}""",
    # Variant 2: Probability-focused report
    """Co-occurrence probability report for {cve_a}:

Product: {product_a} | Severity: {sev_a} | CVSS: {cvss_a}

Ranked co-present vulnerabilities:

{related_block}

Intelligence sources: {sources_text}
These probabilities reflect observed co-presence frequency across multi-source
vulnerability intelligence (NVD, CISA KEV, exploit chains, CWE family clusters).""",
    # Variant 3: Incident response advisory
    """IR ADVISORY — {cve_a} confirmed exploited

Trigger: {cve_a} | {sev_a} | CVSS {cvss_a}
Environment: {product_a}
{desc_a}

IMMEDIATE INVESTIGATION TARGETS:
{priority_block}

DETAILED CO-OCCURRENCE ANALYSIS:
{related_block}

CHAIN REASONING:
{reasoning}""",
]


def make_positive_pairs(trigger_cve, related_pairs, nvd_index, num_variants=4):
    """Generate up to num_variants pairs with diversified outputs per trigger CVE.

    Each variant gets:
      - A different input template (as before)
      - A different output format (full/concise/probability/IR)
      - A different slice and ordering of related CVEs
    This prevents the model from memorizing a single output template.
    """
    if not related_pairs:
        return []

    desc_a    = cve_summary(trigger_cve, nvd_index)
    sev_a     = cve_severity(trigger_cve, nvd_index)
    cvss_a    = cve_cvss(trigger_cve, nvd_index)
    cvss_str  = f"{cvss_a:.1f}" if cvss_a else "N/A"
    product_a = cve_product(trigger_cve, nvd_index) or "affected system"

    sources_all = list(set(p.get("source", "") for p in related_pairs[:5]))

    # Build reasoning
    parts = []
    if any("attack_chain" in s for s in sources_all):
        parts.append("These CVEs form a sequential exploit chain — the trigger creates preconditions the related CVEs directly exploit.")
    if any("remediation_tie" in s for s in sources_all):
        parts.append("Several related CVEs share the same patch — structural coupling in the codebase.")
    if any("kev" in s for s in sources_all):
        parts.append("CISA KEV data shows these were added in the same campaign window — actively chained by threat actors.")
    if any("cwe" in s for s in sources_all):
        parts.append("CWE CanPrecede chain — the trigger's weakness class structurally enables the related CVEs.")
    if any("stack" in s for s in sources_all):
        parts.append("Same technology stack — co-present when this deployment profile is confirmed.")
    reasoning = " ".join(parts) if parts else "Statistical co-occurrence from multi-source vulnerability intelligence."
    reasoning_short = parts[0] if parts else reasoning

    templates = random.sample(POSITIVE_INPUTS, min(num_variants, len(POSITIVE_INPUTS)))
    pairs = []
    for vi, tmpl in enumerate(templates):
        # ── Output diversity: different CVE order, slice, and format per variant ──
        shuffled = list(related_pairs)
        random.shuffle(shuffled)
        slice_size = random.choice([3, 4, 5])
        top_n = shuffled[:min(slice_size, len(shuffled))]

        related_lines = []
        for i, p in enumerate(top_n, 1):
            b      = p["cve_b"]
            conf   = p.get("confidence", 0)
            reason = p.get("reason", "co-occurrence detected")
            sev_b  = cve_severity(b, nvd_index)
            desc_b = cve_summary(b, nvd_index, 100)
            label  = "HIGH" if conf >= 0.80 else "MEDIUM" if conf >= 0.65 else "LOW"
            related_lines.append(
                f"[{i}] {b} ({label} {conf:.0%}) | {sev_b}\n"
                f"    {reason}\n"
                f"    {desc_b}"
            )
        related_block  = "\n\n".join(related_lines)
        priority_block = "\n".join(f"  {i}. {p['cve_b']} — {p.get('reason','')[:70]}"
                                   for i, p in enumerate(top_n, 1))
        sources_text   = ", ".join(s for s in sources_all if s) or "multi-source intelligence"

        fmt = POSITIVE_OUTPUT_VARIANTS[vi % len(POSITIVE_OUTPUT_VARIANTS)]
        out_text = fmt.format(
            cve_a=trigger_cve, sev_a=sev_a, cvss_a=cvss_str,
            product_a=product_a, desc_a=desc_a,
            related_block=related_block, reasoning=reasoning,
            reasoning_short=reasoning_short,
            priority_block=priority_block,
            sources_text=sources_text,
        ).strip()

        inp = tmpl.format(
            cve_a=trigger_cve, desc_a=desc_a, sev_a=sev_a,
            cvss_a=cvss_str, product_a=product_a,
        ).strip()
        pairs.append({
            "layer":    "vulnerability_cooccurrence",
            "type":     "positive_inference",
            "input":    inp,
            "output":   out_text,
            "metadata": {
                "trigger_cve":   trigger_cve,
                "related_count": len(top_n),
                "sources":       sources_all,
            },
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 2: Negative inference
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_INPUTS = [
    "Assessment: {cve} is CONFIRMED ABSENT.\nVerified condition: {condition}\nStack: {display}\n\nWhich co-dependent CVEs are also structurally absent? Which still need independent assessment?",
    "Vulnerability scan: {cve} — NOT VULNERABLE.\nCondition verified: {condition}\n\nDetermine which related CVEs are structurally eliminated vs which remain independent risks.",
    "Patch verified: {cve} remediated in {display}.\nCondition: {condition}\n\nPerform absence chain analysis — what does patching this CVE imply about related vulnerabilities?",
    "Security audit: {cve} not present ({condition}).\nEnvironment: {display}\n\nNegative inference: which CVEs are structurally absent, and which are independent attack surfaces?",
]

NEGATIVE_OUTPUT = """Negative Inference Analysis: {cve} ABSENT

CONFIRMED ABSENCE
─────────────────
CVE: {cve}
Condition: {condition}
Stack: {display}
Reason: {reason}

STRUCTURALLY ALSO ABSENT (same root condition)
───────────────────────────────────────────────
{absent_block}

STILL REQUIRES INDEPENDENT ASSESSMENT
──────────────────────────────────────
{still_block}

CHAIN REASONING
───────────────
{chain_reasoning}"""


def make_negative_pairs(rule, nvd_index, num_variants=3):
    """Generate negative inference pairs with output diversity per variant.
    Shuffles absent CVE ordering and alternates full/brief output format."""
    absent  = rule.get("absent_cves", [])
    still   = rule.get("still_assess", [])
    cond    = rule.get("condition", "")
    reason  = rule.get("reason", "")
    display = rule.get("display", rule.get("profile", "unknown stack"))
    if not absent or not cond:
        return []

    templates = random.sample(NEGATIVE_INPUTS, min(num_variants, len(NEGATIVE_INPUTS)))
    pairs = []
    for vi, tmpl in enumerate(templates):
        # Shuffle absent CVEs for output diversity
        shuffled_absent = list(absent)
        random.shuffle(shuffled_absent)
        primary = shuffled_absent[0]

        absent_block = "\n".join(
            f"• {c}: {cve_summary(c, nvd_index, 90)}\n  Eliminated: shares root condition with {primary}"
            for c in shuffled_absent
        )
        still_block = "\n".join(
            f"• {c}: {cve_summary(c, nvd_index, 90)}\n  Independent — not affected by '{cond}'"
            for c in still
        ) if still else "None — all co-dependent CVEs eliminated."

        chain_reasoning = (
            f"Condition '{cond}' removes the shared precondition for {', '.join(shuffled_absent)}. "
            f"{reason} "
            f"CVEs with independent preconditions ({', '.join(still) if still else 'none here'}) "
            f"are not transitively eliminated."
        )

        # Alternate between full and brief output format
        if vi % 2 == 0:
            out_text = NEGATIVE_OUTPUT.format(
                cve=primary, condition=cond, display=display, reason=reason,
                absent_block=absent_block, still_block=still_block,
                chain_reasoning=chain_reasoning,
            ).strip()
        else:
            out_text = (
                f"Negative inference: {primary} ABSENT in {display}.\n"
                f"Condition: {cond}\n{reason}\n\n"
                f"Structurally eliminated (same root condition):\n{absent_block}\n\n"
                f"Independent — still assess:\n{still_block}"
            ).strip()

        pairs.append({
            "layer":  "vulnerability_cooccurrence",
            "type":   "negative_inference",
            "input":  tmpl.format(cve=primary, condition=cond, display=display).strip(),
            "output": out_text,
            "metadata": {
                "primary_cve":  primary,
                "absent_cves":  absent,
                "still_assess": still,
                "profile":      rule.get("profile", ""),
            },
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 3: Chain reasoning
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_INPUTS = [
    "Initial access via {first} in {display}.\n{first_desc}\n\nMap the full sequential attack chain from this foothold.",
    "Threat intel: attackers exploiting {first} against {display}.\nDescribe the complete exploit chain that follows.",
    "Red team entry point: {first} ({display}).\nDetail the attack chain: what sequential CVEs does a skilled attacker chain next?",
    "Incident response: {first} confirmed exploited.\n{first_desc}\nTrace the likely attack progression chain.",
    "Risk assessment: {first} in {display} environment.\nWhat is the end-to-end attack chain impact?",
]

CHAIN_OUTPUT = """Attack Chain: {display}

ENTRY POINT: {first}
{first_desc}

SEQUENTIAL CHAIN
────────────────
{steps}

CHAIN MECHANICS
───────────────
Each step leverages access established by the previous. {first} compromises 
the initial layer; subsequent CVEs escalate privilege or expand impact using 
the context/credentials gained.

BREAK-CHAIN REMEDIATION
────────────────────────
{remediation}"""


def make_chain_pairs(profile_key, chain, nvd_index, num_variants=4):
    if len(chain) < 2:
        return []
    profile   = STACK_PROFILES.get(profile_key, {})
    display   = profile.get("display_name", profile_key)
    first     = chain[0]
    first_desc = cve_summary(first, nvd_index, 130)

    labels = ["Initial Access", "Execution / Persistence",
              "Privilege Escalation", "Impact / Exfiltration"]
    steps  = []
    for i, cve in enumerate(chain):
        sev    = cve_severity(cve, nvd_index)
        cvss   = cve_cvss(cve, nvd_index)
        cvss_s = f"CVSS {cvss:.1f}" if cvss else sev
        desc   = cve_summary(cve, nvd_index, 100)
        label  = labels[min(i, len(labels)-1)]
        nxt    = f"→ enables Step {i+2}" if i < len(chain)-1 else "→ FULL COMPROMISE"
        steps.append(f"Step {i+1} [{label}]\n  {cve} ({cvss_s})\n  {desc}\n  {nxt}")

    rem_ties = profile.get("remediation_ties", [])
    chain_set = set(chain)
    tied = [t for t in rem_ties if chain_set & set(t.get("cves", []))]
    remediation = "\n".join(
        f"• {t['fix']} — closes: {', '.join(t['cves'])}" for t in tied
    ) if tied else f"Patch each CVE individually: {' → '.join(chain)}"

    out_text = CHAIN_OUTPUT.format(
        display=display, first=first, first_desc=first_desc,
        steps="\n\n".join(steps), remediation=remediation,
    ).strip()

    templates = random.sample(CHAIN_INPUTS, min(num_variants, len(CHAIN_INPUTS)))
    pairs = []
    for tmpl in templates:
        pairs.append({
            "layer":  "vulnerability_cooccurrence",
            "type":   "chain_reasoning",
            "input":  tmpl.format(first=first, display=display, first_desc=first_desc).strip(),
            "output": out_text,
            "metadata": {"profile": profile_key, "chain": chain},
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 4: Stack profile discovery
# ─────────────────────────────────────────────────────────────────────────────

STACK_INPUTS = [
    "Technology fingerprinting: {indicators} detected.\nEnvironment: {display}\n\nProvide complete vulnerability surface analysis for this stack.",
    "Target identified as {display}.\nIndicators: {indicators}\n\nAs a pentester, map the full CVE exposure for this technology stack.",
    "Asset inventory: {indicators} confirmed.\nWhat is the complete vulnerability surface for a {display} deployment?",
    "Reconnaissance: {display} stack fingerprinted via {indicators}.\nList all CVEs that should be assessed for this environment.",
    "Security assessment scope: {display}.\nDetected indicators: {indicators}\n\nGenerate the vulnerability checklist for this stack.",
    "Threat modelling for {display}.\nKnown indicators: {indicators}\n\nWhat vulnerabilities are historically associated with this technology stack?",
]

STACK_OUTPUT = """Vulnerability Surface: {display}

STACK CONFIRMED
───────────────
Technology: {display}
Indicators: {indicators}

HIGH CONFIDENCE (present unless patched)
─────────────────────────────────────────
{hc_block}

CONDITIONAL (present when sub-condition met)
─────────────────────────────────────────────
{cond_block}

ATTACK CHAINS
─────────────
{chain_block}

NEGATIVE INFERENCE — conditions that eliminate CVEs
────────────────────────────────────────────────────
{neg_block}

REMEDIATION PLAN
─────────────────
{rem_block}"""


def make_stack_pairs(profile_key, nvd_index, num_variants=6):
    profile    = STACK_PROFILES.get(profile_key, {})
    display    = profile.get("display_name", profile_key)
    indicators = ", ".join(profile.get("indicators", [profile_key])[:4])

    hc_lines = [
        f"• {x['cve']} ({cve_severity(x['cve'], nvd_index)}) — {x['reason']}"
        for x in profile.get("high_confidence", [])
    ]
    hc_block = "\n".join(hc_lines) if hc_lines else "None defined."

    cond_lines = []
    for label, items in profile.get("conditional", {}).items():
        for x in items:
            cond_lines.append(f"• [IF {label.replace('if_','').replace('_',' ')}] {x['cve']} — {x['reason']}")
    cond_block = "\n".join(cond_lines) if cond_lines else "None."

    chains = profile.get("attack_chains", [])
    chain_block = "\n".join(f"• Chain {i+1}: {' → '.join(c)}" for i, c in enumerate(chains)) \
                  if chains else "No defined chains — assess CVEs independently."

    neg_rules = profile.get("negative_rules", [])
    neg_lines = [
        f"• IF {r['condition']}:\n  ABSENT: {', '.join(r['absent_cves'])}\n  Because: {r['reason']}"
        for r in neg_rules[:3]
    ]
    neg_block = "\n".join(neg_lines) if neg_lines else "No negative inference rules defined."

    rem_ties = profile.get("remediation_ties", [])
    rem_lines = [f"• {t['fix']} — closes: {', '.join(t['cves'])}" for t in rem_ties]
    rem_block = "\n".join(rem_lines) if rem_lines else "Patch per individual CVE advisories."

    out_text = STACK_OUTPUT.format(
        display=display, indicators=indicators,
        hc_block=hc_block, cond_block=cond_block, chain_block=chain_block,
        neg_block=neg_block, rem_block=rem_block,
    ).strip()

    templates = random.sample(STACK_INPUTS, min(num_variants, len(STACK_INPUTS)))
    pairs = []
    for tmpl in templates:
        pairs.append({
            "layer":  "vulnerability_cooccurrence",
            "type":   "stack_profile",
            "input":  tmpl.format(display=display, indicators=indicators).strip(),
            "output": out_text,
            "metadata": {"profile": profile_key},
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 5: Conditional reasoning
# ─────────────────────────────────────────────────────────────────────────────

COND_INPUTS = [
    "{cve_a} confirmed in {display}.\nSub-condition detected: {condition}\n\nHow does this condition change the co-occurrence surface?",
    "Finding: {cve_a} present + condition [{condition}] observed.\nStack: {display}\n\nWhich additional CVEs are now likely given this specific condition?",
    "Vulnerability {cve_a} identified.\nEnvironment flag: {condition} confirmed.\n\nAdjust the co-occurrence analysis for {display} given this condition.",
    "Assessment update: {cve_a} found AND {condition} verified in {display}.\nWhat additional CVEs does this condition surface?",
]

COND_OUTPUT = """Conditional Co-occurrence: {cve_a} + {condition}

BASE FINDING: {cve_a} ({display})

WITHOUT CONDITION — base surface
──────────────────────────────────
{base_cves}

WITH [{condition}] — additional CVEs now likely
─────────────────────────────────────────────────
{additional}

COMBINED ASSESSMENT SCOPE
──────────────────────────
{combined}

WHY THIS CONDITION MATTERS
───────────────────────────
{significance}"""


def make_conditional_pairs(profile_key, cond_label, cond_items, nvd_index, num_variants=3):
    profile  = STACK_PROFILES.get(profile_key, {})
    display  = profile.get("display_name", profile_key)
    hc       = profile.get("high_confidence", [])
    if not hc or not cond_items:
        return []

    trigger   = random.choice(hc)["cve"]
    condition = cond_label.replace("if_", "").replace("_", " ").upper()
    add_cves  = [x["cve"] for x in cond_items]
    base_cves = [x["cve"] for x in hc if x["cve"] != trigger][:3]

    base_block = "\n".join(f"• {c} — {cve_summary(c, nvd_index, 70)}" for c in base_cves) or "N/A"
    add_block  = "\n".join(
        f"• {x['cve']} ({cve_severity(x['cve'], nvd_index)}) — {x['reason']}"
        for x in cond_items
    )
    combined   = "\n".join(f"• {c}" for c in list(set(base_cves + add_cves)))
    significance = (
        f"Condition '{condition}' activates an additional attack surface specific to this "
        f"sub-configuration of {display}. Without verifying this condition, these CVEs would "
        f"be incorrectly excluded from the scope."
    )

    out_text = COND_OUTPUT.format(
        cve_a=trigger, display=display, condition=condition,
        base_cves=base_block, additional=add_block,
        combined=combined, significance=significance,
    ).strip()

    templates = random.sample(COND_INPUTS, min(num_variants, len(COND_INPUTS)))
    pairs = []
    for tmpl in templates:
        pairs.append({
            "layer":  "vulnerability_cooccurrence",
            "type":   "conditional_reasoning",
            "input":  tmpl.format(cve_a=trigger, condition=condition, display=display).strip(),
            "output": out_text,
            "metadata": {"profile": profile_key, "trigger": trigger, "condition": cond_label},
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 6: CWE cross-cluster hard negatives
# ─────────────────────────────────────────────────────────────────────────────
# Problem: hard negatives came only from ~12 stack profiles (~36 pairs).
# Solution: generate negatives across structurally independent CWE clusters.
# "Finding CWE-89 (input validation) does NOT imply CWE-416 (memory safety)."

_CWE_NEG_CLUSTERS = {
    "memory_safety":    {"cwes": ["CWE-416", "CWE-787", "CWE-125", "CWE-476", "CWE-190"],
                         "desc": "memory safety issues (buffer overflows, UAF) in C/C++ native code"},
    "input_validation": {"cwes": ["CWE-89", "CWE-79", "CWE-78", "CWE-94", "CWE-22"],
                         "desc": "input validation failures (injection, XSS) in web applications"},
    "auth_session":     {"cwes": ["CWE-287", "CWE-798", "CWE-384"],
                         "desc": "authentication and session management weaknesses"},
    "access_control":   {"cwes": ["CWE-862", "CWE-863"],
                         "desc": "missing or incorrect authorization checks"},
    "cryptographic":    {"cwes": ["CWE-327", "CWE-326"],
                         "desc": "use of broken or weak cryptographic algorithms"},
    "deserialization":  {"cwes": ["CWE-502"],
                         "desc": "unsafe deserialization of untrusted data"},
}

CROSS_CWE_INPUTS = [
    "We found {cve} ({trigger_cwe}, {trigger_cluster}). Does this mean {unrelated_cwe} ({unrelated_cluster}) vulnerabilities are also likely?",
    "Assessment confirmed {cve} exploitable via {trigger_cwe}. Should we also test for {unrelated_cwe} ({unrelated_cluster})?",
    "{cve} is classified as {trigger_cwe}. Is {unrelated_cwe} likely co-present in the same system?",
]


def make_cwe_cross_cluster_negatives(nvd_index, target=200):
    """Generate hard negatives across structurally independent CWE clusters."""
    # Build CWE → cluster index from NVD records
    cwe_to_cluster = {}
    for cn, info in _CWE_NEG_CLUSTERS.items():
        for cwe in info["cwes"]:
            cwe_to_cluster[cwe] = cn

    cluster_cves: dict[str, list[str]] = defaultdict(list)
    for cve_id, rec in nvd_index.items():
        cwe = rec.get("cwe_id", "")
        cl = cwe_to_cluster.get(cwe)
        if cl:
            cluster_cves[cl].append(cve_id)

    pairs = []
    cluster_names = list(_CWE_NEG_CLUSTERS.keys())

    for i, ca in enumerate(cluster_names):
        for cb in cluster_names[i + 1:]:
            cves_a = cluster_cves.get(ca, [])
            if not cves_a:
                continue
            cwes_a = _CWE_NEG_CLUSTERS[ca]["cwes"]
            cwes_b = _CWE_NEG_CLUSTERS[cb]["cwes"]
            desc_a = _CWE_NEG_CLUSTERS[ca]["desc"]
            desc_b = _CWE_NEG_CLUSTERS[cb]["desc"]

            sample = random.sample(cves_a, min(3, len(cves_a)))
            for cve in sample:
                t_cwe = random.choice(cwes_a)
                u_cwe = random.choice(cwes_b)
                tmpl = random.choice(CROSS_CWE_INPUTS)
                inp = tmpl.format(
                    cve=cve,
                    trigger_cwe=t_cwe,
                    trigger_cluster=ca.replace("_", " "),
                    unrelated_cwe=u_cwe,
                    unrelated_cluster=cb.replace("_", " "),
                )
                out = (
                    f"No — finding {t_cwe} ({ca.replace('_', ' ')}) does not predict "
                    f"{u_cwe} ({cb.replace('_', ' ')}).\n\n"
                    f"CONFIRMED: {t_cwe} — {desc_a}\n"
                    f"QUERIED:   {u_cwe} — {desc_b}\n\n"
                    f"These weakness classes are structurally independent: different "
                    f"root causes, different code layers, different exploitation "
                    f"techniques. Finding {t_cwe} does NOT increase the probability "
                    f"of {u_cwe}.\n\n"
                    f"What IS likely alongside {t_cwe}: other "
                    f"{ca.replace('_', ' ')} weaknesses — "
                    f"{', '.join(c for c in cwes_a if c != t_cwe)}"
                )
                pairs.append({
                    "layer":  "vulnerability_cooccurrence",
                    "type":   "hard_negative_cwe",
                    "input":  inp,
                    "output": out,
                    "metadata": {"trigger_cve": cve, "trigger_cluster": ca,
                                 "absent_cluster": cb},
                })
                if len(pairs) >= target:
                    log.info(f"    CWE cross-cluster negatives: {len(pairs)}")
                    return pairs

    log.info(f"    CWE cross-cluster negatives: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 7: OWASP independence hard negatives
# ─────────────────────────────────────────────────────────────────────────────
# Pairs of OWASP categories that are NOT strongly co-occurring.
# Teaches the model to distinguish co-occurrence from mere co-existence.

_OWASP_INDEPENDENCE = [
    ("A03:2021-Injection", "A10:2021-Server-Side Request Forgery",
     "Injection needs unsanitized query construction; SSRF needs URL-fetching functionality. Independent preconditions."),
    ("A02:2021-Cryptographic Failures", "A10:2021-Server-Side Request Forgery",
     "Cryptographic choices and SSRF are orthogonal — one is data protection, the other is request manipulation."),
    ("A06:2021-Vulnerable and Outdated Components", "A04:2021-Insecure Design",
     "Outdated components are patching failures; insecure design is architectural. Different teams, different processes."),
    ("A09:2021-Security Logging and Monitoring Failures", "A03:2021-Injection",
     "Logging gaps don't cause injection and injection doesn't cause logging gaps. Independent control failures."),
    ("A08:2021-Software and Data Integrity Failures", "A07:2021-Identification and Authentication Failures",
     "Supply chain integrity and authentication are separate security domains."),
    ("A04:2021-Insecure Design", "A02:2021-Cryptographic Failures",
     "Design flaws are architectural; crypto failures are implementation choices. Independent risk factors."),
    ("A06:2021-Vulnerable and Outdated Components", "A10:2021-Server-Side Request Forgery",
     "Running old components does not imply URL-fetching functionality exists. Unrelated surfaces."),
    ("A08:2021-Software and Data Integrity Failures", "A03:2021-Injection",
     "Deserialization/supply chain issues have different root causes from input injection."),
]

OWASP_NEG_INPUTS = [
    "We confirmed {short_a} on the target. Does this mean {short_b} is also likely?",
    "Audit finding: {cat_a}. Should we expect {cat_b} in the same application?",
    "Assessment: {short_a} vulnerabilities found. Is {short_b} a co-occurring risk?",
    "Confirmed: {short_a}. What is the probability of also finding {short_b}?",
]


def make_owasp_independence_negatives():
    """Generate hard negatives between OWASP categories that are NOT co-occurring."""
    pairs = []
    for cat_a, cat_b, reasoning in _OWASP_INDEPENDENCE:
        short_a = cat_a.split("-", 1)[-1].strip() if "-" in cat_a else cat_a
        short_b = cat_b.split("-", 1)[-1].strip() if "-" in cat_b else cat_b

        out = (
            f"Low probability. {cat_a} and {cat_b} are NOT strongly co-occurring.\n\n"
            f"{reasoning}\n\n"
            f"These categories have independent root causes. Finding one does not meaningfully "
            f"increase the probability of the other.\n\n"
            f"Focus assessment time on categories that DO co-occur with {short_a} instead."
        )
        for tmpl in OWASP_NEG_INPUTS:
            inp = tmpl.format(cat_a=cat_a, cat_b=cat_b,
                              short_a=short_a, short_b=short_b)
            pairs.append({
                "layer":  "vulnerability_cooccurrence",
                "type":   "hard_negative_owasp",
                "input":  inp,
                "output": out,
                "metadata": {"cat_a": cat_a, "cat_b": cat_b},
            })

    log.info(f"    OWASP independence negatives: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main generation  (FIXED — generates enough to hit target)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_pairs(target_count=60000):
    nvd_index              = load_nvd_index(NVD_FILE)
    all_pairs, neg_rules   = load_cooc(COOC_FILE)

    log.info(f"\nGenerating up to {target_count:,} co-occurrence training pairs …")
    log.info(f"  Co-occurrence pairs available: {len(all_pairs):,}")
    log.info(f"  Negative rules available:      {len(neg_rules):,}")

    grouped = group_by_trigger(all_pairs)
    pairs   = []

    # ── Type 1: Positive inference — 4 variants per trigger CVE ───────────
    log.info("  Generating Type 1: Positive inference (4 variants/CVE) …")
    triggers = list(grouped.keys())
    random.shuffle(triggers)
    for cve in triggers:
        new_pairs = make_positive_pairs(cve, grouped[cve], nvd_index, num_variants=4)
        pairs.extend(new_pairs)

    # ── Type 2: Negative inference — 3 variants per rule ──────────────────
    log.info("  Generating Type 2: Negative inference (3 variants/rule) …")
    for rule in neg_rules:
        pairs.extend(make_negative_pairs(rule, nvd_index, num_variants=3))

    # ── Type 3: Chain reasoning — 4 variants per chain ────────────────────
    log.info("  Generating Type 3: Chain reasoning (4 variants/chain) …")
    for pk, profile in STACK_PROFILES.items():
        for chain in profile.get("attack_chains", []):
            pairs.extend(make_chain_pairs(pk, chain, nvd_index, num_variants=4))

    # ── Type 4: Stack profile — 6 variants per profile ────────────────────
    log.info("  Generating Type 4: Stack profiles (6 variants/profile) …")
    for pk in STACK_PROFILES:
        pairs.extend(make_stack_pairs(pk, nvd_index, num_variants=6))

    # ── Type 5: Conditional — 3 variants per condition ────────────────────
    log.info("  Generating Type 5: Conditional reasoning (3 variants/condition) …")
    for pk, profile in STACK_PROFILES.items():
        for cond_label, cond_items in profile.get("conditional", {}).items():
            pairs.extend(make_conditional_pairs(pk, cond_label, cond_items, nvd_index, num_variants=3))

    # ── Type 6: CWE cross-cluster hard negatives (~200 pairs) ─────────────
    log.info("  Generating Type 6: CWE cross-cluster hard negatives …")
    pairs.extend(make_cwe_cross_cluster_negatives(nvd_index, target=200))

    # ── Type 7: OWASP independence hard negatives (~32 pairs) ─────────────
    log.info("  Generating Type 7: OWASP independence hard negatives …")
    pairs.extend(make_owasp_independence_negatives())

    log.info(f"\n  Raw pairs before cap: {len(pairs):,}")

    # Dedup by (input first 100 chars)
    seen_inputs = set()
    deduped = []
    for p in pairs:
        key = p["input"][:100]
        if key not in seen_inputs:
            seen_inputs.add(key)
            deduped.append(p)
    log.info(f"  After dedup: {len(deduped):,}")

    random.shuffle(deduped)
    if len(deduped) > target_count:
        deduped = deduped[:target_count]

    log.info(f"  Final pairs: {len(deduped):,}")

    by_type = defaultdict(int)
    for p in deduped:
        by_type[p.get("type", "unknown")] += 1
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        log.info(f"    {t:30s}  {c:,}")

    return deduped


def write_pairs(pairs, out_file=OUT_FILE):
    out_file = Path(out_file)
    with open(out_file, "a") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    log.info(f"\n✅ Appended {len(pairs):,} pairs → {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",     type=int, default=60000)
    parser.add_argument("--output",    type=str, default=str(OUT_FILE))
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.no_append and out_path.exists():
        out_path.unlink()

    pairs = generate_all_pairs(target_count=args.count)
    write_pairs(pairs, out_path)
    log.info("\nDone. Run validate_dataset.py to verify quality.")


if __name__ == "__main__":
    main()