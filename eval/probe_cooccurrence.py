"""
eval/probe_cooccurrence.py
--------------------------
Ground-truth probe evaluation for the correlation & co-occurrence novelty.

WHY THIS EVAL EXISTS
─────────────────────
Standard fine-tuning eval (train/eval loss) cannot tell you whether the model
learned the P(B|A) co-occurrence structure or just memorized output templates.
A model with low eval loss could still:
  • List wrong CVEs that sound plausible
  • Get P(B|A) direction correct but magnitude wrong
  • Hallucinate co-occurring CVEs not in the ground truth graph
  • Fail on the harder asymmetric cases (P(B|A) >> P(A|B))

This script uses your raw_cooccurrence.json and raw_correlations.json as
ground truth and constructs probes that directly test these failure modes.

METRICS COMPUTED
─────────────────
1. Recall@K (K=3,5,10)
   — Of the top-K ground-truth co-occurring vulnerabilities, how many
     does the model mention in its response?
   — Computed separately for OWASP-level, CWE-level, and CVE-level probes.

2. Hallucination Rate
   — Fraction of vulnerabilities the model mentions that are NOT in the
     ground-truth co-occurrence set for that probe.
   — Measures over-confidence/false positives.

3. Probability Calibration Error (PCE) — OWASP probes only
   — Ground truth has P(B|A) probabilities from NVD analysis.
   — Model responses contain percentage statements ("40% probability").
   — PCE = mean |predicted_prob - ground_truth_prob| across probes.
   — Requires the model to actually output probability numbers.

4. Directionality Accuracy — asymmetric pairs only
   — Ground truth: pairs where P(B|A) ≠ P(A|B) by >20% absolute.
   — Test: probe from A→B and B→A. Model should rank B higher when
     probing from A, and A higher when probing from B.
   — Tests whether the model understood statistical direction, not just
     that two things are "related."

5. Signal-type Recall Breakdown
   — For correlation probes, ground truth has signal types:
     shared_cwe | shared_product | shared_attack_technique |
     kev_campaign_temporal | exploit_chain_cooccurrence
   — Does recall differ by signal type? (kev_campaign harder than shared_cwe?)
   — Identifies which correlation signal types the model learned best.

6. Absent Vulnerability Rejection Rate
   — Ground truth also has "likely_absent" lists (vulnerabilities the model
     should NOT predict when it sees the probe CVE).
   — Tests: does the model avoid predicting these?

USAGE
──────
  # Test the Phase 2 merged checkpoint:
  python eval/probe_cooccurrence.py --model ./checkpoints/vuln-foundation-sec-8b-phase2/merged

  # Test Phase 1 (before correlation specialization) as baseline:
  python eval/probe_cooccurrence.py --model ./checkpoints/vuln-foundation-sec-8b/merged

  # Compare both and write JSON results:
  python eval/probe_cooccurrence.py --model <phase2_path> --baseline <phase1_path> --output eval/results.json

  # Quick run (first 50 probes only):
  python eval/probe_cooccurrence.py --model <path> --max-probes 50

  # Use HF Hub model:
  python eval/probe_cooccurrence.py --model adityajayashankar/vuln-foundation-sec-8b-correlation
"""

import json
import re
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
COOCCURRENCE_PATH = Path("data") / "raw_cooccurrence.json"
CORRELATIONS_PATH = Path("data") / "raw_correlations.json"

# ── Evaluation config ──────────────────────────────────────────────────────────
RECALL_K          = [3, 5, 10]
MAX_NEW_TOKENS    = 512
TEMPERATURE       = 0.1
MIN_GT_SUPPORT    = 3      # ignore ground-truth pairs with fewer than 3 supporting records
MIN_GT_PROB       = 0.15   # ignore ground-truth co-occurrences below 15% probability
ASYM_THRESHOLD    = 0.20   # |P(B|A) - P(A|B)| > 0.20 → directional probe
RANDOM_SEED       = 42


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Probe:
    """A single ground-truth probe question."""
    probe_id:       str
    probe_type:     str          # owasp | cwe | cve | directional | absent
    question:       str          # instruction sent to model
    context:        str          # optional input context (CVE description etc.)
    ground_truth:   list[str]    # list of expected entities (CVE-IDs, OWASP cats, CWEs)
    absent_truth:   list[str]    # entities model should NOT mention
    signal_types:   list[str]    # e.g. ["kev_campaign_temporal", "shared_cwe"]
    gt_probs:       dict         # {entity: probability} from ground truth
    direction:      str          # "A→B" or "B→A" for directional probes, else ""


@dataclass
class ProbeResult:
    """Result of evaluating a single probe."""
    probe_id:        str
    probe_type:      str
    recall_at_k:     dict         # {3: 0.67, 5: 0.60, 10: 0.40}
    hallucination:   float        # fraction of model mentions not in ground truth
    prob_error:      float | None # |predicted_prob - gt_prob| average, or None
    direction_ok:    bool | None  # True/False for directional probes, None otherwise
    absent_rejected: float        # fraction of absent entities correctly not mentioned
    signal_types:    list[str]
    model_response:  str          # truncated for storage


# ── Ground truth loaders ───────────────────────────────────────────────────────

def load_cooccurrence(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Co-occurrence data not found: {path}\n"
            f"Run: python run_pipeline.py --correlate"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_correlations(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  ⚠️  Correlations file not found: {path} — skipping CVE-level probes")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # raw_correlations.json is a list of correlation records
    if isinstance(raw, list):
        return raw
    # Or it might be wrapped
    return raw.get("records", [])


# ── Probe builders ─────────────────────────────────────────────────────────────

def build_owasp_probes(cooc_data: dict, max_probes: int) -> list[Probe]:
    """
    Build OWASP-level co-occurrence probes.
    Ground truth: owasp_cooccurrence[focal_cat]["likely_present"]
    """
    probes: list[Probe] = []
    owasp_cooc = cooc_data.get("owasp_cooccurrence", {})

    for focal_cat, data in owasp_cooc.items():
        present = data.get("likely_present", [])
        absent  = data.get("likely_absent", [])

        # Filter to statistically meaningful entries
        strong_present = [
            p for p in present
            if p.get("probability", 0) >= MIN_GT_PROB
            and (
                p.get("support") == "N/A"
                or (isinstance(p.get("support"), (int, float)) and p["support"] >= MIN_GT_SUPPORT)
            )
        ]

        if not strong_present:
            continue

        short_name = focal_cat.split("-", 1)[-1].strip() if "-" in focal_cat else focal_cat

        probe = Probe(
            probe_id     = f"owasp_{focal_cat.replace(':', '_').replace(' ', '_')}",
            probe_type   = "owasp",
            question     = (
                f"During a security audit we confirmed {short_name} ({focal_cat}). "
                f"What other OWASP vulnerability categories are statistically likely to also be present in the same application?"
            ),
            context      = data.get("reasoning", ""),
            ground_truth = [p["category"] for p in strong_present],
            absent_truth = [p.get("category", "") for p in absent if p.get("category")],
            signal_types = ["owasp_cooccurrence"],
            gt_probs     = {p["category"]: p["probability"] for p in strong_present},
            direction    = "",
        )
        probes.append(probe)

    random.shuffle(probes)
    return probes[:max_probes]


def build_cwe_probes(cooc_data: dict, max_probes: int) -> list[Probe]:
    """
    Build CWE family cluster probes.
    Ground truth: cwe_clusters[cluster_name]["members"]
    """
    probes: list[Probe] = []
    cwe_clusters = cooc_data.get("cwe_clusters", {})

    for cluster_name, cluster_data in cwe_clusters.items():
        members = cluster_data.get("members", [])
        if len(members) < 2:
            continue

        # Probe: given one CWE in the cluster, predict the others
        for focal_cwe in members[:3]:   # probe from first 3 members only
            others = [m for m in members if m != focal_cwe]
            probe  = Probe(
                probe_id     = f"cwe_{focal_cwe}_{cluster_name[:20]}",
                probe_type   = "cwe",
                question     = (
                    f"A codebase has a vulnerability classified as {focal_cwe}. "
                    f"Based on weakness family clustering, what other CWE weaknesses are likely to co-exist in the same codebase?"
                ),
                context      = cluster_data.get("description", ""),
                ground_truth = others,
                absent_truth = [],
                signal_types = ["cwe_family_cluster"],
                gt_probs     = {m: 0.7 for m in others},  # cluster members are equally likely
                direction    = "",
            )
            probes.append(probe)

    random.shuffle(probes)
    return probes[:max_probes]


def build_cve_probes(corr_records: list[dict], max_probes: int) -> list[Probe]:
    """
    Build CVE-level correlation probes using raw_correlations.json.
    Tests the multi-signal correlation graph directly.
    """
    probes: list[Probe] = []

    for rec in corr_records:
        cve_id  = rec.get("cve_id", "")
        desc    = rec.get("description", "")
        related = rec.get("related_vulnerabilities", [])

        if not cve_id or not related:
            continue

        # Strong related: high correlation score, at least 2 signals
        strong = [
            r for r in related
            if r.get("correlation_score", 0) >= 2
        ]
        if not strong:
            continue

        # Group by signal type for signal-type recall breakdown
        signal_types = list({
            s.split(":")[0]
            for r in strong
            for s in r.get("signals", [])
        })

        probe = Probe(
            probe_id     = f"cve_{cve_id}",
            probe_type   = "cve",
            question     = f"What vulnerabilities are correlated with {cve_id} and why?",
            context      = desc[:400],
            ground_truth = [r["cve_id"] for r in strong[:10]],
            absent_truth = [],
            signal_types = signal_types,
            gt_probs     = {r["cve_id"]: min(r["correlation_score"] / 10, 0.99) for r in strong},
            direction    = "",
        )
        probes.append(probe)

    random.shuffle(probes)
    return probes[:max_probes]


def build_directional_probes(cooc_data: dict, max_probes: int) -> list[Probe]:
    """
    Build asymmetric directionality probes.

    For OWASP pairs where P(B|A) and P(A|B) differ by > ASYM_THRESHOLD:
      Probe 1 (A→B): given A, expect B in top-ranked predictions
      Probe 2 (B→A): given B, expect A lower-ranked or absent

    This tests whether the model learned statistical direction,
    not just "A and B are related."
    """
    probes: list[Probe] = []
    owasp_cooc = cooc_data.get("owasp_cooccurrence", {})

    # Build full P(B|A) matrix
    prob_matrix: dict[str, dict[str, float]] = {}
    for focal, data in owasp_cooc.items():
        prob_matrix[focal] = {
            p["category"]: p["probability"]
            for p in data.get("likely_present", [])
        }

    # Find asymmetric pairs
    checked: set[frozenset] = set()
    for cat_a, probs_a in prob_matrix.items():
        for cat_b, p_b_given_a in probs_a.items():
            pair = frozenset({cat_a, cat_b})
            if pair in checked:
                continue
            checked.add(pair)

            p_a_given_b = prob_matrix.get(cat_b, {}).get(cat_a, 0.0)
            diff = abs(p_b_given_a - p_a_given_b)

            if diff < ASYM_THRESHOLD:
                continue

            # cat_a → cat_b is the strong direction
            strong_a, strong_b = (cat_a, cat_b) if p_b_given_a > p_a_given_b else (cat_b, cat_a)
            p_strong = max(p_b_given_a, p_a_given_b)
            p_weak   = min(p_b_given_a, p_a_given_b)

            short_a = strong_a.split("-", 1)[-1].strip() if "-" in strong_a else strong_a
            short_b = strong_b.split("-", 1)[-1].strip() if "-" in strong_b else strong_b

            # Probe from the STRONG direction: model should rank B highly
            probes.append(Probe(
                probe_id     = f"dir_{strong_a[:30]}_{strong_b[:30]}_fwd",
                probe_type   = "directional",
                question     = (
                    f"When {short_a} is found in an application, what other vulnerability categories are most likely to also be present?"
                ),
                context      = "",
                ground_truth = [strong_b],
                absent_truth = [],
                signal_types = ["directional_cooccurrence"],
                gt_probs     = {strong_b: p_strong},
                direction    = "A→B",
            ))

            # Probe from the WEAK direction: model should NOT rank A as highly
            probes.append(Probe(
                probe_id     = f"dir_{strong_a[:30]}_{strong_b[:30]}_rev",
                probe_type   = "directional",
                question     = (
                    f"When {short_b} is found in an application, what other vulnerability categories are most likely to also be present?"
                ),
                context      = "",
                ground_truth = [],        # A is weak predictor from B's side
                absent_truth = [strong_a] if p_weak < 0.15 else [],  # only absent if truly weak
                signal_types = ["directional_cooccurrence"],
                gt_probs     = {strong_a: p_weak},
                direction    = "B→A",
            ))

    random.shuffle(probes)
    return probes[:max_probes]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_pipeline(model_path: str):
    """Load model for inference. Uses bfloat16 matching fine-tuning dtype."""
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    pipe = hf_pipeline(
        "text-generation",
        model              = model,
        tokenizer          = tokenizer,
        max_new_tokens     = MAX_NEW_TOKENS,
        temperature        = TEMPERATURE,
        do_sample          = True,
        repetition_penalty = 1.1,
    )
    print("✅ Model ready.")
    return pipe


def query_model(pipe, probe: Probe) -> str:
    """Run a single probe through the model. Returns generated response text."""
    if probe.context.strip():
        prompt = (
            f"[SYSTEM]: You are a vulnerability correlation expert. "
            f"Identify statistically co-occurring vulnerabilities with precise reasoning.\n\n"
            f"### Instruction:\n{probe.question}\n\n"
            f"### Input:\n{probe.context}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"[SYSTEM]: You are a vulnerability correlation expert. "
            f"Identify statistically co-occurring vulnerabilities with precise reasoning.\n\n"
            f"### Instruction:\n{probe.question}\n\n"
            f"### Response:\n"
        )

    try:
        output = pipe(prompt)[0]["generated_text"]
        if output.startswith(prompt):
            return output[len(prompt):].strip()
        marker = "### Response:\n"
        idx = output.rfind(marker)
        return output[idx + len(marker):].strip() if idx != -1 else output.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Scoring ────────────────────────────────────────────────────────────────────

def extract_mentioned_entities(response: str, entity_type: str) -> list[str]:
    """
    Extract mentioned CVE IDs, OWASP categories, or CWE IDs from model response.
    Uses regex patterns matched to each entity type.
    """
    if entity_type == "cve":
        return list(set(re.findall(r"CVE-\d{4}-\d{4,}", response.upper())))

    elif entity_type == "owasp":
        # Match "A01", "A01:2021", "Broken Access Control", etc.
        codes = re.findall(r"A\d{2}(?::\d{4})?", response)
        names = re.findall(
            r"(Broken Access Control|Cryptographic Failures|Injection|"
            r"Insecure Design|Security Misconfiguration|Vulnerable and Outdated|"
            r"Identification and Authentication|Software and Data Integrity|"
            r"Security Logging|Server-Side Request Forgery)",
            response, re.IGNORECASE
        )
        return list(set(codes + names))

    elif entity_type == "cwe":
        return list(set(re.findall(r"CWE-\d+", response.upper())))

    return []


def extract_mentioned_probabilities(response: str) -> dict[str, float]:
    """
    Extract probability statements from model response.
    Handles: "40% probability", "40%", "probability: 0.4", "confidence: 40%"
    Returns dict mapping mentioned category snippet → probability (0-1 float).
    """
    probs = {}
    # Pattern: some text followed by % probability or probability: X%
    pattern = re.compile(
        r"([A-Z][^\n:•]{5,50}?)[\s:–-]+(\d{1,3})%",
        re.IGNORECASE
    )
    for match in pattern.finditer(response):
        entity_hint = match.group(1).strip()[:40]
        prob        = int(match.group(2)) / 100
        if 0.01 <= prob <= 0.99:
            probs[entity_hint] = prob
    return probs


def score_probe(probe: Probe, response: str) -> ProbeResult:
    """Compute all metrics for a single probe result."""

    # Determine entity type from probe type
    if probe.probe_type in ("owasp", "directional"):
        entity_type = "owasp"
    elif probe.probe_type == "cwe":
        entity_type = "cwe"
    else:
        entity_type = "cve"

    mentioned = extract_mentioned_entities(response, entity_type)
    gt_set    = set(probe.ground_truth)
    ab_set    = set(probe.absent_truth)

    # ── Recall@K ──────────────────────────────────────────────────────────
    # We can only compute real Recall@K if we know the order the model listed things.
    # As approximation: if the entity appears anywhere in the response, count it.
    recall_at_k = {}
    for k in RECALL_K:
        top_k_gt = probe.ground_truth[:k]    # top-K ground truth (sorted by probability)
        hits      = sum(1 for e in top_k_gt if any(
            e.lower() in m.lower() or m.lower() in e.lower()
            for m in mentioned
        ))
        recall_at_k[k] = hits / max(len(top_k_gt), 1)

    # ── Hallucination rate ─────────────────────────────────────────────────
    if mentioned and gt_set:
        hallucinated = [
            m for m in mentioned
            if not any(gt.lower() in m.lower() or m.lower() in gt.lower() for gt in gt_set)
        ]
        hallucination = len(hallucinated) / max(len(mentioned), 1)
    else:
        hallucination = 0.0

    # ── Probability calibration error ─────────────────────────────────────
    prob_error = None
    if probe.gt_probs and probe.probe_type == "owasp":
        predicted_probs = extract_mentioned_probabilities(response)
        if predicted_probs:
            errors = []
            for gt_entity, gt_prob in probe.gt_probs.items():
                # Find matching predicted prob by entity name similarity
                for pred_entity, pred_prob in predicted_probs.items():
                    short = gt_entity.split("-", 1)[-1].strip()[:15].lower()
                    if short and short in pred_entity.lower():
                        errors.append(abs(pred_prob - gt_prob))
                        break
            if errors:
                prob_error = sum(errors) / len(errors)

    # ── Directionality accuracy ───────────────────────────────────────────
    direction_ok = None
    if probe.probe_type == "directional":
        if probe.direction == "A→B":
            # B (ground truth[0]) should appear in response
            direction_ok = (
                bool(probe.ground_truth) and
                any(probe.ground_truth[0].lower() in m.lower() or m.lower() in probe.ground_truth[0].lower()
                    for m in mentioned)
            )
        elif probe.direction == "B→A" and probe.absent_truth:
            # A (absent_truth[0]) should NOT appear in response
            direction_ok = not any(
                probe.absent_truth[0].lower() in m.lower()
                for m in mentioned
            )

    # ── Absent entity rejection rate ──────────────────────────────────────
    if ab_set:
        correctly_absent = sum(
            1 for ab in ab_set
            if not any(ab.lower() in m.lower() for m in mentioned)
        )
        absent_rejected = correctly_absent / len(ab_set)
    else:
        absent_rejected = 1.0   # no absent entities to test → trivially correct

    return ProbeResult(
        probe_id        = probe.probe_id,
        probe_type      = probe.probe_type,
        recall_at_k     = recall_at_k,
        hallucination   = hallucination,
        prob_error      = prob_error,
        direction_ok    = direction_ok,
        absent_rejected = absent_rejected,
        signal_types    = probe.signal_types,
        model_response  = response[:500],   # truncate for storage
    )


# ── Aggregate report ───────────────────────────────────────────────────────────

def aggregate_results(results: list[ProbeResult]) -> dict:
    """Compute aggregate metrics, broken down by probe type and signal type."""
    by_type: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by_type[r.probe_type].append(r)

    def _agg_type(rs: list[ProbeResult]) -> dict:
        n = len(rs)
        if n == 0:
            return {}
        recall = {k: sum(r.recall_at_k.get(k, 0) for r in rs) / n for k in RECALL_K}
        hall   = sum(r.hallucination for r in rs) / n
        absent = sum(r.absent_rejected for r in rs) / n
        pcal   = [r.prob_error for r in rs if r.prob_error is not None]
        dirs   = [r.direction_ok for r in rs if r.direction_ok is not None]
        return {
            "n":                    n,
            "recall_at_k":         {f"R@{k}": round(v, 3) for k, v in recall.items()},
            "hallucination_rate":  round(hall, 3),
            "absent_rejection":    round(absent, 3),
            "prob_calib_error":    round(sum(pcal) / len(pcal), 3) if pcal else None,
            "directional_acc":     round(sum(dirs) / len(dirs), 3) if dirs else None,
        }

    # Signal-type breakdown (CVE probes only)
    by_signal: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        if r.probe_type == "cve":
            for sig in r.signal_types:
                by_signal[sig].append(r)

    signal_recall = {}
    for sig, rs in by_signal.items():
        signal_recall[sig] = {
            "n":      len(rs),
            "R@5":    round(sum(r.recall_at_k.get(5, 0) for r in rs) / max(len(rs), 1), 3),
            "hall":   round(sum(r.hallucination for r in rs) / max(len(rs), 1), 3),
        }

    overall = _agg_type(results)
    by_type_agg = {t: _agg_type(rs) for t, rs in by_type.items()}

    return {
        "overall":          overall,
        "by_probe_type":    by_type_agg,
        "by_signal_type":   signal_recall,
        "n_probes":         len(results),
    }


def print_report(report: dict, model_name: str, baseline_report: dict | None = None) -> None:
    print(f"\n{'='*65}")
    print(f"  CO-OCCURRENCE PROBE EVAL — {Path(model_name).name}")
    print(f"{'='*65}")
    print(f"  Total probes evaluated: {report['n_probes']}")

    ov = report["overall"]
    print(f"\n  OVERALL")
    for k_label, v in ov.get("recall_at_k", {}).items():
        delta = ""
        if baseline_report:
            base_v = baseline_report["overall"].get("recall_at_k", {}).get(k_label, 0)
            diff   = v - base_v
            delta  = f"  (Δ {diff:+.3f} vs baseline)"
        print(f"    {k_label}:                  {v:.3f}{delta}")
    print(f"    Hallucination rate:      {ov.get('hallucination_rate', 0):.3f}")
    print(f"    Absent rejection:        {ov.get('absent_rejection', 1):.3f}")
    if ov.get("prob_calib_error") is not None:
        print(f"    Probability calib err:   {ov['prob_calib_error']:.3f}")
    if ov.get("directional_acc") is not None:
        print(f"    Directional accuracy:    {ov['directional_acc']:.3f}")

    print(f"\n  BY PROBE TYPE")
    for ptype, metrics in report.get("by_probe_type", {}).items():
        if not metrics:
            continue
        r5 = metrics.get("recall_at_k", {}).get("R@5", 0)
        h  = metrics.get("hallucination_rate", 0)
        n  = metrics.get("n", 0)
        print(f"    {ptype:<20} n={n:<4}  R@5={r5:.3f}  Hall={h:.3f}")

    print(f"\n  BY CORRELATION SIGNAL TYPE (CVE probes)")
    for sig, m in sorted(report.get("by_signal_type", {}).items(), key=lambda x: -x[1]["R@5"]):
        print(f"    {sig:<35} n={m['n']:<4}  R@5={m['R@5']:.3f}  Hall={m['hall']:.3f}")

    print(f"\n{'='*65}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Co-occurrence probe evaluation")
    parser.add_argument("--model",        required=True,  help="Path or HF repo of model to evaluate")
    parser.add_argument("--baseline",     default=None,   help="Optional Phase 1 model for comparison")
    parser.add_argument("--include-base", action="store_true",
                        help="Auto-compare against raw Foundation-Sec-8B (proves fine-tuning value)")
    parser.add_argument("--output",       default="eval/probe_results.json", help="JSON output path")
    parser.add_argument("--max-probes",   type=int, default=200, help="Max probes per type")
    parser.add_argument("--owasp-only",   action="store_true", help="Run OWASP probes only (fastest)")
    args = parser.parse_args()

    # --include-base sets baseline to the raw pre-trained model if not already set
    BASE_MODEL_ID = "fdtn-ai/Foundation-Sec-8B"
    if args.include_base and not args.baseline:
        args.baseline = BASE_MODEL_ID
        print(f"  --include-base: will compare against {BASE_MODEL_ID}")

    random.seed(RANDOM_SEED)

    # ── Load ground truth ──────────────────────────────────────────────────
    print("Loading ground truth data...")
    cooc_data  = load_cooccurrence(COOCCURRENCE_PATH)
    corr_recs  = load_correlations(CORRELATIONS_PATH)
    print(f"  OWASP co-occurrence categories: {len(cooc_data.get('owasp_cooccurrence', {}))}")
    print(f"  CWE clusters:                   {len(cooc_data.get('cwe_clusters', {}))}")
    print(f"  CVE correlation records:        {len(corr_recs)}")

    # ── Build probes ───────────────────────────────────────────────────────
    print("\nBuilding probes...")
    probes: list[Probe] = []
    probes += build_owasp_probes(cooc_data,  max_probes=args.max_probes)
    if not args.owasp_only:
        probes += build_cwe_probes(cooc_data,    max_probes=args.max_probes)
        probes += build_cve_probes(corr_recs,    max_probes=args.max_probes)
        probes += build_directional_probes(cooc_data, max_probes=args.max_probes // 2)

    print(f"  Total probes: {len(probes)}")
    for ptype in ("owasp", "cwe", "cve", "directional"):
        n = sum(1 for p in probes if p.probe_type == ptype)
        print(f"    {ptype:<14} {n}")

    # ── Load model and evaluate ────────────────────────────────────────────
    pipe = load_model_pipeline(args.model)

    results: list[ProbeResult] = []
    print(f"\nRunning {len(probes)} probes...")
    t_start = time.time()

    for i, probe in enumerate(probes):
        response = query_model(pipe, probe)
        result   = score_probe(probe, response)
        results.append(result)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (len(probes) - i - 1) / rate
            avg_r5  = sum(r.recall_at_k.get(5, 0) for r in results) / len(results)
            print(f"  [{i+1:>3}/{len(probes)}]  avg R@5={avg_r5:.3f}  ETA {eta:.0f}s")

    report = aggregate_results(results)

    # ── Baseline comparison ────────────────────────────────────────────────
    baseline_report = None
    if args.baseline:
        print(f"\n{'─'*65}")
        print(f"  Running baseline model: {args.baseline}")
        baseline_pipe    = load_model_pipeline(args.baseline)
        baseline_results = [score_probe(p, query_model(baseline_pipe, p)) for p in probes]
        baseline_report  = aggregate_results(baseline_results)

    # ── Print and save ─────────────────────────────────────────────────────
    print_report(report, args.model, baseline_report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model":          args.model,
        "baseline":       args.baseline,
        "report":         report,
        "baseline_report": baseline_report,
        "results":        [asdict(r) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Full results saved → {out_path}")
    print(f"\n  To compare Phase 1 vs Phase 2:")
    print(f"    python eval/probe_cooccurrence.py \\")
    print(f"      --model ./checkpoints/vuln-foundation-sec-8b-phase2/merged \\")
    print(f"      --baseline ./checkpoints/vuln-foundation-sec-8b/merged \\")
    print(f"      --output eval/phase1_vs_phase2.json")


if __name__ == "__main__":
    main()