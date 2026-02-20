"""
validate_dataset.py
-------------------
Pre-training dataset health check.

CHANGES:
  - Tokenizer updated from mistralai/Mistral-7B-Instruct-v0.3
    to fdtn-ai/Foundation-Sec-8B (Llama 3.1-based tiktoken tokenizer).
    Security text tokenizes at ~2.7 chars/token under Llama 3 (vs ~2.9 for
    Mistral) due to the larger base vocab (128k vs 32k tokens). The fallback
    heuristic is updated to 2.7. This matters: a Mistral-calibrated estimate
    would mark examples as "fits in 4096" when they tokenize to 4400+.

  - MAX_TOKENS updated to 4096 to match the new SFTConfig.max_length.

  - Correlation/co-occurrence layer size summary added to the report so you
    can confirm these layers are populated enough before running weighted
    training. A minimum of 500 pairs per correlation layer is recommended
    for the 3× oversampling to be meaningful.

Run BEFORE finetuning.py:
    python validate_dataset.py
    python validate_dataset.py --fix              # auto-drop bad examples
    python validate_dataset.py --no-tokenizer     # fast heuristic mode
    python validate_dataset.py --max-tokens 8192  # if using full context
"""

import json
import argparse
from collections import Counter
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_PATH      = "data/training_pairs.jsonl"
MIN_OUTPUT_CHARS  = 80
MAX_TOKENS        = 4096       # must match SFTConfig.max_length in finetuning.py
IDEAL_LAYER_SHARE = 0.05

# Llama 3 / Foundation-Sec-8B: larger vocab (128k) → denser tokenization
# Security text (CVE IDs, hex, code, payloads) averages ~2.7 chars/token
CHARS_PER_TOKEN_FALLBACK = 2.7

# Minimum correlation/co-occurrence pairs for 3× oversampling to be meaningful
MIN_CORRELATION_PAIRS = 500
MIN_COOCCURRENCE_PAIRS = 500

PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
PROMPT_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

# ── Tokenizer (lazy loaded once) ───────────────────────────────────────────────
_tokenizer = None

def get_tokenizer(model_name: str = "fdtn-ai/Foundation-Sec-8B"):
    """
    Load the Foundation-Sec-8B tokenizer once and cache it.
    Foundation-Sec-8B uses the Llama 3.1 tiktoken tokenizer (128k vocab).
    Falls back to char heuristic if transformers is not installed.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        print(f"  Loading tokenizer: {model_name} (first time only)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  Tokenizer loaded. Vocab size: {_tokenizer.vocab_size:,}")
        return _tokenizer
    except Exception as e:
        print(f"  ⚠️  Could not load tokenizer: {e}")
        print(f"      Falling back to char heuristic ({CHARS_PER_TOKEN_FALLBACK} chars/token).")
        return None


def build_prompt(pair: dict) -> str:
    """Build the full training text for a pair (mirrors finetuning.py format_example)."""
    instruction = pair.get("instruction", "").strip()
    inp         = pair.get("input", "").strip()
    output      = pair.get("output", "").strip()

    if inp:
        return PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=instruction, input=inp, output=output
        )
    else:
        return PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=instruction, output=output
        )


def count_tokens(pair: dict, use_tokenizer: bool = True) -> int:
    """
    Count tokens for a training pair.
    Uses Foundation-Sec-8B tokenizer when available, falls back to heuristic.
    """
    text = build_prompt(pair)

    if use_tokenizer:
        tok = get_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=True))

    return int(len(text) / CHARS_PER_TOKEN_FALLBACK)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_pairs(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {path}")
        return []
    pairs = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Line {i+1}: JSON parse error — {e}")
    return pairs


# ── Validation ─────────────────────────────────────────────────────────────────
def validate(pairs: list, max_tokens: int = MAX_TOKENS, use_tokenizer: bool = True) -> dict:
    issues = {
        "empty_output":     [],
        "short_output":     [],
        "over_token_limit": [],
        "truncated_likely": [],
        "duplicates":       [],
        "missing_layer":    [],
    }

    layer_counts: Counter = Counter()
    token_counts: list[int] = []
    seen: dict[tuple, int] = {}

    for i, pair in enumerate(pairs):
        layer = pair.get("layer", "")
        if not layer:
            issues["missing_layer"].append(i)

        layer_counts[layer or "MISSING"] += 1

        output = pair.get("output", "").strip()
        if not output:
            issues["empty_output"].append(i)
            continue
        if len(output) < MIN_OUTPUT_CHARS:
            issues["short_output"].append(i)

        n_tok = count_tokens(pair, use_tokenizer=use_tokenizer)
        token_counts.append(n_tok)

        if n_tok > max_tokens:
            issues["over_token_limit"].append((i, n_tok))
        elif n_tok > max_tokens * 0.9:
            issues["truncated_likely"].append((i, n_tok))

        key = (
            pair.get("instruction", "")[:120],
            pair.get("output", "")[:120],
        )
        if key in seen:
            issues["duplicates"].append((i, seen[key]))
        else:
            seen[key] = i

    return {
        "issues":       issues,
        "layer_counts": layer_counts,
        "token_counts": token_counts,
        "total":        len(pairs),
    }


def print_report(result: dict, max_tokens: int) -> None:
    issues       = result["issues"]
    layer_counts = result["layer_counts"]
    token_counts = result["token_counts"]
    total        = result["total"]

    print(f"\n{'='*60}")
    print(f"  Dataset Validation Report  (Foundation-Sec-8B / 4096-token)")
    print(f"{'='*60}")
    print(f"  Total pairs: {total:,}")

    if token_counts:
        avg  = sum(token_counts) / len(token_counts)
        mmax = max(token_counts)
        p95  = sorted(token_counts)[int(len(token_counts) * 0.95)]
        print(f"  Token stats: avg={avg:.0f}  p95={p95}  max={mmax}  limit={max_tokens}")

    print(f"\n  Layer distribution:")
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        share = count / total * 100
        flag  = "  ⚠️  below 5% share" if share < IDEAL_LAYER_SHARE * 100 else ""
        print(f"    {layer:<38} {count:>7,}  ({share:4.1f}%){flag}")

    # ── Correlation/co-occurrence readiness check ──────────────────────────────
    corr_count = layer_counts.get("vulnerability_correlation", 0)
    cooc_count = layer_counts.get("vulnerability_cooccurrence", 0)
    print(f"\n  Correlation/co-occurrence readiness (3× oversample):")
    corr_ok = corr_count >= MIN_CORRELATION_PAIRS
    cooc_ok = cooc_count >= MIN_COOCCURRENCE_PAIRS
    print(f"    vulnerability_correlation:  {corr_count:,}  {'✅' if corr_ok else f'⚠️  below recommended {MIN_CORRELATION_PAIRS}'}")
    print(f"    vulnerability_cooccurrence: {cooc_count:,}  {'✅' if cooc_ok else f'⚠️  below recommended {MIN_COOCCURRENCE_PAIRS}'}")
    if not corr_ok or not cooc_ok:
        print(f"    → Run: python run_pipeline.py --correlate  to enrich these layers")

    print(f"\n  Issues found:")
    print(f"    empty_output:      {len(issues['empty_output'])}")
    print(f"    short_output:      {len(issues['short_output'])}  (< {MIN_OUTPUT_CHARS} chars)")
    print(f"    over_token_limit:  {len(issues['over_token_limit'])}  (> {max_tokens} tokens)")
    print(f"    truncated_likely:  {len(issues['truncated_likely'])}  (> {int(max_tokens*0.9)} tokens)")
    print(f"    duplicates:        {len(issues['duplicates'])}")
    print(f"    missing_layer:     {len(issues['missing_layer'])}")

    bad = (
        len(issues["empty_output"])
        + len(issues["short_output"])
        + len(issues["over_token_limit"])
        + len(issues["duplicates"])
        + len(issues["missing_layer"])
    )
    print(f"\n  Total problematic pairs: {bad:,} / {total:,}  ({bad/total*100:.1f}%)")
    if bad == 0:
        print("  ✅ Dataset looks clean — ready for finetuning.py")
    else:
        print("  ⚠️  Run with --fix to auto-drop problematic pairs before training")


def fix_dataset(pairs: list, result: dict, path: str, max_tokens: int) -> None:
    """Drop problematic pairs and overwrite the file."""
    issues = result["issues"]
    bad_indices = set()
    bad_indices.update(issues["empty_output"])
    bad_indices.update(issues["short_output"])
    bad_indices.update(i for i, _ in issues["over_token_limit"])
    bad_indices.update(issues["missing_layer"])
    bad_indices.update(i for i, _ in issues["duplicates"])

    clean = [p for i, p in enumerate(pairs) if i not in bad_indices]
    with open(path, "w", encoding="utf-8") as f:
        for p in clean:
            f.write(json.dumps(p) + "\n")
    print(f"\n✅ Dropped {len(bad_indices)} pairs  →  {len(clean):,} clean pairs written to {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Validate training_pairs.jsonl")
    parser.add_argument("--path",         default=DEFAULT_PATH, help="Path to training_pairs.jsonl")
    parser.add_argument("--fix",          action="store_true",  help="Auto-drop bad examples")
    parser.add_argument("--no-tokenizer", action="store_true",  help="Use char heuristic instead of real tokenizer")
    parser.add_argument("--max-tokens",   type=int, default=MAX_TOKENS, help="Token limit per example")
    args = parser.parse_args()

    print(f"Loading: {args.path}")
    pairs = load_pairs(args.path)
    if not pairs:
        return

    print(f"Validating {len(pairs):,} pairs...")
    result = validate(pairs, max_tokens=args.max_tokens, use_tokenizer=not args.no_tokenizer)
    print_report(result, max_tokens=args.max_tokens)

    if args.fix:
        fix_dataset(pairs, result, args.path, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()