"""
validate_dataset.py
-------------------
Pre-training dataset health check.

FIX: Token estimation now uses the ACTUAL Mistral tokenizer instead of the
     naive CHARS_PER_TOKEN=3.8 heuristic.

     Security text (CVE IDs, hex strings, code snippets, payload strings,
     tool names) tokenizes much more densely than prose — the real ratio
     is 2.8–3.2 chars/token, not 3.8. The old heuristic was passing examples
     as "fits in 2048" when they actually tokenize to 2400+, causing silent
     mid-sentence truncation during training.

     The tokenizer is loaded once and cached globally to keep the scan fast
     (~5 min for 10k examples on CPU vs ~3 min with the heuristic).

     Use --no-tokenizer to skip real tokenization and fall back to the fast
     heuristic (useful for quick checks during dataset iteration).

Run BEFORE finetuning.py:
    python validate_dataset.py
    python validate_dataset.py --fix              # auto-drop bad examples
    python validate_dataset.py --no-tokenizer     # fast heuristic mode
    python validate_dataset.py --max-tokens 4096  # if using a larger context window
"""

import json
import argparse
from collections import Counter
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_PATH      = "data/training_pairs.jsonl"
MIN_OUTPUT_CHARS  = 80
MAX_TOKENS        = 2048       # must match SFTConfig.max_length
IDEAL_LAYER_SHARE = 0.05       # warn if any layer < 5% of total

# Fallback heuristic (used with --no-tokenizer)
# Security text avg: 2.9 chars/token (was incorrectly set to 3.8)
CHARS_PER_TOKEN_FALLBACK = 2.9

# Instruction template WITHOUT input block (suppressed when empty — matches finetuning.py fix)
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

def get_tokenizer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
    """
    Load the Mistral tokenizer once and cache it.
    Falls back gracefully if transformers is not installed.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        print(f"  Loading tokenizer: {model_name} (first time only)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    If use_tokenizer=True (default): uses the actual Mistral tokenizer.
    Falls back to char heuristic if tokenizer is unavailable.
    """
    text = build_prompt(pair)

    if use_tokenizer:
        tok = get_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=True))

    # Heuristic fallback
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
        "truncated_likely": [],  # within 10% of limit
        "duplicates":       [],
        "missing_layer":    [],
    }

    layer_counts: Counter = Counter()
    token_counts: list[int] = []
    seen: dict[tuple, int] = {}

    total = len(pairs)
    report_every = max(1, total // 20)  # progress every 5%

    print(f"  Scanning {total:,} examples"
          f" ({'real tokenizer' if use_tokenizer else 'char heuristic'})...")

    for i, pair in enumerate(pairs):
        if i % report_every == 0:
            print(f"    {i:>6,} / {total:,}  ({100*i//total}%)", end="\r")

        output = pair.get("output", "").strip()
        layer  = pair.get("layer", "")

        # Output quality
        if not output:
            issues["empty_output"].append(i)
        elif len(output) < MIN_OUTPUT_CHARS:
            issues["short_output"].append(i)

        # Layer
        if not layer:
            issues["missing_layer"].append(i)
        else:
            layer_counts[layer] += 1

        # Token count
        n_tok = count_tokens(pair, use_tokenizer=use_tokenizer)
        token_counts.append(n_tok)

        if n_tok > max_tokens:
            issues["over_token_limit"].append((i, n_tok))
        elif n_tok > max_tokens * 0.9:
            issues["truncated_likely"].append((i, n_tok))

        # Deduplication
        instr = pair.get("instruction", "").strip()
        key   = (instr[:150], output[:200])
        if key in seen:
            issues["duplicates"].append((i, seen[key]))
        else:
            seen[key] = i

    print(f"    {total:>6,} / {total:,}  (100%)")

    return {
        "total":        total,
        "issues":       issues,
        "layer_counts": dict(layer_counts),
        "token_counts": token_counts,
    }


# ── Report ─────────────────────────────────────────────────────────────────────
def print_report(result: dict, max_tokens: int = MAX_TOKENS):
    total        = result["total"]
    issues       = result["issues"]
    layer_counts = result["layer_counts"]
    token_counts = result["token_counts"]

    print(f"\n{'='*62}")
    print(f"  DATASET VALIDATION REPORT")
    print(f"{'='*62}")
    print(f"  Total examples:     {total:,}")

    # ── Token distribution ────────────────────────────────────────────────
    if token_counts:
        sorted_counts = sorted(token_counts)
        n = len(sorted_counts)
        p50 = sorted_counts[n // 2]
        p90 = sorted_counts[int(n * 0.90)]
        p95 = sorted_counts[int(n * 0.95)]
        p99 = sorted_counts[min(int(n * 0.99), n - 1)]
        avg = sum(sorted_counts) // n

        print(f"\n  Token length distribution:")
        print(f"    Mean:   {avg:>6,}")
        print(f"    p50:    {p50:>6,}")
        print(f"    p90:    {p90:>6,}")
        print(f"    p95:    {p95:>6,}  {'⚠️  near limit' if p95 > max_tokens * 0.85 else ''}")
        print(f"    p99:    {p99:>6,}  {'❌ over limit!' if p99 > max_tokens else ''}")
        print(f"    max:    {max(sorted_counts):>6,}  (limit: {max_tokens:,})")

    # ── Issue counts ──────────────────────────────────────────────────────
    print(f"\n  Quality issues:")
    print(f"    Empty outputs:          {len(issues['empty_output']):>5,}")
    print(f"    Short outputs (<{MIN_OUTPUT_CHARS}c):   {len(issues['short_output']):>5,}")
    print(f"    Over token limit:       {len(issues['over_token_limit']):>5,}  {'❌' if issues['over_token_limit'] else '✅'}")
    print(f"    Near limit (>90%):      {len(issues['truncated_likely']):>5,}")
    print(f"    Duplicates:             {len(issues['duplicates']):>5,}  {'⚠️' if issues['duplicates'] else '✅'}")
    print(f"    Missing layer field:    {len(issues['missing_layer']):>5,}")

    if issues["over_token_limit"]:
        print(f"\n  ⚠️  Top over-limit examples:")
        for idx, n_tok in sorted(issues["over_token_limit"], key=lambda x: -x[1])[:5]:
            print(f"      Line {idx+1:>6}: {n_tok:,} tokens  (limit {max_tokens:,})")

    # ── Layer balance ─────────────────────────────────────────────────────
    print(f"\n  Layer distribution:")
    grand_total = sum(layer_counts.values())
    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
        share = count / max(grand_total, 1)
        flag  = "  ⚠️  UNDER 5%" if share < IDEAL_LAYER_SHARE else ""
        bar   = "█" * int(share * 40)
        print(f"    {layer:<38} {count:>6,}  ({100*share:4.1f}%) {bar}{flag}")

    # ── Verdict ───────────────────────────────────────────────────────────
    severe = (
        len(issues["empty_output"])
        + len(issues["short_output"])
        + len(issues["over_token_limit"])
    )
    print(f"\n  {'─'*58}")
    if severe > total * 0.10:
        print(f"  ❌ VERDICT: NOT READY — {severe:,} examples ({100*severe/max(total,1):.1f}%) fail quality bar.")
        print(f"     Run with --fix to auto-clean before training.")
    elif severe > total * 0.05:
        print(f"  ⚠️  VERDICT: MARGINAL — {severe:,} examples ({100*severe/max(total,1):.1f}%) below quality bar.")
        print(f"     Run with --fix to auto-clean, or re-run build_dataset.py.")
    else:
        print(f"  ✅ VERDICT: READY FOR FINE-TUNING")
        print(f"     {total:,} examples. {len(issues['truncated_likely'])} near token limit (acceptable).")
    print(f"  {'='*62}\n")


# ── Auto-fix ───────────────────────────────────────────────────────────────────
def fix_dataset(pairs: list, path: str, max_tokens: int = MAX_TOKENS, use_tokenizer: bool = True):
    """Drop low-quality examples and rewrite the JSONL file in place."""
    seen:  dict[tuple, bool] = {}
    clean: list[dict]        = []
    dropped_reason: Counter  = Counter()

    for pair in pairs:
        output = pair.get("output", "").strip()
        instr  = pair.get("instruction", "").strip()

        if len(output) < MIN_OUTPUT_CHARS:
            dropped_reason["short_output"] += 1
            continue

        if instr and output == instr:
            dropped_reason["output_equals_instruction"] += 1
            continue

        key = (instr[:150], output[:200])
        if key in seen:
            dropped_reason["duplicate"] += 1
            continue
        seen[key] = True

        n_tok = count_tokens(pair, use_tokenizer=use_tokenizer)
        if n_tok > max_tokens:
            dropped_reason["over_token_limit"] += 1
            continue

        clean.append(pair)

    with open(path, "w", encoding="utf-8") as f:
        for p in clean:
            f.write(json.dumps(p) + "\n")

    total_dropped = sum(dropped_reason.values())
    print(f"\n✅ Fixed dataset written to {path}")
    print(f"   Kept: {len(clean):,}  |  Dropped: {total_dropped:,}")
    for reason, count in sorted(dropped_reason.items(), key=lambda x: -x[1]):
        print(f"     {reason:<32} {count:,}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate training_pairs.jsonl before fine-tuning"
    )
    parser.add_argument(
        "--path", default=DEFAULT_PATH,
        help=f"Path to training pairs JSONL (default: {DEFAULT_PATH})"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"Token budget per example — must match SFTConfig.max_length (default: {MAX_TOKENS})"
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true",
        help="Use fast char heuristic instead of real tokenizer (less accurate)"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Auto-drop bad examples and rewrite the file in place"
    )
    args = parser.parse_args()

    use_tokenizer = not args.no_tokenizer

    print(f"\nLoading: {args.path}")
    pairs = load_pairs(args.path)
    if not pairs:
        return

    print(f"Loaded {len(pairs):,} examples. Running validation...")
    result = validate(pairs, max_tokens=args.max_tokens, use_tokenizer=use_tokenizer)
    print_report(result, max_tokens=args.max_tokens)

    if args.fix:
        print("--fix enabled: cleaning dataset in place...")
        fix_dataset(pairs, args.path, max_tokens=args.max_tokens, use_tokenizer=use_tokenizer)


if __name__ == "__main__":
    main()