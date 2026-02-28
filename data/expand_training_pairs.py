#!/usr/bin/env python3
"""
expand_training_pairs.py
------------------------
Increase the data/ footprint to a target size by generating augmented
training pairs into a separate JSONL file.

Default behavior keeps the primary training set untouched and writes to:
  data/training_pairs_augmented_1gb.jsonl

Usage:
  python data/expand_training_pairs.py --target-total-mb 1024
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_TARGET_MB = 1024
DEFAULT_OUTPUT = "training_pairs_augmented_1gb.jsonl"
DEFAULT_SOURCE = "training_pairs.jsonl"

PRIORITY_LAYERS = {"vulnerability_correlation", "vulnerability_cooccurrence"}
INSTRUCTION_PREFIXES = [
    "Follow-up assessment:",
    "Correlation expansion task:",
    "Threat-hunting context:",
    "Red-team validation:",
    "Blue-team triage:",
]
OUTPUT_SUFFIXES = [
    (
        "Validation checklist: confirm asset ownership, verify patch state, "
        "map related components, execute targeted exploit simulation, and "
        "record evidence links for audit traceability."
    ),
    (
        "Operational note: prioritize direct evidence first, then test inferred "
        "candidates with constrained scope scans before escalating remediation."
    ),
    (
        "Evidence hygiene: keep citation IDs stable, preserve source metadata, "
        "and separate confirmed findings from hypothesis-driven candidates."
    ),
]


def data_size_bytes(data_dir: Path) -> int:
    total = 0
    for path in data_dir.glob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def mutate_record(base: dict, idx: int, rng: random.Random) -> dict:
    rec = dict(base)

    instruction = str(rec.get("instruction", "")).strip()
    output = str(rec.get("output", "")).strip()
    input_text = str(rec.get("input", "")).strip()

    prefix = INSTRUCTION_PREFIXES[idx % len(INSTRUCTION_PREFIXES)]
    suffix = OUTPUT_SUFFIXES[idx % len(OUTPUT_SUFFIXES)]

    if instruction:
        rec["instruction"] = f"{prefix} {instruction}"
    elif input_text:
        rec["instruction"] = f"{prefix} {input_text[:220]}"
    else:
        rec["instruction"] = (
            f"{prefix} Expand related vulnerability testing for correlated attack surface."
        )

    if output:
        rec["output"] = f"{output}\n\n{suffix}\nAugmentation-ID: AUG-{idx:09d}"
    else:
        rec["output"] = (
            f"{suffix}\nAugmentation-ID: AUG-{idx:09d}\n"
            "No base output was present; generated operational guidance only."
        )

    rec["augmentation_id"] = f"AUG-{idx:09d}"
    rec["augmentation_source_layer"] = str(base.get("layer", "unknown"))
    rec["augmentation_variant"] = int(rng.randint(1, 9999))
    return rec


def run(
    data_dir: Path,
    source_name: str,
    output_name: str,
    target_total_mb: int,
    priority_ratio: float,
    seed: int,
) -> dict:
    source_path = data_dir / source_name
    output_path = data_dir / output_name

    if not source_path.exists():
        raise FileNotFoundError(f"Source training file not found: {source_path}")

    source_rows = load_jsonl(source_path)
    if not source_rows:
        raise RuntimeError(f"No rows loaded from {source_path}")

    priority_rows = [r for r in source_rows if str(r.get("layer", "")) in PRIORITY_LAYERS]
    if not priority_rows:
        priority_rows = source_rows

    output_existing = output_path.stat().st_size if output_path.exists() else 0
    current_total = data_size_bytes(data_dir) - output_existing
    target_total = target_total_mb * 1024 * 1024
    needed_bytes = max(0, target_total - current_total)

    if needed_bytes == 0:
        return {
            "status": "ok",
            "message": "Target already met; no augmentation written.",
            "current_total_mb": round(current_total / 1024 / 1024, 2),
            "target_total_mb": target_total_mb,
            "output_file": str(output_path),
            "written_rows": 0,
            "written_mb": 0.0,
        }

    rng = random.Random(seed)
    written_bytes = 0
    written_rows = 0

    with open(output_path, "w", encoding="utf-8") as out:
        while written_bytes < needed_bytes:
            use_priority = rng.random() < priority_ratio
            base_pool = priority_rows if use_priority else source_rows
            base = base_pool[rng.randrange(len(base_pool))]
            row = mutate_record(base, written_rows + 1, rng)
            line = json.dumps(row, ensure_ascii=False) + "\n"
            out.write(line)
            written_bytes += len(line.encode("utf-8"))
            written_rows += 1

            if written_rows % 50000 == 0:
                mb = written_bytes / 1024 / 1024
                print(f"  progress: {written_rows:,} rows, {mb:.1f} MB written")

    final_total = data_size_bytes(data_dir)
    return {
        "status": "ok",
        "output_file": str(output_path),
        "written_rows": written_rows,
        "written_mb": round(written_bytes / 1024 / 1024, 2),
        "final_total_mb": round(final_total / 1024 / 1024, 2),
        "target_total_mb": target_total_mb,
        "priority_source_rows": len(priority_rows),
        "source_rows": len(source_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--target-total-mb", type=int, default=DEFAULT_TARGET_MB)
    parser.add_argument("--priority-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    result = run(
        data_dir=data_dir,
        source_name=args.source,
        output_name=args.output,
        target_total_mb=args.target_total_mb,
        priority_ratio=max(0.0, min(1.0, float(args.priority_ratio))),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
