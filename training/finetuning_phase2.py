"""
training/finetune_phase2.py
---------------------------
PHASE 2: Correlation & Co-occurrence Specialist Fine-tuning

Two-phase training rationale
─────────────────────────────
Phase 1 (finetuning.py) gave the model:
  • General security fluency across all 8 layers
  • CVE/CWE/OWASP vocabulary and syntax
  • Alpaca prompt format conditioning

Phase 2 (this script) gives the model:
  • Deep specialization in relational reasoning — the novelty of your dataset
  • Ability to answer P(B|A) co-occurrence questions with calibrated probabilities
  • Multi-signal correlation reasoning (KEV campaign, exploit chain, shared CWE, etc.)
  • Directionality awareness: P(B|A) ≠ P(A|B) in asymmetric co-occurrence pairs

Architecture
────────────
We add a SECOND LoRA adapter on top of the Phase 1 merged model.
This is the correct approach (vs retraining the Phase 1 adapter) because:
  1. Phase 1 knowledge is frozen — no catastrophic forgetting of general security fluency
  2. Phase 2 adapter learns only the delta: relational/graph reasoning on top
  3. At inference time, both adapters are merged — full fluency + specialist reasoning

Phase 2 LoRA uses r=16 (half of Phase 1 r=32).
Phase 1 already specialized MLP layers heavily. Phase 2 focuses the ADDITIONAL
adapter on attention patterns (q,k,v,o) — correlation reasoning is fundamentally
about learning which entities to attend to together.

Hyperparameters
───────────────
  LR: 5e-5 (4× lower than Phase 1's 2e-4)
    Phase 1 moved the model far along the loss landscape. A large LR in Phase 2
    would overshoot the correlation-specific minima. 5e-5 is the "fine" in
    fine-tuning-the-fine-tuning.

  Epochs: 5 (vs 3 in Phase 1)
    Correlation/co-occurrence pairs are sparse but high-density signal.
    More passes are needed to consolidate relational structure.

  Gradient accumulation: 8 (vs 16 in Phase 1)
    Smaller dataset → fewer gradient accumulation steps needed to simulate
    a reasonable batch size. Effective batch = 1×8 = 8.

  No warmup (warmup_steps=0):
    We're starting from a well-trained Phase 1 checkpoint. Warmup would
    waste steps ramping up LR before it can do useful work.

Memory: ~14GB on A100 (Phase 1 merged model in bfloat16 + small Phase 2 adapter)
"""

import json
import re
import random
import torch
from pathlib import Path
from collections import defaultdict

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

# ── Config ─────────────────────────────────────────────────────────────────────
# Load from the Phase 1 merged checkpoint (or HF Hub after phase 1 push)
PHASE1_MODEL   = "./checkpoints/vuln-foundation-sec-8b/merged"
# fallback: PHASE1_MODEL = "adityajayashankar/vuln-foundation-sec-8b"

DATASET_PATH   = Path("data") / "training_pairs.jsonl"
OUTPUT_DIR     = Path("./checkpoints/vuln-foundation-sec-8b-phase2")
HF_REPO_NAME   = "adityajayashankar/vuln-foundation-sec-8b-correlation"

# Only these two layers go into Phase 2 training
PHASE2_LAYERS  = {"vulnerability_correlation", "vulnerability_cooccurrence"}

EVAL_FRACTION  = 0.10    # 10% eval — higher than phase 1 since dataset is smaller
MIN_EVAL       = 30

# ── Prompt formatter (mirrors finetuning.py exactly) ──────────────────────────
def format_example(example: dict) -> dict:
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if inp:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text, "layer": example.get("layer", "general")}


# ── Load ONLY correlation + co-occurrence pairs ────────────────────────────────
def load_phase2_pairs(path: Path) -> list[dict]:
    """
    Filter training_pairs.jsonl to only the layers that are
    the novelty of this dataset. Logs what gets filtered in/out.
    """
    all_pairs: list[dict] = []
    layer_counts: dict[str, int] = defaultdict(int)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                layer_counts[p.get("layer", "unknown")] += 1
                all_pairs.append(p)
            except json.JSONDecodeError:
                continue

    phase2_pairs = [p for p in all_pairs if p.get("layer") in PHASE2_LAYERS]

    print(f"\n  Full dataset: {len(all_pairs):,} pairs across {len(layer_counts)} layers")
    print(f"  Phase 2 filter: keeping only {PHASE2_LAYERS}")
    print(f"  Phase 2 pairs: {len(phase2_pairs):,}")
    print(f"\n  Layer breakdown of Phase 2 pairs:")
    for layer in PHASE2_LAYERS:
        count = sum(1 for p in phase2_pairs if p.get("layer") == layer)
        print(f"    {layer:<38} {count:>6,} pairs")

    if len(phase2_pairs) < 200:
        print(
            f"\n  ⚠️  WARNING: Only {len(phase2_pairs)} Phase 2 pairs found.\n"
            f"      Run: python run_pipeline.py --correlate\n"
            f"      to regenerate correlation + co-occurrence data before Phase 2."
        )

    return phase2_pairs


# ── CVE-decontaminated split ───────────────────────────────────────────────────────

_CVE_RE = re.compile(r'(CVE-\d{4}-\d{4,}|GHSA-[a-z0-9-]+)')

def _extract_cve(pair: dict) -> str:
    cid = pair.get("cve_id", "")
    if cid:
        return cid
    m = _CVE_RE.search(pair.get("instruction", ""))
    return m.group(1) if m else ""


def stratified_split(
    pairs: list[dict],
    eval_fraction: float = EVAL_FRACTION,
    min_eval: int = MIN_EVAL,
) -> tuple[list[dict], list[dict]]:
    """
    CVE-aware split: all pairs for a given CVE go to train OR eval, never both.
    Prevents eval recall inflation from info leakage.
    """
    cve_pairs: dict[str, list[dict]] = defaultdict(list)
    no_cve: list[dict] = []

    for p in pairs:
        cve = _extract_cve(p)
        if cve:
            cve_pairs[cve].append(p)
        else:
            no_cve.append(p)

    cve_ids = list(cve_pairs.keys())
    random.shuffle(cve_ids)
    total_with_cve = sum(len(ps) for ps in cve_pairs.values())
    target_eval = max(min_eval, int(total_with_cve * eval_fraction))

    eval_cves: set[str] = set()
    eval_count = 0
    for cve in cve_ids:
        if eval_count >= target_eval:
            break
        eval_cves.add(cve)
        eval_count += len(cve_pairs[cve])

    train_pairs, eval_pairs = [], []
    for cve, ps in cve_pairs.items():
        if cve in eval_cves:
            eval_pairs.extend(ps)
        else:
            train_pairs.extend(ps)

    random.shuffle(no_cve)
    n_eval_nc = max(1, int(len(no_cve) * eval_fraction)) if no_cve else 0
    eval_pairs.extend(no_cve[:n_eval_nc])
    train_pairs.extend(no_cve[n_eval_nc:])

    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)

    train_cves = {_extract_cve(p) for p in train_pairs} - {""}
    eval_cves_actual = {_extract_cve(p) for p in eval_pairs} - {""}
    overlap = train_cves & eval_cves_actual
    print(f"\n  CVE-decontaminated Phase 2 split: {len(train_pairs):,} train  /  {len(eval_pairs):,} eval")
    print(f"    Train CVEs: {len(train_cves):,}  |  Eval CVEs: {len(eval_cves_actual):,}  |  Overlap: {len(overlap)}")
    return train_pairs, eval_pairs


def pairs_to_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list([format_example(p) for p in pairs])


# ── Load Phase 1 model for Phase 2 ────────────────────────────────────────────
def load_phase1_model_for_phase2():
    """
    Load the Phase 1 merged model in 4-bit QLoRA, ready to receive
    a new Phase 2 LoRA adapter on top.

    We load the merged (not adapter) checkpoint so Phase 1 weights are
    frozen in the 4-bit quantized base — only Phase 2 adapter trains.
    """
    print(f"\nLoading Phase 1 merged model: {PHASE1_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        PHASE1_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        attn_implementation = "flash_attention_2",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # Re-add pad token in case it was lost during merge
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# ── Phase 2 LoRA config — attention-focused ───────────────────────────────────
def get_phase2_lora_config() -> LoraConfig:
    """
    Phase 2 LoRA: r=16, attention projections only.

    Rationale for attention-only in Phase 2:
      Phase 1's MLP LoRA (r=32) already injected factual CVE/CWE/ATT&CK
      knowledge into the feed-forward layers (where LLMs store facts).
      Correlation REASONING is about learning which tokens/entities to
      attend to simultaneously — that's an attention-level operation.

      Specifically: to answer "what co-occurs with CVE-X?", the model
      needs to learn to jointly attend to the CVE token, its CWE, its
      OWASP category, and the product tokens at the same time.
      That's q/k/v reprogramming, not MLP fact storage.

    r=16 (half of Phase 1):
      Phase 2 is a refinement pass, not a reconstruction.
      We're adding a small specialist adapter on top of general knowledge.
      Larger rank would risk overwriting Phase 1's fluency.

    adapter_name="phase2_correlation":
      Named adapter so both can be tracked, merged selectively, or
      ablated independently during evaluation.
    """
    return LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        bias           = "none",
        target_modules = [
            # Attention only — correlation reasoning is attention-level
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Phase 2: Correlation & Co-occurrence Specialist Training")
    print("=" * 60)

    # ── Load Phase 2 data ──────────────────────────────────────────────────
    pairs = load_phase2_pairs(DATASET_PATH)
    if not pairs:
        print("❌ No Phase 2 pairs found. Aborting.")
        return

    train_pairs, eval_pairs = stratified_split(pairs)
    train_dataset = pairs_to_dataset(train_pairs)
    eval_dataset  = pairs_to_dataset(eval_pairs)

    # ── Load Phase 1 model + add Phase 2 LoRA ─────────────────────────────
    model, tokenizer = load_phase1_model_for_phase2()

    phase2_lora = get_phase2_lora_config()
    model = get_peft_model(model, phase2_lora)
    model.print_trainable_parameters()

    # Verify: only Phase 2 LoRA params should be trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    print(f"  (Should be ~43M — only Phase 2 attention LoRA adapters)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = 5,          # more passes — smaller focused dataset
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,          # effective batch = 8 (smaller data)
        gradient_checkpointing      = True,
        optim                       = "paged_adamw_8bit",
        learning_rate               = 5e-5,       # 4× lower than Phase 1 — fine-grained
        lr_scheduler_type           = "cosine",
        warmup_steps                = 0,          # already well-trained — skip warmup
        bf16                        = True,
        logging_steps               = 20,
        logging_strategy            = "steps",
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_steps                  = 100,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        max_length                  = 4096,
        dataset_text_field          = "text",
        report_to                   = "none",
    )

    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        processing_class = tokenizer,
    )

    print(f"\n🚀 Starting Phase 2 fine-tuning ({len(train_pairs):,} correlation pairs)...")
    trainer.train()

    # ── Save Phase 2 adapter + merged model ───────────────────────────────
    adapter_dir = OUTPUT_DIR / "phase2_adapter"
    merged_dir  = OUTPUT_DIR / "merged"

    trainer.save_model(str(adapter_dir))
    print(f"\n✅ Phase 2 LoRA adapter saved → {adapter_dir}")

    print("\nMerging Phase 2 LoRA into Phase 1 model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"✅ Final merged model saved → {merged_dir}")
    print(f"\n   This model has: Phase 1 (general security) + Phase 2 (correlation specialist)")

    # ── Push to Hub ────────────────────────────────────────────────────────
    print(f"\nPushing to HuggingFace Hub: {HF_REPO_NAME}")
    from huggingface_hub import login
    login()
    merged.push_to_hub(HF_REPO_NAME)
    tokenizer.push_to_hub(HF_REPO_NAME)
    print(f"🚀 Correlation specialist live: https://huggingface.co/{HF_REPO_NAME}")
    print(f"\n  Next step: python eval/probe_cooccurrence.py --model {HF_REPO_NAME}")


if __name__ == "__main__":
    main()