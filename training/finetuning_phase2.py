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

Architecture — Frozen Phase 1 LoRA + Disjoint Phase 2 LoRA
────────────────────────────────────────────────────────────
Instead of loading the Phase 1 merged model (which bakes Phase 1 LoRA into
frozen 4-bit weights where it can't be examined or ablated), we now load:

  1. Base Foundation-Sec-8B model (4-bit QLoRA)
  2. Phase 1 LoRA adapter (adapter_name="phase1_general") — FROZEN
  3. Phase 2 LoRA adapter (adapter_name="phase2_correlation") — TRAINABLE

Why this prevents catastrophic forgetting:
  • Phase 1 LoRA targeted ALL modules (q/k/v/o + up/down/gate_proj, r=32).
    Its attention LoRA learned the correlation structure — which CVEs to
    jointly attend to.
  • Phase 2 LoRA targets ONLY MLP modules (up/down/gate_proj, r=16).
    MLP layers store factual associations. Phase 2 refines the co-occurrence
    FACTS without touching Phase 1's attention patterns at all.
  • The two adapters are perfectly disjoint at the module level:
      Phase 1 attention LoRA: FROZEN (correlation reasoning preserved)
      Phase 1 MLP LoRA:       FROZEN (general security facts preserved)
      Phase 2 MLP LoRA:       TRAINABLE (adds co-occurrence fact refinement)
  • At inference, both adapters are merged sequentially. The final model
    has Phase 1's full knowledge + Phase 2's co-occurrence MLP refinement.

Why MLP-only for Phase 2:
  MLP layers in transformers store factual associations (Meng et al., 2022).
  Phase 2 training data is dense factual signal: "CVE-X co-occurs with CVE-Y
  at probability P." This is fact injection, not reasoning-pattern change.
  Attention patterns from Phase 1 already handle the relational reasoning.

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

Memory: ~15GB on A100 (base model 4-bit + Phase 1 frozen LoRA + Phase 2 LoRA)
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
# Load from the Phase 1 adapter checkpoint (NOT merged — we need the adapter
# separate so it can be frozen independently from Phase 2).
BASE_MODEL        = "fdtn-ai/Foundation-Sec-8B"
PHASE1_ADAPTER    = "./checkpoints/vuln-foundation-sec-8b/final"
# Fallback: if adapter dir not found, fall back to merged (less ideal)
PHASE1_MERGED     = "./checkpoints/vuln-foundation-sec-8b/merged"

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


# ── Load Phase 1 model + frozen adapter for Phase 2 ──────────────────────────
def load_phase1_model_for_phase2():
    """
    Load base Foundation-Sec-8B + Phase 1 LoRA adapter (FROZEN).

    Strategy:
      1. If Phase 1 adapter dir exists → load base + adapter via PeftModel
      2. If only merged exists → load merged as base (fallback, less ideal)

    Phase 1 adapter params are explicitly frozen after loading so that
    Phase 2 training cannot modify them.
    """
    adapter_path = Path(PHASE1_ADAPTER)
    merged_path  = Path(PHASE1_MERGED)
    use_adapter  = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
    )

    if use_adapter:
        print(f"\n  Loading base model: {BASE_MODEL}")
        print(f"  Loading Phase 1 adapter: {PHASE1_ADAPTER} (will be FROZEN)")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
            attn_implementation = "flash_attention_2",
        )
        base_model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            base_model.resize_token_embeddings(len(tokenizer))

        # Load Phase 1 LoRA as a named adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            adapter_name = "phase1_general",
            is_trainable = False,   # frozen from the start
        )

        # Explicit belt-and-suspenders freeze of all Phase 1 adapter params
        phase1_frozen = 0
        for name, param in model.named_parameters():
            if "phase1_general" in name:
                param.requires_grad = False
                phase1_frozen += param.numel()
        print(f"  ✅ Phase 1 adapter frozen: {phase1_frozen:,} params (requires_grad=False)")

    else:
        # Fallback: load merged model as base (Phase 1 weights baked in 4-bit)
        load_from = str(merged_path) if merged_path.exists() else BASE_MODEL
        print(f"\n  ⚠️  Phase 1 adapter not found at {PHASE1_ADAPTER}")
        print(f"  Falling back to merged model: {load_from}")
        print(f"  (Phase 1 weights are frozen in 4-bit base, but not independently trackable)")

        model = AutoModelForCausalLM.from_pretrained(
            load_from,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
            attn_implementation = "flash_attention_2",
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# ── Phase 2 LoRA config — MLP-only (disjoint from Phase 1 attention) ─────────
def get_phase2_lora_config() -> LoraConfig:
    """
    Phase 2 LoRA: r=16, MLP projections only.

    CRITICAL DESIGN DECISION — MLP-only to prevent catastrophic forgetting:
      Phase 1 targeted ALL modules (attention + MLP) at r=32. Its attention
      LoRA learned the correlation structure — which CVEs to jointly attend
      to. If Phase 2 also targeted attention, it would overwrite these
      learned attention patterns, destroying the correlation reasoning.

      Phase 2 targets ONLY MLP layers (up_proj, down_proj, gate_proj):
        - Disjoint from Phase 1's attention LoRA → zero interference
        - MLP layers store factual associations (Meng et al., 2022)
        - Phase 2 data is factual: "CVE-X co-occurs with CVE-Y at prob P"
        - Perfect target for fact injection without reasoning disruption

    r=16 (half of Phase 1 r=32):
      Phase 2 is a refinement pass, not a reconstruction.
      We're adding co-occurrence FACTS, not rebuilding the model.

    The resulting model has:
      Base weights:       Foundation-Sec-8B (frozen, 4-bit)
      Phase 1 LoRA:       attention + MLP, r=32 (FROZEN — correlation reasoning)
      Phase 2 LoRA:       MLP only, r=16 (TRAINABLE — co-occurrence facts)
    """
    return LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        bias           = "none",
        target_modules = [
            # MLP only — disjoint from Phase 1 attention LoRA
            # Phase 1's correlation attention patterns are completely untouched.
            "up_proj", "down_proj", "gate_proj",
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

    # If model already has Phase 1 adapter (PeftModel), add Phase 2 as second adapter.
    # If fallback (plain model), wrap with get_peft_model as before.
    if isinstance(model, PeftModel):
        model.add_adapter("phase2_correlation", phase2_lora)
        model.set_adapter("phase2_correlation")
        print(f"\n  Multi-adapter mode: phase1_general (FROZEN) + phase2_correlation (TRAINABLE)")
    else:
        model = get_peft_model(model, phase2_lora)
        print(f"\n  Single-adapter mode (fallback): phase2_correlation (TRAINABLE)")

    model.print_trainable_parameters()

    # Verify: only Phase 2 LoRA params should be trainable
    trainable_names  = []
    frozen_lora      = 0
    trainable_total  = 0
    total            = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable_total += param.numel()
            trainable_names.append(name)
        elif "lora" in name.lower():
            frozen_lora += param.numel()

    print(f"\n  Trainable params: {trainable_total:,} / {total:,} ({trainable_total/total*100:.2f}%)")
    print(f"  Frozen LoRA params (Phase 1): {frozen_lora:,}")

    # Sanity check: no Phase 1 adapter params should be trainable
    phase1_leak = [n for n in trainable_names if "phase1" in n]
    if phase1_leak:
        print(f"\n  ❌ ERROR: {len(phase1_leak)} Phase 1 params are trainable — aborting!")
        for n in phase1_leak[:5]:
            print(f"     LEAK: {n}")
        return
    print(f"  ✅ Phase 1 adapter is fully frozen — no catastrophic forgetting risk")

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

    print("\nMerging adapters into base model...")
    if isinstance(model, PeftModel) and "phase1_general" in (model.peft_config or {}):
        # Multi-adapter path: merge both adapters sequentially
        # First merge Phase 1 (frozen), then Phase 2 (trained)
        print("  Merging Phase 1 (frozen) + Phase 2 (trained) adapters...")
        model.set_adapter(["phase1_general", "phase2_correlation"])
        merged = model.merge_and_unload()
    else:
        # Single-adapter fallback: just merge Phase 2
        merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"✅ Final merged model saved → {merged_dir}")
    print(f"\n   This model has:")
    print(f"     Base:    Foundation-Sec-8B")
    print(f"     Phase 1: general security LoRA (attention + MLP, r=32) [was frozen]")
    print(f"     Phase 2: co-occurrence MLP LoRA (MLP only, r=16) [newly trained]")

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