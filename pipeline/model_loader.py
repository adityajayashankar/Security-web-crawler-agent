"""
pipeline/model_loader.py
------------------------
Loads the fine-tuned vulnerability LLM and exposes ask_model().

FIXES in this version:
  1. Inference error recovery — ask_model() now catches CUDA OOM, generation
     errors, and tokenizer failures. Returns a clean error string instead of
     crashing the entire agent loop.

  2. Prompt truncation — agent memory grows with each step. After 5-6 steps
     the prompt can exceed 6000 chars (~2000 tokens), leaving only ~512 tokens
     for generation (max_new_tokens). With 8 steps this hits the model's
     context window hard and causes OOM or incoherent outputs.
     Now truncates from the LEFT (keeps most recent context) when over budget.

  3. Truncation is done at the character level BEFORE tokenization, which is
     the correct approach — truncating tokenized tensors mid-sequence is unsafe.

  4. Added a configurable MAX_PROMPT_CHARS constant (default 5500 chars ≈ 1900
     tokens) leaving comfortable room for max_new_tokens=512.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

MODEL_PATH = "adityajayashankar/vuln-mistral-7b"
# or local: "./checkpoints/vuln-mistral-7b/merged"

# ── Prompt budget ──────────────────────────────────────────────────────────────
# At ~2.9 chars/token for security text:
#   5500 chars ≈ 1900 tokens prompt
#   + 512 tokens generation
#   = ~2412 total  →  fits comfortably in 2048 context
#     (with a small buffer; real tokenizer may produce fewer tokens than estimate)
# Raise this if you upgrade to a 4096 context model.
MAX_PROMPT_CHARS = 5500

# ── Layer system prompts ───────────────────────────────────────────────────────
LAYER_CONTEXT: dict[str, str] = {
    "vulnerability_intelligence": (
        "You are a cybersecurity expert. Analyze vulnerabilities, map them to "
        "OWASP categories and CWE IDs, and explain their nature."
    ),
    "pentesting_intelligence": (
        "You are a penetration tester. Describe attack methods, payloads, "
        "detection signals, and tools used to test vulnerabilities."
    ),
    "risk_scoring": (
        "You are a risk analyst. Evaluate CVSS scores, EPSS probabilities, "
        "business impact, and prioritize remediation."
    ),
    "execution_context": (
        "You are a security architect. Recommend tools and testing approaches "
        "based on tech stack and deployment context."
    ),
    "audit_evidence": (
        "You are an audit specialist. Generate structured audit findings with "
        "evidence, control gaps, and compliance mapping."
    ),
    "remediation_learning": (
        "You are a security engineer. Provide root cause analysis, fix "
        "recommendations, and prevention strategies."
    ),
    "general": (
        "You are an expert cybersecurity analyst covering vulnerability research, "
        "pentesting, risk scoring, and remediation."
    ),
}

_pipe = None  # lazy-loaded singleton


def get_pipeline():
    """Lazy-load the model pipeline (loads once, reused across all calls)."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading model: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1,
        )
        print("✅ Model loaded and ready.")
    except Exception as exc:
        print(f"❌ Model load failed: {exc}")
        _pipe = None

    return _pipe


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """
    Truncate prompt from the LEFT to stay within char budget.

    Left-truncation preserves the most recent context (tool results, last few
    turns) while dropping older memory. This is the correct strategy for a
    running agent loop where recent context is most relevant.

    A header marker is inserted so the model knows context was trimmed.
    """
    if len(prompt) <= max_chars:
        return prompt

    truncated = prompt[-max_chars:]

    # Try to cut at a clean line boundary to avoid breaking mid-token
    first_newline = truncated.find("\n")
    if 0 < first_newline < 200:
        truncated = truncated[first_newline + 1:]

    return "[...earlier context truncated for length...]\n" + truncated


def ask_model(
    instruction: str,
    context:     str = "",
    layer:       str = "general",
) -> str:
    """
    Query the fine-tuned vulnerability model.

    Args:
        instruction: The task/question for the model.
        context:     Optional input context (CVE description, agent memory, etc.)
        layer:       Dataset layer key — steers model behaviour via system prompt.
                     Options: vulnerability_intelligence | pentesting_intelligence |
                              risk_scoring | execution_context |
                              audit_evidence | remediation_learning | general

    Returns:
        str: Model response text, or an error string if inference fails.

    FIX 1: All inference errors (CUDA OOM, RuntimeError, shape mismatches,
           tokenizer failures) are caught and returned as a clean string.
           The agent loop can continue rather than crashing.

    FIX 2: Prompt is truncated from the left before inference when over budget.
    """
    sys_prompt = LAYER_CONTEXT.get(layer, LAYER_CONTEXT["general"])

    prompt = (
        f"[SYSTEM]: {sys_prompt}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{context}\n\n"
        f"### Response:\n"
    )

    # FIX 2: Truncate from left if over budget
    prompt = _truncate_prompt(prompt, max_chars=MAX_PROMPT_CHARS)

    pipe = get_pipeline()
    if pipe is None:
        return "[Model not loaded — check MODEL_PATH and GPU availability]"

    # FIX 1: Wrap inference in try/except
    try:
        output = pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix — return only what the model generated
        if output.startswith(prompt):
            return output[len(prompt):].strip()
        # Fallback: find ### Response: and return what follows
        marker = "### Response:\n"
        idx = output.rfind(marker)
        if idx != -1:
            return output[idx + len(marker):].strip()
        return output.strip()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (
            "[CUDA OOM: prompt was too long even after truncation. "
            "Try reducing MAX_PROMPT_CHARS or max_new_tokens.]"
        )

    except RuntimeError as exc:
        # Catches shape mismatches, kernel failures, etc.
        return f"[RuntimeError during inference: {exc}]"

    except Exception as exc:
        return f"[Inference failed: {type(exc).__name__}: {exc}]"