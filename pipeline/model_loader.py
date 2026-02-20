"""
pipeline/model_loader.py
------------------------
Loads the fine-tuned vulnerability LLM and exposes ask_model().

CHANGES in this version:
  [MODEL] Updated MODEL_PATH to the Foundation-Sec-8B fine-tuned checkpoint.
    The merged model at HuggingFace Hub replaces the old Mistral-7B checkpoint.
    Foundation-Sec-8B's pre-training on 80B tokens of security-domain text
    means the base model already understands CVE syntax, CWE taxonomy,
    MITRE ATT&CK, and exploit terminology — fine-tuning specializes further.

  [CONTEXT] MAX_PROMPT_CHARS raised to 10000 chars ≈ 3700 tokens.
    Foundation-Sec-8B has an 8192-token context window (vs Mistral's 4096),
    and the fine-tuned model was trained at max_length=4096. This gives
    substantially more room for multi-CVE correlation context in the agent loop.
    Llama 3 tokenizes security text at ~2.7 chars/token:
      10000 chars ≈ 3700 tokens prompt
      +  512 tokens generation
      = ~4212 total  →  well within the 4096 training context

  [BF16] dtype updated to bfloat16.
    Foundation-Sec-8B was fine-tuned with bf16=True. Loading in float16
    can cause mild numerical instability at inference — use bfloat16.

  [LAYER] vulnerability_correlation added to LAYER_CONTEXT.
    The training now emphasizes this layer; inference should expose it
    as a first-class routing option for the agent loop.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

MODEL_PATH = "adityajayashankar/vuln-foundation-sec-8b"
# or local: "./checkpoints/vuln-foundation-sec-8b/merged"

# ── Prompt budget ──────────────────────────────────────────────────────────────
# Llama 3 / Foundation-Sec-8B: 8192-token context, fine-tuned at 4096.
# At ~2.7 chars/token for security text:
#   10000 chars ≈ 3700 tokens prompt
#   +  512 tokens generation
#   = ~4212 total  →  fits within the 4096 fine-tuning window
# Raise MAX_PROMPT_CHARS toward 20000 if you push context window to 8192.
MAX_PROMPT_CHARS = 10000

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
    # NEW — correlation layer now a first-class inference routing option
    "vulnerability_correlation": (
        "You are a vulnerability intelligence analyst. Identify CVEs, CWEs, "
        "ATT&CK techniques, and OWASP categories that co-occur with or are "
        "correlated to the given vulnerability. Reason about exploit chains, "
        "shared affected products, and campaign co-occurrence patterns."
    ),
    "vulnerability_cooccurrence": (
        "You are a threat modeling expert. Given a vulnerability or OWASP category, "
        "identify which other vulnerability classes statistically co-occur in the same "
        "codebase or attack campaign, and explain the co-occurrence pattern."
    ),
    "general": (
        "You are an expert cybersecurity analyst covering vulnerability research, "
        "pentesting, risk scoring, correlation analysis, and remediation."
    ),
}

_pipe = None   # lazy-loaded singleton


def get_pipeline():
    """Lazy-load the Foundation-Sec-8B fine-tuned pipeline (once, then reused)."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"Loading model: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype  = torch.bfloat16,   # bfloat16 — matches fine-tuning dtype
            device_map   = "auto",
            trust_remote_code = True,
        )
        _pipe = hf_pipeline(
            "text-generation",
            model         = model,
            tokenizer     = tokenizer,
            max_new_tokens = 512,
            temperature    = 0.1,
            do_sample      = True,
            repetition_penalty = 1.1,
        )
        print("✅ Foundation-Sec-8B fine-tuned model loaded and ready.")
    except Exception as exc:
        print(f"❌ Model load failed: {exc}")
        _pipe = None

    return _pipe


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """
    Truncate prompt from the LEFT to stay within char budget.
    Preserves the most recent context (tool results, last few turns).
    """
    if len(prompt) <= max_chars:
        return prompt

    truncated = prompt[-max_chars:]
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
    Query the fine-tuned Foundation-Sec-8B vulnerability model.

    Args:
        instruction: The task/question for the model.
        context:     Optional input context (CVE description, agent memory, etc.)
        layer:       Dataset layer key — steers model behaviour via system prompt.
                     Options: vulnerability_intelligence | pentesting_intelligence |
                              risk_scoring | execution_context | audit_evidence |
                              remediation_learning | vulnerability_correlation |
                              vulnerability_cooccurrence | general

    Returns:
        str: Model response text, or a clean error string if inference fails.
    """
    sys_prompt = LAYER_CONTEXT.get(layer, LAYER_CONTEXT["general"])

    if context.strip():
        prompt = (
            f"[SYSTEM]: {sys_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"[SYSTEM]: {sys_prompt}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    # Truncate from left if over budget
    prompt = _truncate_prompt(prompt, max_chars=MAX_PROMPT_CHARS)

    pipe = get_pipeline()
    if pipe is None:
        return "[Model not loaded — check MODEL_PATH and GPU availability]"

    try:
        output = pipe(prompt)[0]["generated_text"]
        if output.startswith(prompt):
            return output[len(prompt):].strip()
        marker = "### Response:\n"
        idx = output.rfind(marker)
        if idx != -1:
            return output[idx + len(marker):].strip()
        return output.strip()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return (
            "[CUDA OOM: prompt was too long even after truncation. "
            "Reduce MAX_PROMPT_CHARS or max_new_tokens.]"
        )
    except RuntimeError as exc:
        return f"[RuntimeError during inference: {exc}]"
    except Exception as exc:
        return f"[Inference failed: {type(exc).__name__}: {exc}]"