"""
model_loader.py
---------------
Loads the fine-tuned vulnerability LLM and exposes ask_model().
Imported by agent.py and tools.py.

The model was trained on 6 layers of vulnerability data.
ask_model() accepts an optional 'layer' hint to set context for the model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = "your-username/vuln-mistral-7b"  # or local: "./checkpoints/vuln-mistral-7b/merged"

# ── Layer system prompts ───────────────────────────────────────────────────
# Match the layers in training_pairs.jsonl so the model knows its role.
LAYER_CONTEXT = {
    "vulnerability_intelligence": "You are a cybersecurity expert. Analyze vulnerabilities, map them to OWASP categories and CWE IDs, and explain their nature.",
    "pentesting_intelligence":    "You are a penetration tester. Describe attack methods, payloads, detection signals, and tools used to test vulnerabilities.",
    "risk_scoring":               "You are a risk analyst. Evaluate CVSS scores, EPSS probabilities, business impact, and prioritize remediation.",
    "execution_context":          "You are a security architect. Recommend tools and testing approaches based on tech stack and deployment context.",
    "audit_evidence":             "You are an audit specialist. Generate structured audit findings with evidence, control gaps, and compliance mapping.",
    "remediation_learning":       "You are a security engineer. Provide root cause analysis, fix recommendations, and prevention strategies.",
    "general":                    "You are an expert cybersecurity analyst covering vulnerability research, pentesting, risk scoring, and remediation."
}

_pipe = None

def get_pipeline():
    """Lazy-load the model pipeline (loads once, reused across all calls)."""
    global _pipe
    if _pipe is None:
        print(f"Loading model: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1
        )
        print("✅ Model loaded and ready.")
    return _pipe

def ask_model(
    instruction: str,
    context:     str  = "",
    layer:       str  = "general"
) -> str:
    """
    Query the fine-tuned vulnerability model.

    Args:
        instruction: The task/question for the model.
        context:     Optional input context (CVE description, code snippet, etc.)
        layer:       One of the 6 dataset layers — steers model behavior.
                     Options: vulnerability_intelligence | pentesting_intelligence |
                              risk_scoring | execution_context |
                              audit_evidence | remediation_learning | general

    Returns:
        str: Model response text.
    """
    sys_prompt = LAYER_CONTEXT.get(layer, LAYER_CONTEXT["general"])

    prompt = (
        f"[SYSTEM]: {sys_prompt}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{context}\n\n"
        f"### Response:\n"
    )

    pipe   = get_pipeline()
    output = pipe(prompt)[0]["generated_text"]

    # Strip prompt from output — return only what the model generated
    return output[len(prompt):].strip()