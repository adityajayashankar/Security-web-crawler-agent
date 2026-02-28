"""
pipeline/model_loader.py
------------------------
LLM backend for the vulnerability agent pipeline.

Backend priority:
  1. Groq       — free, 14,400 req/day, very fast (set GROQ_API_KEY)
  2. OpenRouter — free models as secondary (set OPENROUTER_API_KEY)
  3. Ollama     — local, zero rate limits (install ollama + pull a model)

Get free keys:
  Groq:       https://console.groq.com  (free, no credit card)
  OpenRouter: https://openrouter.ai/keys (free models available)

Ollama local setup (optional, no API key needed):
  winget install Ollama.Ollama
  ollama pull llama3.2        # 2GB, fast
  ollama pull mistral         # 4GB, good quality
"""

import os
import time

# ── Groq models (free tier, generous limits) ──────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality on groq
    "llama-3.1-8b-instant",      # fastest, use when 70b is rate limited
    "mixtral-8x7b-32768",        # good fallback
    "gemma2-9b-it",              # lightweight fallback
]

# ── OpenRouter free models (secondary) ────────────────────────────────────────
OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

# ── Ollama local models (tertiary, zero rate limits) ──────────────────────────
OLLAMA_MODELS = [
    "llama3.2",
    "mistral",
    "llama3.1",
    "phi3",
]

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MAX_RETRIES      = 2
RETRY_BASE_DELAY = 5

# ── Layer system prompts ───────────────────────────────────────────────────────
LAYER_CONTEXT: dict[str, str] = {
    "vulnerability_intelligence": (
        "You are a cybersecurity expert. Analyze vulnerabilities, map them to "
        "OWASP Top 10 categories and CWE IDs, and explain their nature and impact."
    ),
    "pentesting_intelligence": (
        "You are a penetration tester. Describe attack methods, payload examples, "
        "detection signals, and tools used to test for vulnerabilities."
    ),
    "risk_scoring": (
        "You are a risk analyst. Evaluate CVSS scores, EPSS exploit probabilities, "
        "business impact, and prioritize remediation actions."
    ),
    "execution_context": (
        "You are a security architect. Recommend tools and testing approaches "
        "based on the tech stack and deployment context."
    ),
    "audit_evidence": (
        "You are an audit specialist. Generate structured audit findings with "
        "evidence, control gaps, and compliance mapping."
    ),
    "remediation_learning": (
        "You are a security engineer. Provide detailed remediation steps, "
        "root cause analysis, and prevention strategies."
    ),
    "vulnerability_correlation": (
        "You are a threat intelligence analyst specializing in vulnerability "
        "correlation. Given a CVE, identify related vulnerabilities that are "
        "likely to co-exist on the same system or be exploited in the same "
        "attack chain. Use CVE co-occurrence data, CWE families, OWASP "
        "categories, and exploit chain patterns to reason about relationships."
    ),
    "vulnerability_cooccurrence": (
        "You are a threat intelligence analyst. Analyze statistical co-occurrence "
        "patterns between vulnerabilities. Explain which CVEs tend to appear "
        "together in real-world attacks, campaigns, and affected systems. "
        "Distinguish between direct evidence (exploit chains, same campaign) "
        "and inferred relationships (same CWE cluster, same OWASP category)."
    ),
    "general": (
        "You are a multi-layer cybersecurity audit agent. Analyze vulnerabilities, "
        "assess risk, recommend testing approaches, and synthesize findings into "
        "clear, actionable security reports."
    ),
}


# ── Backend 1: Groq ───────────────────────────────────────────────────────────

def _ask_groq(system: str, user: str) -> str | None:
    """Try Groq API. Returns response text or None if unavailable."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        try:
            # Groq uses the same OpenAI-compatible API
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )
            _use_openai_compat = True
        except ImportError:
            print("  ⚠️  Neither groq nor openai package installed. Run: pip install groq")
            return None
    else:
        client = Groq(api_key=api_key)
        _use_openai_compat = False

    for model in GROQ_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=1024,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"  ⏳ groq/{model.split('-')[0]} rate-limited, retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ⚠️  groq/{model} exhausted, trying next...")
                        break
                elif "401" in err or "auth" in err.lower():
                    print("  ❌ Groq auth failed — check GROQ_API_KEY")
                    return None
                elif "404" in err or "does not exist" in err.lower():
                    break
                else:
                    print(f"  ⚠️  Groq error: {e}")
                    break

    return None


# ── Backend 2: OpenRouter ─────────────────────────────────────────────────────

def _ask_openrouter(system: str, user: str) -> str | None:
    """Try OpenRouter free models. Returns response text or None if unavailable."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/vuln-pipeline",
            "X-Title":      "VulnAnalysisAgent",
        },
    )

    for model in OPENROUTER_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=1024,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"  ⏳ openrouter/{model.split('/')[-1]} rate-limited, retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ⚠️  openrouter/{model.split('/')[-1]} exhausted, trying next...")
                        break
                elif "401" in err or "auth" in err.lower():
                    print("  ❌ OpenRouter auth failed — check OPENROUTER_API_KEY")
                    return None
                elif "404" in err:
                    break
                else:
                    print(f"  ⚠️  OpenRouter error: {e}")
                    break

    return None


# ── Backend 3: Ollama (local) ─────────────────────────────────────────────────

def _ask_ollama(system: str, user: str) -> str | None:
    """Try local Ollama. Returns response text or None if unavailable."""
    try:
        import requests
        # Quick availability check
        ping = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if ping.status_code != 200:
            return None

        available = {m["name"].split(":")[0] for m in ping.json().get("models", [])}
        if not available:
            return None

        # Pick first available model from our preferred list
        model = next((m for m in OLLAMA_MODELS if m in available), None)
        if not model:
            model = next(iter(available))  # just use whatever is installed

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    except Exception:
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def ask_model(
    instruction: str,
    context:     str = "",
    layer:       str = "general",
) -> str:
    """
    Query the best available LLM backend.

    Tries in order: Groq → OpenRouter → Ollama
    Returns the first successful response.

    Args:
        instruction: The task/question for the model.
        context:     Optional input context.
        layer:       Selects the system prompt.
    """
    system = LAYER_CONTEXT.get(layer, LAYER_CONTEXT["general"])
    user   = f"{instruction}\n\n{context}".strip() if context.strip() else instruction

    # 1. Try Groq (fastest, most generous free tier)
    if os.environ.get("GROQ_API_KEY"):
        result = _ask_groq(system, user)
        if result:
            return result

    # 2. Try OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        result = _ask_openrouter(system, user)
        if result:
            return result

    # 3. Try local Ollama
    result = _ask_ollama(system, user)
    if result:
        return result

    # Nothing worked
    keys_set = []
    if os.environ.get("GROQ_API_KEY"):      keys_set.append("GROQ_API_KEY ✓")
    if os.environ.get("OPENROUTER_API_KEY"): keys_set.append("OPENROUTER_API_KEY ✓")

    return (
        "[No LLM backend available]\n"
        f"Keys found: {', '.join(keys_set) if keys_set else 'none'}\n"
        "Options:\n"
        "  1. Groq (recommended): https://console.groq.com → set GROQ_API_KEY\n"
        "  2. OpenRouter: https://openrouter.ai/keys → set OPENROUTER_API_KEY\n"
        "  3. Local: install Ollama + run 'ollama pull llama3.2'"
    )