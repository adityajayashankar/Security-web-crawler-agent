"""
agent.py
--------
Agentic pipeline loop for vulnerability analysis.
The agent uses the fine-tuned model + tools to reason through a query
step by step, covering all 6 dataset layers as needed.

Imported and run from main.py.
"""

from pipeline.model_loader import ask_model
from pipeline.tools import (
    tool_lookup_cve,
    tool_map_owasp,
    tool_get_pentest_method,
    tool_select_tool,
    tool_fetch_epss,
    tool_score_risk,
    tool_generate_finding,
    tool_get_remediation,
)

# ── Tool registry ──────────────────────────────────────────────────────────
# Format: "tool_name": (function, "short description for the model")
TOOLS = {
    "lookup_cve":         (tool_lookup_cve,         "Fetch CVE details, CWE, and CVSS from NVD. Arg: CVE ID"),
    "map_owasp":          (tool_map_owasp,           "Map vulnerability to OWASP Top 10 category. Arg: description"),
    "get_pentest_method": (tool_get_pentest_method,  "Get attack method, payloads, detection signals. Arg: description"),
    "select_tool":        (tool_select_tool,         "Recommend security testing tool. Arg: OWASP category"),
    "fetch_epss":         (tool_fetch_epss,          "Get EPSS exploit probability score. Arg: CVE ID"),
    "score_risk":         (tool_score_risk,          "Generate full risk assessment. Arg: vulnerability description"),
    "generate_finding":   (tool_generate_finding,    "Generate audit finding report. Args: name|cve|desc|cvss|owasp"),
    "get_remediation":    (tool_get_remediation,     "Get fix recommendation and root cause. Arg: description"),
}

TOOL_MENU = "\n".join([f"  - {k}: {v[1]}" for k, v in TOOLS.items()])

AGENT_SYSTEM_PROMPT = f"""You are a multi-layer cybersecurity audit agent.
You have access to these tools:
{TOOL_MENU}

To call a tool reply: ACTION: tool_name(argument)
When you have enough information reply: FINAL: <your complete answer>
Think step by step. Cover vulnerability details, OWASP mapping, risk scoring, and remediation.
"""

def parse_action(text: str):
    """
    Parse: ACTION: tool_name(arg)
    Returns (tool_name, arg) or (None, None)
    """
    import re
    m = re.search(r"ACTION:\s*(\w+)\((.+)\)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None

def run_agent(user_query: str, max_steps: int = 8, verbose: bool = True) -> str:
    """
    Run the vulnerability analysis agent loop.

    Args:
        user_query: Free-text question or CVE ID to analyze.
        max_steps:  Max reasoning steps before forcing final answer.
        verbose:    Print step-by-step reasoning.

    Returns:
        str: Final analysis report.
    """
    memory = [
        f"[SYSTEM]: {AGENT_SYSTEM_PROMPT}",
        f"[USER]: {user_query}"
    ]

    for step in range(1, max_steps + 1):
        context = "\n".join(memory)

        # Ask model what to do next
        response = ask_model(
            instruction = (
                "Based on the conversation so far, decide the next step. "
                "Call a tool or provide the final answer."
            ),
            context = context,
            layer   = "general"
        )

        if verbose:
            print(f"\n[Step {step}] {response[:200]}{'...' if len(response) > 200 else ''}")

        memory.append(f"[AGENT]: {response}")

        # Check for final answer
        if "FINAL:" in response:
            final = response.split("FINAL:", 1)[1].strip()
            return final

        # Check for tool call
        tool_name, tool_arg = parse_action(response)

        if tool_name and tool_name in TOOLS:
            tool_fn = TOOLS[tool_name][0]

            try:
                # generate_finding takes pipe-separated args: name|cve|desc|cvss|owasp
                if tool_name == "generate_finding":
                    parts = [p.strip() for p in tool_arg.split("|")]
                    parts += [""] * (5 - len(parts))    # pad to 5 args
                    result = tool_fn(*parts[:5])
                else:
                    result = tool_fn(tool_arg)

                if verbose:
                    print(f"  → Tool result: {result[:150]}...")
                memory.append(f"[TOOL {tool_name}]: {result}")

            except Exception as e:
                memory.append(f"[TOOL {tool_name} ERROR]: {e}")

        else:
            # No valid tool call and no FINAL — nudge the model
            memory.append("[SYSTEM]: Please call a tool or provide your FINAL answer.")

    # Fallback: ask for summary of what was gathered
    summary = ask_model(
        instruction = "Summarize all findings gathered so far into a final vulnerability report.",
        context     = "\n".join(memory),
        layer       = "audit_evidence"
    )
    return summary