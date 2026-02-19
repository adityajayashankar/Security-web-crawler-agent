"""
pipeline/agents.py
------------------
Agentic pipeline loop for vulnerability analysis.

FIXES in this version:
  1. parse_action() was case-sensitive and only matched "ACTION:" exactly.
     LLM outputs commonly drift to "Action:", "TOOL:", "Tool:", "CALL:" etc.
     Now uses case-insensitive regex that accepts all common synonyms.

  2. generate_finding used pipe "|" as arg delimiter. CVE descriptions and
     advisory text frequently contain "|" (log lines, pipe-separated lists,
     regex patterns), causing wrong argument unpacking silently.
     Fixed: generate_finding now accepts a JSON string as its single arg,
     which is unambiguous regardless of content.

  3. FINAL: extraction was fragile — model sometimes outputs
     "FINAL ANSWER:" or "FINAL RESPONSE:" and the check failed.
     Now handles all common variants.

  4. Agent loop now tracks whether any tools were actually called and
     produces a meaningful summary even when the model never called FINAL:.
"""

import re
import json

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


# ── Tool registry ──────────────────────────────────────────────────────────────
# FIX 2: generate_finding description updated to use JSON arg format.
TOOLS = {
    "lookup_cve": (
        tool_lookup_cve,
        "Fetch CVE details, CWE, and CVSS from NVD. Arg: CVE-ID string"
    ),
    "map_owasp": (
        tool_map_owasp,
        "Map vulnerability to OWASP Top 10 category. Arg: description string"
    ),
    "get_pentest_method": (
        tool_get_pentest_method,
        "Get attack method, payloads, detection signals. Arg: vulnerability description"
    ),
    "select_tool": (
        tool_select_tool,
        "Recommend security testing tool. Arg: OWASP category string"
    ),
    "fetch_epss": (
        tool_fetch_epss,
        "Get EPSS exploit probability score. Arg: CVE-ID string"
    ),
    "score_risk": (
        tool_score_risk,
        "Generate full risk assessment. Arg: vulnerability description"
    ),
    "generate_finding": (
        tool_generate_finding,
        # FIX 2: JSON arg format avoids pipe-in-content bugs
        'Generate audit finding report. Arg: JSON string — {"name":"...","cve":"...","desc":"...","cvss":"...","owasp":"..."}'
    ),
    "get_remediation": (
        tool_get_remediation,
        "Get fix recommendation and root cause. Arg: vulnerability description"
    ),
}

TOOL_NAMES = set(TOOLS.keys())
TOOL_MENU  = "\n".join([f"  - {k}: {v[1]}" for k, v in TOOLS.items()])

AGENT_SYSTEM_PROMPT = f"""You are a multi-layer cybersecurity audit agent.
You have access to these tools:
{TOOL_MENU}

To call a tool, reply on its own line:
  ACTION: tool_name(argument)

When you have gathered enough information, reply:
  FINAL: <your complete vulnerability analysis report>

Think step by step. Cover: vulnerability details, OWASP mapping, EPSS score, risk scoring, and remediation.
Always end with FINAL: when your analysis is complete.
"""


# ── FIX 1 + 2: Robust action and FINAL parsing ────────────────────────────────

# Matches: ACTION / TOOL / CALL / action / tool / call  followed by tool_name(arg)
_ACTION_RE = re.compile(
    r"(?:ACTION|TOOL|CALL|EXECUTE|USE):\s*(\w+)\s*\((.+?)\)\s*$",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# Matches: FINAL / FINAL ANSWER / FINAL RESPONSE / CONCLUSION:
_FINAL_RE = re.compile(
    r"(?:FINAL(?:\s+(?:ANSWER|RESPONSE|REPORT))?|CONCLUSION):\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_action(text: str) -> tuple[str | None, str | None]:
    """
    Extract (tool_name, argument) from agent output.

    FIX 1: Case-insensitive, accepts ACTION/TOOL/CALL/EXECUTE/USE.
    FIX 2: Only returns valid tool names (prevents phantom tool calls).

    Returns (None, None) if no valid tool call found.
    """
    for match in _ACTION_RE.finditer(text):
        tool_name = match.group(1).strip()
        tool_arg  = match.group(2).strip()
        if tool_name in TOOL_NAMES:
            return tool_name, tool_arg
        # Try case-insensitive match
        for registered in TOOL_NAMES:
            if tool_name.lower() == registered.lower():
                return registered, tool_arg
    return None, None


def parse_final(text: str) -> str | None:
    """
    Extract the final answer from agent output.
    FIX 3: Handles FINAL ANSWER:, FINAL RESPONSE:, CONCLUSION: variants.
    Returns the content after the keyword, or None if not found.
    """
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def call_tool(tool_name: str, tool_arg: str) -> str:
    """
    Execute a tool call with proper argument handling.

    FIX 2: generate_finding now accepts a JSON string argument.
    The old pipe-delimiter approach broke when CVE descriptions or OWASP
    categories contained "|" characters (common in log-format descriptions).

    JSON format example:
      generate_finding({"name":"SQL Injection","cve":"CVE-2023-1234",
                        "desc":"...","cvss":"9.8","owasp":"A03:2021"})
    """
    tool_fn = TOOLS[tool_name][0]

    if tool_name == "generate_finding":
        # FIX 2: parse JSON arg — fall back to old pipe format for backward compat
        try:
            kwargs = json.loads(tool_arg)
            return tool_fn(
                name  = kwargs.get("name", ""),
                cve   = kwargs.get("cve", ""),
                desc  = kwargs.get("desc", ""),
                cvss  = kwargs.get("cvss", ""),
                owasp = kwargs.get("owasp", ""),
            )
        except (json.JSONDecodeError, TypeError):
            # Backward compat: try old pipe format
            parts = [p.strip() for p in tool_arg.split("|")]
            parts += [""] * (5 - len(parts))
            return tool_fn(*parts[:5])

    return tool_fn(tool_arg)


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    max_steps:  int  = 8,
    verbose:    bool = True,
) -> str:
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
        f"[USER]: {user_query}",
    ]

    tools_called: list[str] = []      # track which tools ran (for fallback summary)
    tool_results: list[str] = []      # raw results for fallback summary

    for step_num in range(1, max_steps + 1):
        context = "\n".join(memory)

        response = ask_model(
            instruction=(
                "Based on the conversation so far, decide the next step. "
                "Call a tool using ACTION: tool_name(argument) "
                "or provide your complete FINAL: analysis."
            ),
            context=context,
            layer="general",
        )

        if verbose:
            preview = response[:250] + ("..." if len(response) > 250 else "")
            print(f"\n[Step {step_num}] {preview}")

        memory.append(f"[AGENT]: {response}")

        # FIX 3: Check for final answer (handles FINAL ANSWER:, CONCLUSION:, etc.)
        final_text = parse_final(response)
        if final_text:
            if verbose:
                print(f"\n✅ Agent completed in {step_num} step(s).")
            return final_text

        # FIX 1: Robust tool call parsing
        tool_name, tool_arg = parse_action(response)

        if tool_name:
            try:
                result = call_tool(tool_name, tool_arg)
                tools_called.append(tool_name)
                tool_results.append(f"[{tool_name}]: {result}")

                if verbose:
                    print(f"  → {tool_name}({tool_arg[:60]}...) = {result[:150]}...")
                memory.append(f"[TOOL {tool_name}]: {result}")

            except Exception as exc:
                err_msg = f"Tool {tool_name} failed: {exc}"
                memory.append(f"[TOOL {tool_name} ERROR]: {err_msg}")
                if verbose:
                    print(f"  ❌ {err_msg}")

        else:
            # No valid tool call and no FINAL — nudge the model once
            nudge = (
                "[SYSTEM]: No valid tool call found. "
                "Please call a tool with ACTION: tool_name(argument) "
                f"or end with FINAL: <your analysis>. "
                f"Available tools: {', '.join(TOOL_NAMES)}"
            )
            memory.append(nudge)
            if verbose:
                print(f"  ⚠️  No action parsed — nudging model.")

    # ── Fallback: max steps reached without FINAL ──────────────────────────
    if verbose:
        print(f"\n⚠️  Max steps ({max_steps}) reached — generating forced summary.")

    # Build a clean context from tool results only (avoids repeating the full memory)
    summary_context = (
        f"Original query: {user_query}\n\n"
        + "\n\n".join(tool_results)
        if tool_results
        else "\n".join(memory[-6:])    # last 6 turns as fallback context
    )

    return ask_model(
        instruction=(
            "The analysis agent has gathered the following information. "
            "Synthesize it into a complete vulnerability analysis report covering: "
            "CVE details, OWASP category, risk score, attack method, and remediation."
        ),
        context=summary_context,
        layer="audit_evidence",
    )