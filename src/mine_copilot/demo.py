"""Non-UI helpers for the Streamlit demo (`app.py`), kept here so they test without Streamlit."""

import re
import time
from pathlib import Path

from mine_copilot.agent.loop import AgentResult
from mine_copilot.config import ROOT

REPO_URL = "https://github.com/rameenft/mine-safety-copilot"
AGENT_REPORT = ROOT / "evals" / "reports" / "agent.md"

# One golden-set question per type, so the demo shows every path through the agent.
EXAMPLES = {
    "Regulation": "How high must berms be on haul roads?",
    "Statistics": "Which year had the most fatal accidents?",
    "Hybrid": "How many fatal accidents involved conveyors, and what rule covers guarding them?",
    "Refusal": "What does 30 CFR Part 75 require for roof bolting in underground coal mines?",
}


class Metered:
    """Wraps a Provider to total tokens and LLM calls for one question."""

    def __init__(self, provider):
        self.provider = provider
        self.model = getattr(provider, "model", type(provider).__name__)
        self.calls = 0
        self.tokens = {"input": 0, "output": 0}

    def chat(self, system, messages, tools):
        reply = self.provider.chat(system, messages, tools)
        self.calls += 1
        for k in self.tokens:
            self.tokens[k] += reply.usage.get(k, 0)
        return reply


def timed_run(run, question: str, provider, tools) -> tuple[AgentResult, dict]:
    """Run the agent; return the result plus {model, llm_calls, tokens, seconds}."""
    metered, t0 = Metered(provider), time.time()
    result = run(question, metered, tools)
    return result, {"model": metered.model, "llm_calls": metered.calls,
                    "tokens": metered.tokens, "seconds": round(time.time() - t0, 1)}


def grounding_problems(result: AgentResult) -> list[str]:
    """Human-readable list of claims the grounding check could not trace to a tool result."""
    labels = {"sections": "section §", "documents": "document", "numbers": "number"}
    return [f"{labels.get(kind, kind)} {v}" for kind, vals in result.ungrounded.items() for v in vals]


def friendly_error(e: Exception) -> str:
    """Map the failures a demo actually hits to a one-line fix instead of a stack trace."""
    msg = str(e)
    if isinstance(e, KeyError) and "GEMINI_API_KEY" in msg:
        return "No API key: add `GEMINI_API_KEY=...` to `.env`, then restart the app."
    if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in msg:
        return "Rate limit or quota reached (429). Wait a minute and try again."
    if getattr(e, "code", None) == 503:
        return "The model is overloaded (503). Try again in a few seconds."
    if isinstance(e, FileNotFoundError) or "no such table" in msg:
        return ("Data not built: run `python -m mine_copilot.ingest.build` and "
                "`python -m mine_copilot.retrieval.build`.")
    return f"Something went wrong: {type(e).__name__}: {msg[:200]}"


def eval_summary(path: Path = AGENT_REPORT) -> dict | None:
    """Headline numbers from the latest agent eval report (the `| all |` row), if present."""
    if not path.exists():
        return None
    text = path.read_text()
    model = re.search(r"^# Agent eval — (.+)$", text, re.MULTILINE)
    row = re.search(r"^\| all \| (\d+) \| ([\d.]+) \| ([\d.–]+) \| ([\d.]+) \| ([\d.]+) \|$",
                    text, re.MULTILINE)
    if not row:
        return None
    n, acc, cite, refusal, grounded = row.groups()
    return {"model": model.group(1) if model else "?", "n": int(n), "answer_acc": acc,
            "citation_recall": cite, "refusal_acc": refusal, "grounded": grounded}
