"""Tool-calling loop + post-hoc grounding check.

The model may only state what tool results support. After it answers, `ground()` compares every
cited section and every number in the answer with what the tools actually returned; anything
unsupported is dropped from the citations and reported in `ungrounded`.
"""

import json
import re
from dataclasses import asdict, dataclass, field

from mine_copilot.agent.tools import TOOL_SPECS, YEARS, Tools, call_tool
from mine_copilot.llm import Provider

MAX_TOOL_TURNS = 5
FINAL_NUDGE = "Tool budget used up. Answer now using only the tool results above."
REFUSAL_PREFIX = "Out of scope:"
PARTIAL_PREFIX = "Not covered:"

SYSTEM = f"""You are Mine Safety Copilot, a safety assistant for US SURFACE metal/nonmetal mines.
Your only sources are tool results: 30 CFR Part 56 regulations and MSHA accident records
({YEARS[0]}-{YEARS[1]}).

Rules:
1. Always use tools before answering. Never answer from memory.
2. Regulation questions: search_regulations, then get_regulation for the section you rely on.
   Cite sections as "§ 56.xxxx".
3. Statistics: use accident_stats; report numbers exactly as returned. If a number includes
   non-fatal records, your first sentence must say it comes from a 3,000-record sample. Every
   fatality is kept, so the sample over-represents severe accidents: never call it representative.
   Fatal-only counts are complete. Counts from narrative_contains are keyword matches: say
   accidents "mentioning" the term, never "caused by" it. You may cite example document numbers.
4. Questions with both parts: answer both, each grounded in its own tool result.
5. Refuse only when the WHOLE question is out of scope: coal mines, underground mines
   (Part 57/75), years outside {YEARS[0]}-{YEARS[1]}, legal/financial advice, or anything the tools
   cannot answer. A refusal starts with "{REFUSAL_PREFIX}" and says briefly what you do cover.
6. If only PART of a question is in scope, answer that part with citations, then add a final
   line starting "{PARTIAL_PREFIX}" naming what you can't answer and where to look. Don't
   answer the uncovered part. Pointers you may give: training is 30 CFR Part 46; penalties are
   30 CFR Part 100; underground metal/nonmetal is Part 57; coal is Parts 70-75; non-mine
   workplaces are OSHA; years outside {YEARS[0]}-{YEARS[1]} are on MSHA's data portal.
7. If tools return nothing relevant, say so rather than guessing.
Be concise: a direct answer first, then the supporting citation(s)."""

SECTION_RE = re.compile(r"56\.\d+[A-Z]?\b")
DOC_RE = re.compile(r"\b\d{12}\b")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "7. AZ: 363" -> 7 isn't a claim
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
ALWAYS_OK = {"30", "56", "57", "75"}  # "30 CFR Part 56" etc.
# Regulations spell small numbers out ("seven feet"); the model often writes digits.
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}


@dataclass
class AgentResult:
    question: str
    answer: str
    refused: bool
    citations: dict = field(default_factory=dict)  # {"sections": [...], "documents": [...]}
    ungrounded: dict = field(default_factory=dict)  # {"sections": [...], "numbers": [...]}
    trace: list[dict] = field(default_factory=list)  # [{"tool", "args", "result"}]

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def _evidence_numbers(text: str) -> set[str]:
    words = {NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}
    return _numbers(text) | words


def ground(answer: str, question: str, trace: list[dict]) -> tuple[dict, dict]:
    """Split the answer's claims into grounded citations and ungrounded leftovers."""
    evidence = " ".join(json.dumps(t["result"]) for t in trace)
    seen_sections = set(SECTION_RE.findall(evidence))
    seen_docs = set(DOC_RE.findall(evidence))

    sections = list(dict.fromkeys(SECTION_RE.findall(answer)))
    docs = list(dict.fromkeys(DOC_RE.findall(answer)))
    # Strip cited IDs before checking loose numbers so "56.14107" isn't read as two numbers.
    rest = LIST_MARKER_RE.sub(" ", DOC_RE.sub(" ", SECTION_RE.sub(" ", answer)))
    # SYSTEM is evidence too: it states the coverage years a refusal may repeat.
    allowed = _evidence_numbers(evidence + " " + SYSTEM) | _numbers(question) | ALWAYS_OK
    loose = sorted(_numbers(rest) - allowed, key=lambda n: float(n))

    citations = {"sections": [s for s in sections if s in seen_sections],
                 "documents": [d for d in docs if d in seen_docs]}
    ungrounded = {"sections": [s for s in sections if s not in seen_sections],
                  "documents": [d for d in docs if d not in seen_docs],
                  "numbers": loose}
    return citations, ungrounded


def run(question: str, provider: Provider, tools: Tools | None = None,
        max_turns: int = MAX_TOOL_TURNS) -> AgentResult:
    tools = tools or Tools()
    messages: list[dict] = [{"role": "user", "text": question}]
    trace: list[dict] = []

    for turn in range(max_turns + 1):
        # Last turn: withhold tools and say so; without the nudge Gemini can return empty text.
        if turn == max_turns:
            messages.append({"role": "user", "text": FINAL_NUDGE})
        reply = provider.chat(SYSTEM, messages, TOOL_SPECS if turn < max_turns else None)
        messages.append({"role": "assistant", "text": reply.text,
                         "tool_calls": reply.tool_calls, "raw": reply.raw})
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            result = call_tool(tools, call["name"], call["args"])
            trace.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append({"role": "tool", "name": call["name"], "content": result})

    answer = reply.text or "I could not produce an answer from the available sources."
    refused = answer.startswith(REFUSAL_PREFIX)
    citations, ungrounded = ground(answer, question, trace)
    return AgentResult(question, answer, refused, citations, ungrounded, trace)
