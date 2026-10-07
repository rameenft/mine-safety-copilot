"""End-to-end agent eval on the golden set: answer accuracy, citation recall, refusal accuracy.

Usage: python evals/eval_agent.py [--model gemini-2.5-flash] [--only reg-01,agg-02] [--rescore]
Each answer is cached in evals/runs/<model>/<id>.json, so a run cut short by quota resumes where it
stopped; --rescore rebuilds the report from the cache without any API calls.
Writes evals/reports/agent.md.
"""

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-2.5-flash"
MIN_CALL_GAP = 13  # seconds between LLM calls: free-tier gemini-2.5-flash allows 5 requests/minute
FACT_THRESHOLD = 0.6  # share of a fact's content words that must appear in the answer
STOPWORDS = {"a", "an", "the", "of", "to", "be", "is", "are", "at", "on", "in", "and", "or",
             "for", "by", "with", "shall", "must", "that", "this", "it"}


# --- scoring (pure; unit-tested) ------------------------------------------------------------------

def _words(text: str) -> set[str]:
    # Crude stemming: "pulleys"/"pulley", "locked"/"lock" should match.
    return {re.sub(r"(es|s|ed|ing)$", "", w) for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOPWORDS}


def fact_present(fact: str, answer: str) -> bool:
    """Token-overlap match, so 'must be guarded' counts for 'shall be guarded'."""
    want = _words(fact)
    return not want or len(want & _words(answer)) / len(want) >= FACT_THRESHOLD


def value_present(expected, answer: str) -> bool:
    if isinstance(expected, (int, float)):
        nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", answer)}
        return float(expected) in nums
    return re.search(rf"\b{re.escape(str(expected))}\b", answer, re.IGNORECASE) is not None


def lead(answer: str) -> str:
    """The opening paragraph: where a caveat has to be to count as 'up front'."""
    return answer.strip().split("\n\n")[0]


def phrase_checks(item: dict, answer: str) -> list[str]:
    """Case-insensitive phrase rules from manual review; returns what's missing or forbidden."""
    low, head = answer.lower(), lead(answer).lower()
    miss = [f"phrase: {p!r}" for p in item.get("required_phrases", []) if p.lower() not in low]
    miss += [f"phrase up front: {p!r}" for p in item.get("lead_phrases", []) if p.lower() not in head]
    # "not (be considered) representative" is the honest caveat, so a negated phrase is fine.
    miss += [f"forbidden: {p!r}" for p in item.get("forbidden_phrases", [])
             if any(not re.search(r"\bnot\b", low[max(0, m.start() - 30):m.start()])
                    for m in re.finditer(rf"\b{re.escape(p.lower())}", low))]
    return miss


def score_item(item: dict, result: dict) -> dict:
    """Per-question scores. None means the metric doesn't apply to this question type."""
    answer, refused = result["answer"], result["refused"]
    expected_secs = set(item.get("expected_sections", []))
    cited = set(result["citations"]["sections"])
    s = {"refusal_ok": refused == item["must_refuse"],
         "citation_recall": len(expected_secs & cited) / len(expected_secs) if expected_secs else None,
         "grounded": not any(result["ungrounded"].values()),
         "missing": []}

    if item["must_refuse"]:
        s["correct"] = refused
        return s
    checks = []
    if "expected_facts" in item:
        for f in item["expected_facts"]:
            checks.append(ok := fact_present(f, answer))
            if not ok:
                s["missing"].append(f"fact: {f}")
    if "expected_value" in item:
        checks.append(ok := value_present(item["expected_value"], answer))
        if not ok:
            s["missing"].append(f"value: {item['expected_value']}")
    if expected_secs:  # hybrid questions carry sections but no facts: the section is the answer
        checks.append(bool(expected_secs & cited))
        if not expected_secs & cited:
            s["missing"].append(f"section: {sorted(expected_secs)}")
    phrases = phrase_checks(item, answer)
    s["missing"] += phrases
    s["correct"] = not refused and all(checks) and not phrases
    return s


def _mean(xs: list) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(items: list[dict], scores: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    for kind in ["reg", "agg", "hybrid", "partial", "refuse", "all"]:
        ids = [i["id"] for i in items if (kind == "all" or i["type"] == kind) and i["id"] in scores]
        if not ids:
            continue
        sc = [scores[i] for i in ids]
        rows[kind] = {"n": len(ids),
                      "answer_acc": _mean([s["correct"] for s in sc]),
                      "citation_recall": _mean([s["citation_recall"] for s in sc]),
                      "refusal_acc": _mean([s["refusal_ok"] for s in sc]),
                      "grounded": _mean([s["grounded"] for s in sc])}
    answerable = [scores[i["id"]] for i in items if not i["must_refuse"] and i["id"] in scores]
    rows["all"]["false_refusal"] = _mean([not s["refusal_ok"] for s in answerable])
    return rows


# --- run + report -------------------------------------------------------------------------------

class Throttled:
    """Wraps a Provider so consecutive calls are at least `gap` seconds apart."""

    def __init__(self, provider, gap: float):
        self.provider, self.gap, self.last = provider, gap, 0.0
        self.calls = 0
        self.tokens = {"input": 0, "output": 0}

    def chat(self, system, messages, tools):
        time.sleep(max(0.0, self.last + self.gap - time.time()))
        try:
            reply = self.provider.chat(system, messages, tools)
        finally:
            self.last, self.calls = time.time(), self.calls + 1
        for k in self.tokens:
            self.tokens[k] += reply.usage.get(k, 0)
        return reply


def run_items(items: list[dict], model: str, run_dir: Path, gap: float) -> None:
    from mine_copilot.agent.loop import run
    from mine_copilot.llm import GeminiProvider

    provider = Throttled(GeminiProvider(model), gap)
    todo = [i for i in items if not (run_dir / f"{i['id']}.json").exists()]
    print(f"{len(items) - len(todo)} cached, {len(todo)} to run with {model}")
    for item in todo:
        t0, before = time.time(), dict(provider.tokens)
        result = run(item["question"], provider).to_dict()
        result["seconds"] = round(time.time() - t0, 1)
        result["tokens"] = {k: provider.tokens[k] - before[k] for k in before}
        (run_dir / f"{item['id']}.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"  {item['id']}: {len(result['trace'])} tool calls, {result['seconds']}s "
              f"({provider.calls} LLM calls, tokens so far {provider.tokens})")


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def report(items: list[dict], scores: dict, results: dict, model: str) -> str:
    rows = summarize(items, scores)
    tool_calls = sum(len(r["trace"]) for r in results.values())
    tok = {k: sum(r.get("tokens", {}).get(k, 0) for r in results.values()) for k in ("input", "output")}
    lines = [f"# Agent eval — {model}", "",
             f"{rows['all']['n']} golden questions · run {date.today()} · {tool_calls} tool calls "
             f"· {tok['input']:,} input / {tok['output']:,} output tokens "
             f"· false-refusal rate {_fmt(rows['all']['false_refusal'])}", "",
             "| type | n | answer acc | citation recall | refusal acc | grounded |",
             "|---|---|---|---|---|---|"]
    for kind, r in rows.items():
        lines.append(f"| {kind} | {r['n']} | {_fmt(r['answer_acc'])} | {_fmt(r['citation_recall'])} "
                     f"| {_fmt(r['refusal_acc'])} | {_fmt(r['grounded'])} |")
    lines += ["", "**Metrics.** Answer acc: expected facts (token overlap ≥ "
              f"{FACT_THRESHOLD}), value and section all present. Citation recall: expected sections "
              "among *grounded* citations. Grounded: no ungrounded numbers/sections in the answer. "
              "Phrase rules (from manual review): sample stats must say “sample” in the opening "
              "paragraph and never “representative”; keyword counts must say “mention”; partial "
              "answers must include “Not covered:”.",
              "", f"**Caveat.** {len(items)} questions written alongside the system; one miss moves "
              "a type's score by 0.11–0.25. Rule-based scoring checks that facts are present, not that "
              "everything else in the answer is right. Treat this as a regression gate, not a benchmark.",
              "", "## Failures", ""]
    fails = []
    for item in items:
        s = scores.get(item["id"])
        if s is None or (s["correct"] and s["grounded"]):
            continue
        r = results[item["id"]]
        why = "; ".join(s["missing"]) or ("wrong refusal" if not s["refusal_ok"] else "")
        ung = {k: v for k, v in r["ungrounded"].items() if v}
        fails.append(f"- **{item['id']}** {item['question']}\n  - missing: {why or '–'}"
                     + (f"\n  - ungrounded: {ung}" if ung else "")
                     + f"\n  - answer: {r['answer'][:300].replace(chr(10), ' ')}")
    return "\n".join(lines + (fails or ["None."])) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--only", help="comma-separated golden ids")
    p.add_argument("--gap", type=float, default=MIN_CALL_GAP,
                   help="min seconds between LLM calls (use ~1 on a paid key)")
    p.add_argument("--rescore", action="store_true", help="report from cache, no API calls")
    args = p.parse_args()

    items = yaml.safe_load((HERE / "golden_set.yaml").read_text())
    if args.only:
        items = [i for i in items if i["id"] in args.only.split(",")]
    run_dir = HERE / "runs" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.rescore:
        try:
            run_items(items, args.model, run_dir, args.gap)
        except Exception as e:  # quota etc.: keep what we have, report on it
            print(f"stopped early: {type(e).__name__}: {str(e)[:200]}")

    results = {i["id"]: json.loads(f.read_text()) for i in items
               if (f := run_dir / f"{i['id']}.json").exists()}
    if not results:
        print("no results to score")
        return
    # Re-ground from the cached trace so grounding-check fixes apply without new API calls.
    from mine_copilot.agent.loop import ground
    for r in results.values():
        r["citations"], r["ungrounded"] = ground(r["answer"], r["question"], r["trace"])
    scores = {k: score_item(next(i for i in items if i["id"] == k), r) for k, r in results.items()}
    text = report(items, scores, results, args.model)
    if len(results) == len(items) and not args.only:
        (HERE / "reports" / "agent.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
