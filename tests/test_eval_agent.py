"""Scoring logic of evals/eval_agent.py — no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from eval_agent import fact_present, score_item, summarize, value_present  # noqa: E402


def _result(answer, sections=(), refused=False, ungrounded=None):
    return {"answer": answer, "refused": refused, "citations": {"sections": list(sections)},
            "ungrounded": ungrounded or {"sections": [], "numbers": []}, "trace": []}


def test_fact_present_tolerates_paraphrase():
    assert fact_present("shall be guarded", "Pulleys must be guarded.")
    assert fact_present("at least once each shift", "Examine each shift, at least once.")
    assert not fact_present("blocked against hazardous motion", "Turn the power off.")


def test_value_present_numbers_and_strings():
    assert value_present(80, "There were 80 fatalities.")
    assert value_present(64.5, "Average was 64.5 days.")
    assert not value_present(27, "There were 270 accidents.")
    assert value_present("MACHINERY", "Machinery caused the most.")
    assert not value_present("AZ", "Arizona-based mines")  # whole-word only


def test_score_hybrid_needs_value_and_grounded_section():
    item = {"id": "h", "type": "hybrid", "must_refuse": False, "expected_value": 6,
            "expected_sections": ["56.14107"]}
    assert score_item(item, _result("6 fatal; see § 56.14107", ["56.14107"]))["correct"]
    s = score_item(item, _result("6 fatal; see § 56.14107", []))  # cited but not grounded
    assert not s["correct"] and s["citation_recall"] == 0


def test_score_refusal_and_false_refusal():
    refuse = {"id": "r", "type": "refuse", "must_refuse": True}
    agg = {"id": "a", "type": "agg", "must_refuse": False, "expected_value": 27}
    scores = {"r": score_item(refuse, _result("Out of scope: coal.", refused=True)),
              "a": score_item(agg, _result("Out of scope: dunno.", refused=True))}
    assert scores["r"]["correct"] and not scores["a"]["correct"]
    rows = summarize([refuse, agg], scores)
    assert rows["all"]["refusal_acc"] == 0.5 and rows["all"]["false_refusal"] == 1.0
