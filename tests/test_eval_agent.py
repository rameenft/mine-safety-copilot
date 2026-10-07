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


def test_phrase_rules_from_manual_review():
    agg = {"id": "a", "type": "agg", "must_refuse": False, "expected_value": "AZ",
           "lead_phrases": ["sample"], "forbidden_phrases": ["representative"]}
    ok = "In the 3,000-record sample, AZ has the most.\n\nDetails follow."
    assert score_item(agg, _result(ok))["correct"]
    late = "AZ has the most.\n\nThese counts come from a sample."
    assert "phrase up front: 'sample'" in score_item(agg, _result(late))["missing"]
    bad = "In this representative sample, AZ has the most."
    assert not score_item(agg, _result(bad))["correct"]
    assert score_item(agg, _result("The sample is not representative; AZ leads."))["correct"]
    assert score_item(agg, _result("A sample; it should not be considered representative. AZ."))["correct"]


def test_partial_needs_not_covered_line_and_is_its_own_type():
    part = {"id": "p", "type": "partial", "must_refuse": False, "expected_sections": ["56.15002"],
            "required_phrases": ["Not covered:"]}
    good = _result("Hard hats: § 56.15002.\nNot covered: training (30 CFR Part 46).", ["56.15002"])
    assert score_item(part, good)["correct"]
    assert not score_item(part, _result("Hard hats: § 56.15002.", ["56.15002"]))["correct"]
    rows = summarize([part], {"p": score_item(part, good)})
    assert rows["partial"]["answer_acc"] == 1.0


def test_forbidden_section_is_the_expired_version_only():
    item = {"id": "r", "type": "reg", "must_refuse": False, "expected_sections": ["56.5005T"],
            "forbidden_sections": ["56.5005"]}
    assert score_item(item, _result("See § 56.5005T.", ["56.5005T"]))["correct"]
    s = score_item(item, _result("See § 56.5005T and § 56.5005.", ["56.5005T", "56.5005"]))
    assert not s["correct"] and "cited expired: 56.5005" in s["missing"]
