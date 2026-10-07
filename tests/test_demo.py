"""Streamlit demo — helpers plus the app's render path with FakeProvider, no network."""

from pathlib import Path

import pytest

from mine_copilot.agent.loop import REFUSAL_PREFIX, AgentResult
from mine_copilot.demo import Metered, eval_summary, friendly_error, grounding_problems
from mine_copilot.llm import FakeProvider, Reply

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "mine_copilot" / "app.py"
CALL = Reply(tool_calls=[{"name": "get_regulation", "args": {"section_id": "56.14107"}}],
             usage={"input": 100, "output": 10})


# --- helpers ---------------------------------------------------------------------------------

def test_metered_totals_usage():
    m = Metered(FakeProvider([CALL, Reply(text="ok", usage={"input": 50, "output": 5})]))
    m.chat("s", [], None), m.chat("s", [], None)
    assert m.calls == 2 and m.tokens == {"input": 150, "output": 15} and m.model == "FakeProvider"


def test_grounding_problems_lists_each_claim():
    r = AgentResult("q", "a", False, ungrounded={"sections": ["56.9999"], "documents": [],
                                                 "numbers": ["9"]})
    assert grounding_problems(r) == ["section § 56.9999", "number 9"]


def test_friendly_errors():
    class Api(Exception):
        code = 429
    assert "GEMINI_API_KEY" in friendly_error(KeyError("GEMINI_API_KEY"))
    assert "429" in friendly_error(Api("quota"))
    assert "ingest.build" in friendly_error(FileNotFoundError("chunks.json"))


def test_eval_summary_parses_report(tmp_path):
    p = tmp_path / "agent.md"
    p.write_text("# Agent eval — m1\n\n| all | 24 | 1.00 | 0.90 | 1.00 | 0.96 |\n")
    assert eval_summary(p) == {"model": "m1", "n": 24, "answer_acc": "1.00",
                               "citation_recall": "0.90", "refusal_acc": "1.00", "grounded": "0.96"}
    assert eval_summary(tmp_path / "missing.md") is None


# --- app render path -------------------------------------------------------------------------

def ask(tools, replies, question="Guarding rules?"):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["tools"] = tools
    at.session_state["provider"] = FakeProvider(replies)
    at.run()
    at.text_input(key="question").input(question)
    at.button[-1].click().run()  # the form's Ask button is rendered last
    assert not at.exception
    return at


def test_app_grounded_answer(tools):
    at = ask(tools, [CALL, Reply(text="Guard parts within 7 feet (§ 56.14107).")])
    assert "Grounded" in at.success[0].value and not at.error
    labels = [e.label for e in at.expander]
    assert any(lbl.startswith("§ 56.14107 — Moving machine parts") for lbl in labels)
    assert "Tool trace (1 call)" in labels
    assert "LLM calls" in at.sidebar.markdown[-1].value


def test_app_flags_ungrounded_number(tools):
    at = ask(tools, [CALL, Reply(text="There were 999 cases (§ 56.14107).")])
    assert "number 999" in at.error[0].value and not at.success


def test_app_refusal_and_error(tools):
    at = ask(tools, [Reply(text=f"{REFUSAL_PREFIX} I only cover surface mines.")])
    assert at.warning[0].value.startswith(REFUSAL_PREFIX)

    at = ask(tools, [], question="anything")  # FakeProvider runs dry -> IndexError
    assert "Something went wrong" in at.error[0].value
