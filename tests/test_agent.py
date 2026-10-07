"""Agent tools, loop and grounding — fixture DB + FakeProvider, no network."""

import sqlite3

import pandas as pd
import pytest

from mine_copilot.agent.loop import FINAL_NUDGE, REFUSAL_PREFIX, ground, run
from mine_copilot.agent.tools import TOOL_SPECS, Tools, call_tool
from mine_copilot.config import FIXTURES_DIR
from mine_copilot.ingest.accidents import KEEP_COLUMNS, filter_accidents, load_mines, read_pipe_file
from mine_copilot.ingest.regulations import parse_part56
from mine_copilot.llm import FakeProvider, Reply
from mine_copilot.retrieval.index import RegIndex

ACC = filter_accidents(read_pipe_file(FIXTURES_DIR / "accidents_sample.txt", KEEP_COLUMNS))
SECTIONS = parse_part56((FIXTURES_DIR / "part56_sample.xml").read_bytes(), "2026-10-01")


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "fixture.db"
    with sqlite3.connect(db) as conn:
        ACC.to_sql("accidents", conn, index=False)
        load_mines(FIXTURES_DIR / "mines_sample.txt").to_sql("mines", conn, index=False)
        pd.DataFrame(SECTIONS).to_sql("regulations", conn, index=False)
    return Tools(db, index=RegIndex.from_sections(SECTIONS), search_mode="bm25")


# --- tools -----------------------------------------------------------------------------------

def test_stats_match_pandas(tools):
    fatal = ACC[ACC["SEVERITY"] == "fatal"]
    out = tools.accident_stats(severity="fatal")
    assert out["rows"] == [{"count": len(fatal)}] and out["total_matching"] == len(fatal)
    assert set(out["example_document_nos"]) <= set(fatal["DOCUMENT_NO"])

    by_year = tools.accident_stats(severity="fatal", group_by="year")["rows"]
    assert by_year[0]["count"] == fatal["CAL_YR"].value_counts().max()


def test_narrative_terms_are_ored(tools):
    terms = ["conveyor", "belt"]
    expected = ACC["NARRATIVE"].str.contains("conveyor|belt", case=False).sum()
    assert tools.accident_stats(narrative_contains=terms)["total_matching"] == expected


def test_stats_rejects_bad_input(tools):
    assert "outside data coverage" in tools.accident_stats(year_from=2018)["error"]
    assert "Unknown filter" in tools.accident_stats(mine_name="x")["error"]
    assert "group_by" in tools.accident_stats(group_by="NARRATIVE; DROP TABLE accidents")["error"]
    assert "severity" in tools.accident_stats(severity="deadly")["error"]


def test_injection_in_values_is_just_data(tools):
    out = tools.accident_stats(classification="' OR 1=1 --")
    assert out["total_matching"] == 0


def test_get_and_search_regulation(tools):
    assert tools.get_regulation("§ 56.14107")["section_id"] == "56.14107"
    assert "not found" in tools.get_regulation("75.202")["error"]
    hits = tools.search_regulations("berm height on haul roads")["results"]
    assert hits[0]["section_id"] == "56.9300"


def test_call_tool_handles_bad_calls(tools):
    assert "Unknown tool" in call_tool(tools, "drop_db", {})["error"]
    assert "Bad arguments" in call_tool(tools, "get_regulation", {})["error"]
    assert {t["name"] for t in TOOL_SPECS} == {"search_regulations", "get_regulation",
                                               "accident_stats"}


# --- grounding -------------------------------------------------------------------------------

TRACE = [{"tool": "accident_stats", "args": {},
          "result": {"rows": [{"count": 6}], "example_document_nos": ["220210280004"]}},
         {"tool": "get_regulation", "args": {},
          "result": {"section_id": "56.14107", "text": "at least seven feet away"}}]


def test_ground_keeps_supported_claims():
    cites, bad = ground("There were 6 (e.g. 220210280004). Guards unless 7 feet away (§ 56.14107 "
                        "of 30 CFR Part 56). I cover 2021-2024 only.",
                        "How many conveyor deaths in 2023?", TRACE)
    assert cites == {"sections": ["56.14107"], "documents": ["220210280004"]}
    assert not any(bad.values())


def test_ground_ignores_list_markers():
    trace = [{"tool": "accident_stats", "args": {}, "result": {"rows": [["AZ", 363], ["TX", 263]]}}]
    _, ungrounded = ground("Top states:\n1. AZ: 363\n7. TX: 263", "which state?", trace)
    assert ungrounded["numbers"] == []


def test_ground_flags_invented_claims():
    cites, bad = ground("There were 9 deaths; see § 56.9999 and 999999999999.", "q", TRACE)
    assert cites == {"sections": [], "documents": []}
    assert bad == {"sections": ["56.9999"], "documents": ["999999999999"], "numbers": ["9"]}


# --- loop ------------------------------------------------------------------------------------

def test_loop_runs_tools_then_answers(tools):
    n = int((ACC["SEVERITY"] == "fatal").sum())
    fake = FakeProvider([
        Reply(tool_calls=[{"name": "accident_stats", "args": {"severity": "fatal"}},
                          {"name": "get_regulation", "args": {"section_id": "56.14107"}}]),
        Reply(text=f"{n} fatal accidents. Guarding: § 56.14107."),
    ])
    res = run("How many fatal accidents, and what covers guarding?", fake, tools)
    assert [t["tool"] for t in res.trace] == ["accident_stats", "get_regulation"]
    assert res.citations["sections"] == ["56.14107"] and not res.refused
    assert not any(res.ungrounded.values())
    # Second model call saw both tool results.
    assert [m["role"] for m in fake.calls[1]["messages"]][-2:] == ["tool", "tool"]


def test_loop_detects_refusal(tools):
    fake = FakeProvider([Reply(text=f"{REFUSAL_PREFIX} I only cover surface mines.")])
    res = run("Roof bolting in underground coal?", fake, tools)
    assert res.refused and res.trace == []


def test_loop_forces_answer_after_max_turns(tools):
    call = Reply(tool_calls=[{"name": "search_regulations", "args": {"query": "berms"}}])
    fake = FakeProvider([call, call, Reply(text="Berms: § 56.9300.")])
    res = run("berms?", fake, tools, max_turns=2)
    assert fake.calls[-1]["tools"] is None  # tools withheld on the final turn
    assert fake.calls[-1]["messages"][-1] == {"role": "user", "text": FINAL_NUDGE}
    assert len(res.trace) == 2 and res.citations["sections"] == ["56.9300"]


def test_section_status_labels_dated_versions():
    from datetime import date

    from mine_copilot.agent.tools import section_status

    old = "The following is required until April 7, 2026. Except as permitted..."
    new = "As of April 8, 2026 the following is required, except as permitted..."
    assert section_status("56.5001", old, date(2026, 10, 7)).startswith("EXPIRED")
    assert "56.5001T" in section_status("56.5001", old, date(2026, 10, 7))
    assert section_status("56.5001", old, date(2025, 1, 1)).startswith("IN FORCE until")
    assert section_status("56.5001T", new, date(2026, 10, 7)).startswith("IN FORCE since")
    assert section_status("56.5001T", new, date(2025, 1, 1)).startswith("NOT YET")
    assert section_status("56.14107", "(a) Moving machine parts shall be guarded", date(2026, 1, 1)) is None


def test_expired_section_redirects_to_in_force_version(tmp_path):
    import sqlite3

    from mine_copilot.agent.tools import Tools

    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE regulations (section_id, subpart, heading, text, source_url)")
    con.executemany("INSERT INTO regulations VALUES (?, '', 'h', ?, 'u')", [
        ("56.5005", "The following is required until April 7, 2026. Old text."),
        ("56.5005T", "As of April 8, 2026, the following is required. New text.")])
    con.commit()
    out = Tools(db_path=db).get_regulation("56.5005")
    assert out["section_id"] == "56.5005T" and "Cite § 56.5005T" in out["note"]
