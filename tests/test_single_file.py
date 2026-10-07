"""The one-file version (mine_safety_copilot.py/.ipynb) must match src/ and behave the same."""

import json
import sys
from pathlib import Path

import pytest
from conftest import ACC, SECTIONS

from mine_copilot.config import FIXTURES_DIR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_single_file as gen


def test_committed_files_are_up_to_date():
    cells = gen.cells()
    assert (ROOT / "mine_safety_copilot.py").read_text() == gen.to_py(cells), \
        "run: python scripts/build_single_file.py"
    nb = json.loads((ROOT / "mine_safety_copilot.ipynb").read_text())
    assert [c["source"] for c in nb["cells"]] == \
        [c["source"] for c in gen.to_notebook(cells)["cells"]]  # outputs may differ


@pytest.fixture(scope="module")
def one_file(tmp_path_factory):
    import os
    os.environ.setdefault("COPILOT_HOME", str(tmp_path_factory.mktemp("home")))
    ns = {"__name__": "single_file_test"}  # not __main__: definitions only, nothing runs
    path = ROOT / "mine_safety_copilot.py"
    exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102 - our own generated file
    return ns


def test_one_file_agent_matches_package(one_file, tools):
    ns = one_file
    t = ns["Tools"](tools.db_path, index=ns["RegIndex"].from_sections(SECTIONS), search_mode="bm25")
    assert t.accident_stats(severity="fatal") == tools.accident_stats(severity="fatal")
    raw = ns["read_pipe_file"](FIXTURES_DIR / "accidents_sample.txt", ns["KEEP_COLUMNS"])
    assert ns["filter_accidents"](raw).equals(ACC)  # ACCIDENT_YEARS rename kept the filter

    reply = ns["Reply"]
    fake = ns["FakeProvider"]([
        reply(text="", tool_calls=[{"name": "accident_stats", "args": {"severity": "fatal"}}]),
        reply(text=f"There were {t.accident_stats(severity='fatal')['total_matching']} fatal "
                   "accidents.")])
    result = ns["run"]("How many fatal accidents?", fake, t)
    assert not any(result.ungrounded.values()) and not result.refused
