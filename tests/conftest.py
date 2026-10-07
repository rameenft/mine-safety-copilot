"""Shared fixtures: a fixture-backed SQLite DB and BM25 index, so tool tests need no network."""

import sqlite3

import pandas as pd
import pytest

from mine_copilot.agent.tools import Tools
from mine_copilot.config import FIXTURES_DIR
from mine_copilot.ingest.accidents import KEEP_COLUMNS, filter_accidents, load_mines, read_pipe_file
from mine_copilot.ingest.regulations import parse_part56
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
