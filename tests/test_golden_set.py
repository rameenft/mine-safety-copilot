from pathlib import Path

import pytest
import yaml

from mine_copilot.config import DB_PATH

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden_set.yaml"
TYPES = {"reg", "agg", "hybrid", "partial", "refuse"}


@pytest.fixture(scope="module")
def items():
    return yaml.safe_load(GOLDEN.read_text())


def test_schema(items):
    assert 25 <= len(items) <= 35
    assert len({i["id"] for i in items}) == len(items)
    for i in items:
        assert i["type"] in TYPES and i["question"] and i["verified_by"]
        assert i["must_refuse"] == (i["type"] == "refuse")
        if i["type"] in {"reg", "hybrid"}:
            assert i["expected_sections"]
        if i["type"] in {"agg", "hybrid"}:
            assert "expected_value" in i
        if i["type"] == "partial":
            assert "Not covered:" in i["required_phrases"]
            assert i.get("expected_sections") or "expected_value" in i


def test_every_type_covered(items):
    assert {i["type"] for i in items} == TYPES


@pytest.mark.skipif(not DB_PATH.exists(), reason="processed DB not built")
def test_matches_real_db(items):
    import sys
    sys.path.insert(0, str(GOLDEN.parent))
    from build_golden import build

    assert build() == items
