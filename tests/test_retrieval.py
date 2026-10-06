import numpy as np
import pytest

from mine_copilot.config import FIXTURES_DIR
from mine_copilot.ingest.regulations import parse_part56
from mine_copilot.retrieval.index import RegIndex, rrf, tokenize

XML = (FIXTURES_DIR / "part56_sample.xml").read_bytes()


@pytest.fixture(scope="module")
def index():
    return RegIndex.from_sections(parse_part56(XML, "2026-10-01"))  # BM25 only, no network


def top(index, query, k=3):
    return [h["section_id"] for h in index.search(query, k=k)]


def test_tokenize_keeps_section_ids():
    assert tokenize("See § 56.14107 for the guards") == ["see", "56.14107", "guards"]


def test_bm25_finds_topic(index):
    assert top(index, "berm height on haul roads")[0] == "56.9300"
    assert top(index, "hard hats where falling objects")[0] == "56.15002"
    assert "56.12016" in top(index, "lock out electrically powered equipment before work")


def test_named_section_is_pinned(index):
    hits = index.search("what does 56.18010 say about guarding pulleys", k=3)
    assert hits[0]["section_id"] == "56.18010" and hits[0]["pinned"]
    assert not hits[1]["pinned"]


def test_one_hit_per_section(index):
    ids = top(index, "equipment", k=10)
    assert len(ids) == len(set(ids))


def test_rrf_rewards_agreement():
    # 2 is mid-ranked in both lists; 0 and 3 top only one list each.
    assert rrf([[0, 2, 1], [3, 2, 4]])[0] == 2


def test_hybrid_uses_supplied_vectors(index, tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(index.chunks), 8)).astype(np.float32)
    dense = RegIndex(index.chunks, emb)
    target = next(i for i, c in enumerate(dense.chunks) if c["section_id"] == "56.4100")
    hits = dense.search("zzz", k=1, mode="dense", query_vec=emb[target])
    assert hits[0]["section_id"] == "56.4100"

    dense.save(tmp_path)
    assert RegIndex.load(tmp_path).embeddings.shape == emb.shape


def test_rrf_weights_favour_a_list():
    assert rrf([[0, 1], [1, 0]], weights=[1.0, 2.0])[0] == 1
