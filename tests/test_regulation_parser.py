from mine_copilot.config import FIXTURES_DIR
from mine_copilot.ingest.fixtures import FIXTURE_SECTIONS
from mine_copilot.ingest.regulations import chunk_section, parse_part56

XML = (FIXTURES_DIR / "part56_sample.xml").read_bytes()


def by_id():
    return {r["section_id"]: r for r in parse_part56(XML, "2026-10-01")}


def test_one_record_per_section():
    records = parse_part56(XML, "2026-10-01")
    assert sorted(r["section_id"] for r in records) == sorted(FIXTURE_SECTIONS)


def test_fields_of_known_section():
    r = by_id()["56.14107"]
    assert r["heading"] == "Moving machine parts"
    assert r["subpart"].startswith("Subpart M")
    assert r["text"].startswith("(a) Moving machine parts shall be guarded")
    assert "seven feet" in r["text"]
    assert r["source_url"] == "https://www.ecfr.gov/on/2026-10-01/title-30/section-56.14107"


def test_suffixed_section_and_editorial_notes_skipped():
    r = by_id()["56.5001T"]
    assert "Link to an amendment" not in r["text"]


def test_short_section_is_one_chunk():
    chunks = chunk_section(by_id()["56.14107"])
    assert [c["chunk_id"] for c in chunks] == ["56.14107#0"]


def test_long_section_splits_on_paragraphs_and_keeps_id():
    record = by_id()["56.9300"]
    chunks = chunk_section(record, max_chars=600)
    assert len(chunks) > 1
    assert all(c["section_id"] == "56.9300" for c in chunks)
    assert all(c["heading"] == record["heading"] for c in chunks)
    assert "\n\n".join(c["text"] for c in chunks) == record["text"]  # nothing lost
