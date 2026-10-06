from mine_copilot.config import FIXTURES_DIR
from mine_copilot.ingest.accidents import (
    KEEP_COLUMNS,
    SURFACE_SUBUNITS,
    filter_accidents,
    load_accidents,
    load_mines,
    read_pipe_file,
    sample_accidents,
)

ACC = FIXTURES_DIR / "accidents_sample.txt"


def test_filters_drop_coal_years_underground_and_empty_narratives():
    df = filter_accidents(read_pipe_file(ACC, KEEP_COLUMNS))
    assert len(df) == 100  # fixture: 100 good rows + 17 that must be rejected
    assert set(df["COAL_METAL_IND"]) == {"M"}
    assert df["CAL_YR"].between(2021, 2024).all()
    assert set(df["SUBUNIT"]) <= SURFACE_SUBUNITS
    assert df["NARRATIVE"].notna().all()
    assert not df["DOCUMENT_NO"].str.endswith("X").any()  # blanked-narrative rows


def test_missing_tokens_become_null():
    df = read_pipe_file(ACC, KEEP_COLUMNS)
    for col in df.columns:
        values = df[col].dropna().str.lower()
        assert not values.isin({"?", "no value found", ""}).any(), col


def test_derived_fields():
    df = filter_accidents(read_pipe_file(ACC, KEEP_COLUMNS))
    assert df["ACCIDENT_DT"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert (df.loc[df["DEGREE_INJURY"] == "FATALITY", "SEVERITY"] == "fatal").all()


def test_sampling_keeps_all_fatalities_and_is_seeded():
    df = filter_accidents(read_pipe_file(ACC, KEEP_COLUMNS))
    a = sample_accidents(df, cap=30, seed=1)
    b = sample_accidents(df, cap=30, seed=1)
    assert len(a) == 30
    assert (a["SEVERITY"] == "fatal").sum() == (df["SEVERITY"] == "fatal").sum()
    assert a["DOCUMENT_NO"].tolist() == b["DOCUMENT_NO"].tolist()


def test_mines_lookup_joins():
    acc = load_accidents(ACC)
    mines = load_mines(FIXTURES_DIR / "mines_sample.txt", set(acc["MINE_ID"]))
    assert mines["MINE_ID"].is_unique
    assert set(acc["MINE_ID"]) <= set(mines["MINE_ID"])
