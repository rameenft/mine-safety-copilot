"""Write small, committed test fixtures from the raw data.

Usage: python -m mine_copilot.ingest.fixtures   (needs data/raw from ingest.build)
"""

import xml.etree.ElementTree as ET

import pandas as pd

from mine_copilot import config
from mine_copilot.ingest.accidents import (
    KEEP_COLUMNS,
    MINE_COLUMNS,
    filter_accidents,
    read_pipe_file,
)

FIXTURE_SECTIONS = [
    "56.1000", "56.3200", "56.4100", "56.5001T", "56.9100", "56.9101", "56.9200", "56.9300",
    "56.9301", "56.11001", "56.12016", "56.12017", "56.14100", "56.14101", "56.14105",
    "56.14107", "56.14130", "56.15002", "56.15005", "56.18010",
]


def accident_fixture(seed: int = 7) -> pd.DataFrame:
    """~100 rows that pass the filters (all fatal ones first) + ~20 that must be dropped."""
    raw = pd.read_csv(config.RAW_DIR / "Accidents.txt", sep="|", dtype=str, encoding="latin-1",
                      usecols=KEEP_COLUMNS, keep_default_na=False)
    passing_ids = set(filter_accidents(read_pipe_file(config.RAW_DIR / "Accidents.txt", KEEP_COLUMNS))
                      ["DOCUMENT_NO"])
    passing = raw[raw["DOCUMENT_NO"].isin(passing_ids)]
    fatal = passing[passing["DEGREE_INJURY"].str.strip() == "FATALITY"].sample(10, random_state=seed)
    good = pd.concat([fatal, passing.drop(fatal.index).sample(90, random_state=seed)])

    yr = raw["CAL_YR"]
    rejects = pd.concat([
        raw[raw["COAL_METAL_IND"] == "C"].sample(5, random_state=seed),                # coal
        raw[(raw["COAL_METAL_IND"] == "M") & (yr == "2019")].sample(5, random_state=seed),  # year
        raw[(raw["COAL_METAL_IND"] == "M") & (yr == "2022")
            & (raw["SUBUNIT"] == "UNDERGROUND")].sample(5, random_state=seed),         # subunit
    ])
    no_narr = good.head(2).copy()  # synthetic: real rows with the narrative blanked
    no_narr["DOCUMENT_NO"] = no_narr["DOCUMENT_NO"] + "X"
    no_narr["NARRATIVE"] = ["?", ""]
    return pd.concat([good, rejects, no_narr]).sample(frac=1, random_state=seed)


def regulation_fixture() -> bytes:
    """Full Part 56 XML with every section outside FIXTURE_SECTIONS removed."""
    root = ET.parse(config.RAW_DIR / f"part56_{config.ECFR_DATE}.xml").getroot()
    keep = set(FIXTURE_SECTIONS)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "DIV8" and child.get("N") not in keep:
                parent.remove(child)
    return ET.tostring(root, encoding="utf-8")


def main() -> None:
    out = config.FIXTURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    acc = accident_fixture()
    acc.to_csv(out / "accidents_sample.txt", sep="|", index=False)
    mines = read_pipe_file(config.RAW_DIR / "Mines.txt", MINE_COLUMNS)
    mines[mines["MINE_ID"].isin(set(acc["MINE_ID"]))].to_csv(
        out / "mines_sample.txt", sep="|", index=False)
    (out / "part56_sample.xml").write_bytes(regulation_fixture())
    print(f"wrote {len(acc)} accident rows and {len(FIXTURE_SECTIONS)} sections to {out}")


if __name__ == "__main__":
    main()
