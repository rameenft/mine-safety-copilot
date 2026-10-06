"""Load, filter and sample MSHA accident records; load the mines lookup."""

from pathlib import Path

import pandas as pd

KEEP_COLUMNS = [
    "DOCUMENT_NO", "MINE_ID", "ACCIDENT_DT", "CAL_YR", "COAL_METAL_IND", "SUBUNIT",
    "DEGREE_INJURY", "ACCIDENT_TYPE", "CLASSIFICATION", "MINING_EQUIP", "OCCUPATION",
    "ACTIVITY", "INJURY_SOURCE", "NATURE_INJURY", "DAYS_LOST", "NARRATIVE",
]
MINE_COLUMNS = ["MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE", "PRIMARY_SIC"]

# Part 56 covers surface metal/nonmetal mines. "SURFACE AT UNDERGROUND" falls under Part 57,
# and office/shop/other subunits are not mining operations.
SURFACE_SUBUNITS = {"STRIP, QUARY, OPEN PIT", "MILL OPERATION/PREPARATION PLANT", "DREDGE"}
YEARS = range(2021, 2025)
MISSING_TOKENS = {"?", "no value found", ""}

SEVERITY = {
    "FATALITY": "fatal",
    "PERM TOT OR PERM PRTL DISABLTY": "permanent_disability",
    "DAYS AWAY FROM WORK ONLY": "lost_time",
    "DYS AWY FRM WRK & RESTRCTD ACT": "lost_time",
    "DAYS RESTRICTED ACTIVITY ONLY": "restricted_duty",
    "NO DYS AWY FRM WRK,NO RSTR ACT": "no_lost_time",
}


def read_pipe_file(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read an MSHA pipe-delimited file (ASCII/latin-1), keeping only `columns`."""
    df = pd.read_csv(path, sep="|", dtype=str, encoding="latin-1",
                     usecols=columns, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip())
    return df.mask(df.apply(lambda col: col.str.lower().isin(MISSING_TOKENS)))


def filter_accidents(df: pd.DataFrame) -> pd.DataFrame:
    """Surface metal/nonmetal accidents, 2021-2024, with a narrative."""
    years = df["CAL_YR"].astype("Int64")
    keep = (
        (df["COAL_METAL_IND"] == "M")
        & years.isin(list(YEARS))
        & df["SUBUNIT"].isin(SURFACE_SUBUNITS)
        & df["NARRATIVE"].notna()
    )
    out = df[keep].copy()
    out["CAL_YR"] = out["CAL_YR"].astype(int)
    out["ACCIDENT_DT"] = pd.to_datetime(out["ACCIDENT_DT"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    out["DAYS_LOST"] = pd.to_numeric(out["DAYS_LOST"], errors="coerce").astype("Int64")
    out["SEVERITY"] = out["DEGREE_INJURY"].map(SEVERITY).fillna("other")
    return out


def sample_accidents(df: pd.DataFrame, cap: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Keep every fatality, then fill up to `cap` with a seeded random sample."""
    fatal = df[df["SEVERITY"] == "fatal"]
    rest = df[df["SEVERITY"] != "fatal"]
    n = max(0, min(cap - len(fatal), len(rest)))
    out = pd.concat([fatal, rest.sample(n=n, random_state=seed)])
    return out.sort_values("DOCUMENT_NO").reset_index(drop=True)


def load_accidents(path: Path, cap: int = 3000, seed: int = 42) -> pd.DataFrame:
    return sample_accidents(filter_accidents(read_pipe_file(path, KEEP_COLUMNS)), cap, seed)


def load_mines(path: Path, mine_ids: set[str] | None = None) -> pd.DataFrame:
    df = read_pipe_file(path, MINE_COLUMNS)
    if mine_ids is not None:
        df = df[df["MINE_ID"].isin(mine_ids)]
    return df.drop_duplicates("MINE_ID").reset_index(drop=True)
