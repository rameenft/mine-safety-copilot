"""Paths and pinned data settings shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "mine_copilot.db"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

MSHA_BASE = "https://arlweb.msha.gov/OpenGovernmentData/DataSets"
ECFR_DATE = "2026-10-01"  # pinned eCFR snapshot of 30 CFR Part 56
ECFR_URL = f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}/title-30.xml?part=56"

INDEX_DIR = PROCESSED_DIR / "index"  # chunks.json + embeddings.npy
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
