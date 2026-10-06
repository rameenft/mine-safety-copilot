"""Download raw data (if missing) and build the SQLite database.

Usage: python -m mine_copilot.ingest.build
"""

import gzip
import sqlite3
import urllib.request
import zipfile

import pandas as pd

from mine_copilot import config
from mine_copilot.ingest.accidents import load_accidents, load_mines
from mine_copilot.ingest.regulations import parse_part56


def download(url: str, dest, compressed: bool = False) -> None:
    if dest.exists():
        return
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"} if compressed else {})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    dest.write_bytes(body)


def fetch_raw() -> dict:
    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("Accidents", "Mines"):
        zip_path = raw / f"{name}.zip"
        download(f"{config.MSHA_BASE}/{name}.zip", zip_path)
        if not (raw / f"{name}.txt").exists():
            zipfile.ZipFile(zip_path).extractall(raw)
    xml_path = raw / f"part56_{config.ECFR_DATE}.xml"
    download(config.ECFR_URL, xml_path, compressed=True)
    return {"accidents": raw / "Accidents.txt", "mines": raw / "Mines.txt", "regs": xml_path}


def build_db(paths: dict, db_path=config.DB_PATH) -> dict:
    accidents = load_accidents(paths["accidents"])
    mines = load_mines(paths["mines"], set(accidents["MINE_ID"].dropna()))
    regs = pd.DataFrame(parse_part56(paths["regs"].read_bytes(), config.ECFR_DATE))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        accidents.to_sql("accidents", conn, if_exists="replace", index=False)
        mines.to_sql("mines", conn, if_exists="replace", index=False)
        regs.to_sql("regulations", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_acc_doc ON accidents(DOCUMENT_NO)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_reg_id ON regulations(section_id)")
    return {"accidents": len(accidents), "mines": len(mines), "regulations": len(regs)}


if __name__ == "__main__":
    print(build_db(fetch_raw()))
