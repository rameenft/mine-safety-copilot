"""Mine Safety Copilot: the whole project in one runnable file.

Usage: python mine_safety_copilot.py ["your question" ...]
Generated from src/ by scripts/build_single_file.py; do not edit by hand.
"""


# %% [markdown]
# # Mine Safety Copilot — the whole project in one file
#
# Grounded safety Q&A for US **surface metal/nonmetal** mines, answering only from **30 CFR Part 56**
# and **MSHA accident records (2021–2024)**. Every cited section and number is checked against tool
# output; out-of-scope questions are refused, half-in-scope ones get a partial answer.
#
# **To run:** set `GEMINI_API_KEY` (you'll be prompted otherwise) and run every cell. The first run
# downloads ~60 MB of public government data and embeds 459 regulation chunks (a few minutes, a few
# cents of API usage). Later runs reuse everything.
#
# This file is generated from the repo's `src/` package by `scripts/build_single_file.py`.


# %% [markdown]
# ## Imports


# %%
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from functools import cached_property, lru_cache
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Protocol
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


# %% [markdown]
# ## Configuration
#
# Paths and pinned settings. Everything is written under `COPILOT_HOME` (default `./mine_copilot_workspace`).
#
# *Source: `src/mine_copilot/config.py`*


# %%
ROOT = Path(os.environ.get("COPILOT_HOME", "mine_copilot_workspace")).resolve()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "mine_copilot.db"

MSHA_BASE = "https://arlweb.msha.gov/OpenGovernmentData/DataSets"
ECFR_DATE = "2026-10-01"  # pinned eCFR snapshot of 30 CFR Part 56
ECFR_URL = f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_DATE}/title-30.xml?part=56"

INDEX_DIR = PROCESSED_DIR / "index"  # chunks.json + embeddings.npy
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


# %% [markdown]
# ## M1 · Accident data: filter, severity, stratified sample
#
# Keeps surface metal/nonmetal accidents from 2021–2024 that have a narrative, maps MSHA's 11 injury codes to 6 `SEVERITY` values, then keeps **every fatality** plus a seeded random sample (stratified sampling) up to 3,000 rows.
#
# *Source: `src/mine_copilot/ingest/accidents.py`*


# %%
KEEP_COLUMNS = [
    "DOCUMENT_NO", "MINE_ID", "ACCIDENT_DT", "CAL_YR", "COAL_METAL_IND", "SUBUNIT",
    "DEGREE_INJURY", "ACCIDENT_TYPE", "CLASSIFICATION", "MINING_EQUIP", "OCCUPATION",
    "ACTIVITY", "INJURY_SOURCE", "NATURE_INJURY", "DAYS_LOST", "NARRATIVE",
]
MINE_COLUMNS = ["MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE", "PRIMARY_SIC"]

# Part 56 covers surface metal/nonmetal mines. "SURFACE AT UNDERGROUND" falls under Part 57,
# and office/shop/other subunits are not mining operations.
SURFACE_SUBUNITS = {"STRIP, QUARY, OPEN PIT", "MILL OPERATION/PREPARATION PLANT", "DREDGE"}
ACCIDENT_YEARS = range(2021, 2025)
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
        & years.isin(list(ACCIDENT_YEARS))
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


# %% [markdown]
# ## M1 · Regulations: parse eCFR XML, structure-aware chunking
#
# Parses 30 CFR Part 56 into one record per section. Long sections are split on paragraph boundaries (≤1,500 chars) and every chunk keeps its section ID, so a search hit can always be cited and the full section fetched.
#
# *Source: `src/mine_copilot/ingest/regulations.py`*


# %%
# XREF holds editorial "link to an amendment" notes, not regulatory text.
SKIP_TAGS = {"HEAD", "CITA", "AUTH", "SOURCE", "XREF"}
# IDs can carry a suffix, e.g. 56.5001T (a newer version printed alongside the old one).
HEAD_RE = re.compile(r"^§\s*(?P<id>56\.\d+[A-Z]?)\s*(?P<heading>.*?)\.?$")


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_part56(xml: bytes | str, ecfr_date: str) -> list[dict]:
    """Return [{section_id, subpart, heading, text, source_url}] for every non-reserved section."""
    root = ET.fromstring(xml)
    records = []
    for subpart in root.iter("DIV6"):
        subpart_head = subpart.find("HEAD")
        subpart_name = _text(subpart_head) if subpart_head is not None else ""
        for sec in subpart.iter("DIV8"):
            if sec.get("TYPE") != "SECTION":
                continue
            match = HEAD_RE.match(_text(sec.find("HEAD")))
            if not match or "[Reserved]" in match["heading"]:
                continue
            paragraphs = [_text(child) for child in sec if child.tag not in SKIP_TAGS]
            text = "\n\n".join(p for p in paragraphs if p)
            if not text:
                continue
            sid = match["id"]
            records.append({
                "section_id": sid,
                "subpart": subpart_name,
                "heading": match["heading"].strip(),
                "text": text,
                "source_url": f"https://www.ecfr.gov/on/{ecfr_date}/title-30/section-{sid}",
            })
    return records


def chunk_section(record: dict, max_chars: int = 1500) -> list[dict]:
    """One chunk per section; long sections are split on paragraph boundaries.

    Every chunk keeps the section_id and heading so citations survive splitting.
    """
    paragraphs = record["text"].split("\n\n")
    groups, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            groups.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    groups.append(current)
    return [
        {**record, "chunk_id": f"{record['section_id']}#{i}", "text": text}
        for i, text in enumerate(groups)
    ]


# %% [markdown]
# ## M1 · Download and build the SQLite database
#
# Downloads the MSHA files and the pinned eCFR snapshot (skipped if already present) and loads three tables: `accidents`, `mines`, `regulations`.
#
# *Source: `src/mine_copilot/ingest/build.py`*


# %%
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
    raw = RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("Accidents", "Mines"):
        zip_path = raw / f"{name}.zip"
        download(f"{MSHA_BASE}/{name}.zip", zip_path)
        if not (raw / f"{name}.txt").exists():
            zipfile.ZipFile(zip_path).extractall(raw)
    xml_path = raw / f"part56_{ECFR_DATE}.xml"
    download(ECFR_URL, xml_path, compressed=True)
    return {"accidents": raw / "Accidents.txt", "mines": raw / "Mines.txt", "regs": xml_path}


def build_db(paths: dict, db_path=DB_PATH) -> dict:
    accidents = load_accidents(paths["accidents"])
    mines = load_mines(paths["mines"], set(accidents["MINE_ID"].dropna()))
    regs = pd.DataFrame(parse_part56(paths["regs"].read_bytes(), ECFR_DATE))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        accidents.to_sql("accidents", conn, if_exists="replace", index=False)
        mines.to_sql("mines", conn, if_exists="replace", index=False)
        regs.to_sql("regulations", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_acc_doc ON accidents(DOCUMENT_NO)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_reg_id ON regulations(section_id)")
    return {"accidents": len(accidents), "mines": len(mines), "regulations": len(regs)}


# %% [markdown]
# ## M3 · Retrieval index: BM25, dense, RRF hybrid
#
# Three search modes over the chunks. Dense (embeddings + cosine similarity on a NumPy matrix, no vector DB) is the default because it measured best on the golden set; section IDs named in a query are pinned to the top.
#
# *Source: `src/mine_copilot/retrieval/index.py`*


# %%
SECTION_RE = re.compile(r"\b56\.\d+[A-Z]?\b")
TOKEN_RE = re.compile(r"56\.\d+[a-z]?|[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "in",
    "is", "it", "must", "of", "on", "or", "shall", "that", "the", "to", "what", "when", "which",
    "with",
}
RRF_K = 60  # standard RRF damping constant (Cormack et al., 2009)
DENSE_WEIGHT = 2.0  # equal-weight RRF let weak BM25 ranks drag dense hits down (see eval report)
DEFAULT_MODE = "dense"  # best recall on the golden set; hybrid kept for comparison


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def doc_text(chunk: dict) -> str:
    """What gets indexed: ID + heading prepended so every chunk carries its topic."""
    return f"§ {chunk['section_id']} {chunk['heading']}\n{chunk['text']}"


def rrf(rankings: list[list[int]], weights: list[float] | None = None, k: int = RRF_K) -> list[int]:
    """Fuse ranked lists of item indices; items ranked high in any list float up."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class RegIndex:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray | None = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(doc_text(c)) for c in chunks])
        self.section_ids = {c["section_id"] for c in chunks}

    @classmethod
    def from_sections(cls, sections: list[dict], embed_fn=None) -> "RegIndex":
        chunks = [c for s in sections for c in chunk_section(s)]
        emb = embed_fn([doc_text(c) for c in chunks]) if embed_fn else None
        return cls(chunks, emb)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(json.dumps(self.chunks))
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Path) -> "RegIndex":
        chunks = json.loads((directory / "chunks.json").read_text())
        emb_path = directory / "embeddings.npy"
        return cls(chunks, np.load(emb_path) if emb_path.exists() else None)

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [i for i in np.argsort(-scores) if scores[i] > 0]

    def _dense_rank(self, query_vec: np.ndarray) -> list[int]:
        return list(np.argsort(-(self.embeddings @ query_vec)))

    def search(self, query: str, k: int = 5, mode: str = DEFAULT_MODE, query_vec=None) -> list[dict]:
        """mode: 'bm25' | 'dense' | 'hybrid'. Without embeddings, falls back to BM25."""
        if self.embeddings is None:
            mode = "bm25"
        if mode != "bm25" and query_vec is None:

            query_vec = embed_query(query)

        if mode == "bm25":
            order = self._bm25_rank(query)
        elif mode == "dense":
            order = self._dense_rank(query_vec)
        else:
            order = rrf([self._bm25_rank(query), self._dense_rank(query_vec)], [1.0, DENSE_WEIGHT])

        # Explicitly named sections ("what does 56.14107 say") are pinned to the top.
        pinned = [s for s in dict.fromkeys(SECTION_RE.findall(query)) if s in self.section_ids]
        hits, seen = [], set()
        for sid in pinned:
            first = next(c for c in self.chunks if c["section_id"] == sid)
            hits.append({**first, "pinned": True})
            seen.add(sid)
        for i in order:
            chunk = self.chunks[i]
            if chunk["section_id"] in seen:
                continue
            seen.add(chunk["section_id"])
            hits.append({**chunk, "pinned": False})
            if len(hits) >= k:
                break
        return hits[:k]


# %% [markdown]
# ## M3 · Gemini embeddings
#
# Batched, L2-normalised embeddings with a retry when the per-minute quota is hit.
#
# *Source: `src/mine_copilot/retrieval/embed.py`*


# %%
BATCH = 100  # max texts per request; free tier also caps ~100 texts/minute
EMBED_RETRY_WAIT = 65  # seconds to wait out a per-minute quota window


@lru_cache(maxsize=1)
def _client():
    from google import genai

    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(texts: list[str], task_type: str) -> np.ndarray:
    """task_type is RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    from google.genai import errors, types

    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    vecs = []
    for i in range(0, len(texts), BATCH):
        for attempt in range(5):
            try:
                resp = _client().models.embed_content(
                    model=EMBED_MODEL, contents=texts[i : i + BATCH], config=cfg
                )
                break
            except errors.ClientError as e:
                if e.code != 429 or attempt == 4:
                    raise
                print(f"rate-limited at {i}/{len(texts)}, waiting {EMBED_RETRY_WAIT}s")
                time.sleep(EMBED_RETRY_WAIT)
        vecs.extend(e.values for e in resp.embeddings)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@lru_cache(maxsize=512)
def embed_query(query: str) -> np.ndarray:
    return embed([query], "RETRIEVAL_QUERY")[0]


# %% [markdown]
# ## M3 · Build the index
#
# Chunks every section, embeds the chunks once and saves `chunks.json` + `embeddings.npy`.
#
# *Source: `src/mine_copilot/retrieval/build.py`*


# %%
def load_sections() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM regulations ORDER BY rowid")]


def build_index(use_embeddings: bool = True) -> RegIndex:
    embed_fn = None
    if use_embeddings:
        gemini_embed = embed

        def embed_fn(texts):
            return gemini_embed(texts, "RETRIEVAL_DOCUMENT")

    index = RegIndex.from_sections(load_sections(), embed_fn)
    index.save(INDEX_DIR)
    dims = "none" if index.embeddings is None else index.embeddings.shape
    print(f"indexed {len(index.chunks)} chunks / {len(index.section_ids)} sections, emb={dims}")
    return index


# %% [markdown]
# ## M4 · LLM provider interface
#
# Every model call goes through `Provider.chat`, so swapping Gemini for another model is a config change. `FakeProvider` replays scripted replies for offline tests.
#
# *Source: `src/mine_copilot/llm.py`*


# %%
RETRY_WAIT = {429: 65, 503: 10}  # seconds: quota resets per minute; overload clears fast


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "args": dict}]
    raw: Any = None  # provider-native turn, replayed verbatim (Gemini 3 needs thought signatures)
    usage: dict = field(default_factory=dict)  # {"input": tokens, "output": tokens incl. thinking}


class Provider(Protocol):
    def chat(self, system: str, messages: list[dict], tools: list[dict] | None) -> Reply: ...


class FakeProvider:
    """Returns scripted replies in order; records what it was sent. For offline tests."""

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class GeminiProvider:
    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def _contents(self, messages: list[dict]):
        from google.genai import types

        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=m["text"])]))
            elif m["role"] == "assistant":
                out.append(m.get("raw") or types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=m["text"] or " ")]
                    + [types.Part.from_function_call(name=c["name"], args=c["args"])
                       for c in m.get("tool_calls", [])],
                ))
            elif m["role"] == "tool":
                part = types.Part.from_function_response(name=m["name"], response=m["content"])
                # Consecutive tool results go in one user turn, matching the model's call turn.
                if out and out[-1].role == "user" and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(types.Content(role="user", parts=[part]))
        return out

    def chat(self, system, messages, tools):
        from google.genai import errors, types

        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters_json_schema=t["parameters"])
                 for t in tools or []]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
        )
        for attempt in range(4):
            try:
                resp = _gemini_client().models.generate_content(
                    model=self.model, contents=self._contents(messages), config=cfg
                )
                break
            except errors.APIError as e:
                # A per-day quota won't reset in a minute; fail fast instead of sleeping.
                if e.code not in (429, 503) or attempt == 3 or "PerDay" in str(e):
                    raise
                print(f"[llm] {e.code}, retrying in {RETRY_WAIT[e.code]}s")
                time.sleep(RETRY_WAIT[e.code])

        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts if content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [{"name": p.function_call.name, "args": dict(p.function_call.args or {})}
                 for p in parts if p.function_call]
        u = resp.usage_metadata
        usage = {"input": (u.prompt_token_count or 0) if u else 0,
                 "output": ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0}
        return Reply(text=text.strip(), tool_calls=calls, raw=content, usage=usage)


# %% [markdown]
# ## M4 · The agent's three tools (and their guardrails)
#
# `search_regulations`, `get_regulation` and `accident_stats`. Stats use allow-listed filters and bound parameters on a read-only DB (no text-to-SQL). Expired dated sections (56.5001 vs 56.5001T) are hidden and redirected here, in code, because the model ignored prompt rules and labels about them.
#
# *Source: `src/mine_copilot/agent/tools.py`*


# %%
YEARS = (2021, 2024)
SEVERITIES = ["fatal", "permanent_disability", "lost_time", "restricted_duty", "no_lost_time",
              "other"]
# Text filters: case-insensitive substring match against these columns.
TEXT_FILTERS = {
    "classification": "a.CLASSIFICATION",
    "accident_type": "a.ACCIDENT_TYPE",
    "subunit": "a.SUBUNIT",
    "equipment": "a.MINING_EQUIP",
    "occupation": "a.OCCUPATION",
}
GROUP_BY = {
    "year": "a.CAL_YR", "severity": "a.SEVERITY", "state": "m.STATE",
    **TEXT_FILTERS,
}
METRICS = {
    "count": "COUNT(*)",
    "avg_days_lost": "ROUND(AVG(a.DAYS_LOST), 1)",
    "total_days_lost": "SUM(a.DAYS_LOST)",
}
SAMPLE_NOTE = ("Every fatality 2021-2024 is included; non-fatal rows are a random fill up to "
               "3,000 rows. Counts that include non-fatal rows describe this sample, not all "
               "accidents, and the sample over-represents severe accidents.")
SNIPPET_CHARS = 600
# Dated versions (56.5001 "required until April 7, 2026" / 56.5001T "As of April 8, 2026").
# The model doesn't know today's date, so the tool labels which version is in force.
UNTIL_RE = re.compile(r"required until (\w+ \d{1,2}, \d{4})")
AS_OF_RE = re.compile(r"^As of (\w+ \d{1,2}, \d{4})")


def section_status(section_id: str, text: str, today: date | None = None) -> str | None:
    """'EXPIRED …' / 'IN FORCE …' / 'NOT YET IN FORCE …' for dated sections, else None."""
    today = today or date.today()  # noqa: DTZ011 - calendar dates, no time zone involved
    if m := UNTIL_RE.search(text[:200]):
        until = datetime.strptime(m.group(1), "%B %d, %Y").date()  # noqa: DTZ007
        if until < today:
            return (f"EXPIRED after {m.group(1)}; do not cite as current. "
                    f"Use § {section_id}T if it exists.")
        return f"IN FORCE until {m.group(1)}"
    if m := AS_OF_RE.search(text):
        start = datetime.strptime(m.group(1), "%B %d, %Y").date()  # noqa: DTZ007
        return (f"IN FORCE since {m.group(1)}" if start <= today
                else f"NOT YET IN FORCE (starts {m.group(1)})")
    return None


class Tools:
    def __init__(self, db_path: Path = DB_PATH, index=None, search_mode: str | None = None):
        load_dotenv()
        self.db_path = db_path
        self._index = index
        # Dense search needs the embedding API; without a key fall back to BM25.
        self.search_mode = search_mode or ("dense" if os.environ.get("GEMINI_API_KEY") else "bm25")

    @cached_property
    def index(self):
        if self._index is None:

            self._index = RegIndex.load(INDEX_DIR)
        return self._index

    def _query(self, sql: str, params: list) -> list[sqlite3.Row]:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()

    # --- tools -------------------------------------------------------------------------------

    @cached_property
    def statuses(self) -> dict[str, str]:
        rows = self._query("SELECT section_id, text FROM regulations", [])
        return {r["section_id"]: st for r in rows
                if (st := section_status(r["section_id"], r["text"]))}

    def search_regulations(self, query: str, k: int = 5) -> dict:
        k = min(int(k), 8)
        expired = sum(st.startswith("EXPIRED") for st in self.statuses.values())
        hits = self.index.search(query, k=k + expired, mode=self.search_mode)  # refill hidden ones
        results = []
        for h in hits:
            # Expired dated sections are hidden: the model was seen citing them despite a label.
            if self.statuses.get(h["section_id"], "").startswith("EXPIRED"):
                continue
            r = {"section_id": h["section_id"], "heading": h["heading"],
                 "snippet": h["text"][:SNIPPET_CHARS]}
            if st := self.statuses.get(h["section_id"]):
                r["status"] = st
            results.append(r)
        return {"results": results[:k]}

    def get_regulation(self, section_id: str) -> dict:
        sid = section_id.strip().removeprefix("§").strip().removeprefix("30 CFR").strip()
        rows = self._query("SELECT section_id, heading, text, source_url FROM regulations "
                           "WHERE section_id = ?", [sid])
        if not rows:
            return {"error": f"Section {sid} not found in 30 CFR Part 56."}
        out = dict(rows[0])
        st = section_status(sid, out["text"])
        if st and st.startswith("EXPIRED") and (new := self.get_regulation(sid + "T")).get("text"):
            new["note"] = (f"§ {sid} expired; this is its replacement, § {sid}T, in force today. "
                           f"Cite § {sid}T.")
            return new
        if st:
            out["status"] = st
        return out

    def accident_stats(self, severity: str | None = None, year_from: int | None = None,
                       year_to: int | None = None, narrative_contains: list[str] | None = None,
                       group_by: str | None = None, metric: str = "count", limit: int = 10,
                       **text_filters) -> dict:
        unknown = set(text_filters) - set(TEXT_FILTERS)
        if unknown:
            return {"error": f"Unknown filter(s) {sorted(unknown)}. Allowed: "
                             f"{sorted(TEXT_FILTERS)} + severity, year_from, year_to, "
                             "narrative_contains."}
        if severity is not None and severity not in SEVERITIES:
            return {"error": f"severity must be one of {SEVERITIES}"}
        if group_by is not None and group_by not in GROUP_BY:
            return {"error": f"group_by must be one of {sorted(GROUP_BY)}"}
        if metric not in METRICS:
            return {"error": f"metric must be one of {sorted(METRICS)}"}
        for y in (year_from, year_to):
            if y is not None and not YEARS[0] <= int(y) <= YEARS[1]:
                return {"error": f"Year {y} is outside data coverage ({YEARS[0]}-{YEARS[1]})."}

        where, params = [], []
        if severity:
            where.append("a.SEVERITY = ?")
            params.append(severity)
        if year_from is not None:
            where.append("a.CAL_YR >= ?")
            params.append(int(year_from))
        if year_to is not None:
            where.append("a.CAL_YR <= ?")
            params.append(int(year_to))
        for name, value in text_filters.items():
            if value:
                where.append(f"{TEXT_FILTERS[name]} LIKE ?")
                params.append(f"%{value}%")
        terms = [t for t in (narrative_contains or []) if t.strip()]
        if terms:  # any term matches
            where.append("(" + " OR ".join("a.NARRATIVE LIKE ?" for _ in terms) + ")")
            params.extend(f"%{t}%" for t in terms)

        base = "FROM accidents a LEFT JOIN mines m USING (MINE_ID)"
        cond = f" WHERE {' AND '.join(where)}" if where else ""
        total = self._query(f"SELECT COUNT(*) {base}{cond}", params)[0][0]
        examples = [r[0] for r in self._query(
            f"SELECT a.DOCUMENT_NO {base}{cond} ORDER BY a.ACCIDENT_DT DESC LIMIT 3", params)]

        agg = METRICS[metric]
        if group_by:
            col = GROUP_BY[group_by]
            rows = self._query(f"SELECT {col} AS grp, {agg} AS value {base}{cond} "
                               f"GROUP BY grp ORDER BY value DESC LIMIT ?",
                               params + [min(int(limit), 25)])
            result = [{group_by: r["grp"], metric: r["value"]} for r in rows]
        else:
            result = [{metric: self._query(f"SELECT {agg} {base}{cond}", params)[0][0]}]

        return {"rows": result, "total_matching": total, "example_document_nos": examples,
                "note": SAMPLE_NOTE}


# JSON schemas the model sees. Kept next to the implementations so they can't drift far.
TOOL_SPECS = [
    {
        "name": "search_regulations",
        "description": "Semantic search over 30 CFR Part 56 (surface metal/nonmetal mine safety "
                       "standards). Returns section IDs, headings and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused topic, e.g. 'conveyor "
                                                           "guarding' or 'berms on haul roads'."},
                "k": {"type": "integer", "description": "Number of sections (default 5, max 8)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_regulation",
        "description": "Full text of one Part 56 section by ID, e.g. '56.14107'.",
        "parameters": {
            "type": "object",
            "properties": {"section_id": {"type": "string"}},
            "required": ["section_id"],
        },
    },
    {
        "name": "accident_stats",
        "description": "Count or aggregate MSHA surface metal/nonmetal accident records "
                       f"({YEARS[0]}-{YEARS[1]}). Returns rows, total matching records, example "
                       "document numbers, and a sampling note.",
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": SEVERITIES,
                             "description": "'other' mixes occupational illness, natural-cause "
                                            "deaths/injuries, non-employee injuries and no-injury "
                                            "events; say so when reporting it."},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "classification": {"type": "string", "description": "e.g. MACHINERY, "
                                   "POWERED HAULAGE, HANDLING OF MATERIALS, SLIP OR FALL OF "
                                   "PERSON (substring match)."},
                "accident_type": {"type": "string", "description": "Substring, e.g. 'Struck by'."},
                "subunit": {"type": "string", "description": "STRIP/QUARRY/OPEN PIT, MILL, or "
                            "DREDGE (substring)."},
                "equipment": {"type": "string", "description": "Mining equipment (substring)."},
                "occupation": {"type": "string", "description": "Occupation (substring)."},
                "narrative_contains": {"type": "array", "items": {"type": "string"},
                                       "description": "Keywords; a record matches if its "
                                       "narrative contains ANY of them."},
                "group_by": {"type": "string", "enum": sorted(GROUP_BY)},
                "metric": {"type": "string", "enum": sorted(METRICS),
                           "description": "Default count."},
                "limit": {"type": "integer", "description": "Max groups (default 10)."},
            },
        },
    },
]


def call_tool(tools: Tools, name: str, args: dict) -> dict:
    fn = {"search_regulations": tools.search_regulations,
          "get_regulation": tools.get_regulation,
          "accident_stats": tools.accident_stats}.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}."}
    try:
        return fn(**args)
    except TypeError as e:  # bad/missing arguments from the model
        return {"error": f"Bad arguments for {name}: {e}"}


# %% [markdown]
# ## M4 · Agent loop and grounding check
#
# The system prompt (scope, sample caveats, partial answers with `Not covered:`), a loop of at most 5 tool turns, and a post-answer check that every cited section and number appears in the tool output.
#
# *Source: `src/mine_copilot/agent/loop.py`*


# %%
MAX_TOOL_TURNS = 5
FINAL_NUDGE = "Tool budget used up. Answer now using only the tool results above."
REFUSAL_PREFIX = "Out of scope:"
PARTIAL_PREFIX = "Not covered:"
TODAY = date.today()  # noqa: DTZ011 - the model doesn't know the date; tell it

SYSTEM = f"""You are Mine Safety Copilot, a safety assistant for US SURFACE metal/nonmetal mines.
Your only sources are tool results: 30 CFR Part 56 regulations and MSHA accident records
({YEARS[0]}-{YEARS[1]}). Today's date is {TODAY:%B %d, %Y}.

Rules:
1. Always use tools before answering. Never answer from memory.
2. Regulation questions: search_regulations, then get_regulation for the section you rely on.
   Cite sections as "§ 56.xxxx". Some sections have a dated "T" version (e.g. 56.5001T). The
   tools return only the version in force today; trust its status and dates over your memory.
3. Statistics: use accident_stats; report numbers exactly as returned. If a number includes
   non-fatal records, your first sentence must say it comes from a 3,000-record sample. Every
   fatality is kept, so the sample over-represents severe accidents: never call it representative.
   Fatal-only counts are complete. Counts from narrative_contains are keyword matches: say
   accidents "mentioning" the term, never "caused by" it. You may cite example document numbers.
4. Questions with both parts: answer both, each grounded in its own tool result.
5. Refuse only when the WHOLE question is out of scope: coal mines, underground mines
   (Part 57/75), years outside {YEARS[0]}-{YEARS[1]}, legal/financial advice, or anything the tools
   cannot answer. A refusal starts with "{REFUSAL_PREFIX}" and says briefly what you do cover.
6. If only PART of a question is in scope, answer that part with citations, then add a final
   line starting "{PARTIAL_PREFIX}" naming what you can't answer and where to look. Don't
   answer the uncovered part. Pointers you may give: training is 30 CFR Part 46; penalties are
   30 CFR Part 100; respirable crystalline silica limits are 30 CFR Part 60; underground metal/nonmetal is Part 57; coal is Parts 70-75; non-mine
   workplaces are OSHA; years outside {YEARS[0]}-{YEARS[1]} are on MSHA's data portal.
7. If tools return nothing relevant, say so rather than guessing.
Be concise: a direct answer first, then the supporting citation(s)."""

CITATION_RE = re.compile(r"56\.\d+[A-Z]?\b")
DOC_RE = re.compile(r"\b\d{12}\b")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "7. AZ: 363" -> 7 isn't a claim
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
ALWAYS_OK = {"30", "56", "57", "75"}  # "30 CFR Part 56" etc.
# Regulations spell small numbers out ("seven feet"); the model often writes digits.
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}


@dataclass
class AgentResult:
    question: str
    answer: str
    refused: bool
    citations: dict = field(default_factory=dict)  # {"sections": [...], "documents": [...]}
    ungrounded: dict = field(default_factory=dict)  # {"sections": [...], "numbers": [...]}
    trace: list[dict] = field(default_factory=list)  # [{"tool", "args", "result"}]

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def _evidence_numbers(text: str) -> set[str]:
    words = {NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}
    return _numbers(text) | words


def ground(answer: str, question: str, trace: list[dict]) -> tuple[dict, dict]:
    """Split the answer's claims into grounded citations and ungrounded leftovers."""
    evidence = " ".join(json.dumps(t["result"]) for t in trace)
    seen_sections = set(CITATION_RE.findall(evidence))
    seen_docs = set(DOC_RE.findall(evidence))

    sections = list(dict.fromkeys(CITATION_RE.findall(answer)))
    docs = list(dict.fromkeys(DOC_RE.findall(answer)))
    # Strip cited IDs before checking loose numbers so "56.14107" isn't read as two numbers.
    rest = LIST_MARKER_RE.sub(" ", DOC_RE.sub(" ", CITATION_RE.sub(" ", answer)))
    # SYSTEM is evidence too: it states the coverage years a refusal may repeat.
    allowed = _evidence_numbers(evidence + " " + SYSTEM) | _numbers(question) | ALWAYS_OK
    loose = sorted(_numbers(rest) - allowed, key=lambda n: float(n))

    citations = {"sections": [s for s in sections if s in seen_sections],
                 "documents": [d for d in docs if d in seen_docs]}
    ungrounded = {"sections": [s for s in sections if s not in seen_sections],
                  "documents": [d for d in docs if d not in seen_docs],
                  "numbers": loose}
    return citations, ungrounded


def run(question: str, provider: Provider, tools: Tools | None = None,
        max_turns: int = MAX_TOOL_TURNS) -> AgentResult:
    tools = tools or Tools()
    messages: list[dict] = [{"role": "user", "text": question}]
    trace: list[dict] = []

    for turn in range(max_turns + 1):
        # Last turn: withhold tools and say so; without the nudge Gemini can return empty text.
        if turn == max_turns:
            messages.append({"role": "user", "text": FINAL_NUDGE})
        reply = provider.chat(SYSTEM, messages, TOOL_SPECS if turn < max_turns else None)
        messages.append({"role": "assistant", "text": reply.text,
                         "tool_calls": reply.tool_calls, "raw": reply.raw})
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            result = call_tool(tools, call["name"], call["args"])
            trace.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append({"role": "tool", "name": call["name"], "content": result})

    answer = reply.text or "I could not produce an answer from the available sources."
    refused = answer.startswith(REFUSAL_PREFIX)
    citations, ungrounded = ground(answer, question, trace)
    return AgentResult(question, answer, refused, citations, ungrounded, trace)


# %% [markdown]
# ## M2 · Golden set (the exam)
#
# 32 questions. Expected facts are checked verbatim against section text; expected numbers come from SQL at build time; phrase rules come from the manual review.
#
# *Source: `evals/build_golden.py`*


# %%
GOLDEN_SAMPLE_NOTE = "Non-fatal counts describe the 3,000-row sample, not the population; say so."
# Phrase checks added after manual review (M8): stats that include non-fatal rows must flag the
# sample up front and never call it representative; keyword counts must say "mention", not "cause".
SAMPLE_CHECKS = {"lead_phrases": ["sample"], "forbidden_phrases": ["representative"]}
MENTION_CHECK = {"required_phrases": ["mention"]}
PARTIAL_CHECK = {"required_phrases": ["Not covered:"]}

QUESTIONS = [
    # --- reg: answer lives in one Part 56 section ---
    {"id": "reg-01", "type": "reg",
     "question": "What does MSHA require for moving machine parts like pulleys and flywheels?",
     "expected_sections": ["56.14107"],
     "expected_facts": ["shall be guarded", "pulleys"]},
    {"id": "reg-02", "type": "reg",
     "question": "When must miners wear safety belts and lines?",
     "expected_sections": ["56.15005"],
     "expected_facts": ["danger of falling", "second person shall tend the lifeline"]},
    {"id": "reg-03", "type": "reg",
     "question": "How high must berms be on haul roads?",
     "expected_sections": ["56.9300"],
     "expected_facts": ["mid-axle height"]},
    {"id": "reg-04", "type": "reg",
     "question": "What must be done before mechanical work on electrically powered equipment?",
     "expected_sections": ["56.12016"],
     "expected_facts": ["deenergized", "locked out"]},
    {"id": "reg-05", "type": "reg",
     "question": "What are the rules for doing repairs or maintenance on machinery?",
     "expected_sections": ["56.14105"],
     "expected_facts": ["power is off", "blocked against hazardous motion"]},
    {"id": "reg-06", "type": "reg",
     "question": "How often must working places be examined, and by whom?",
     "expected_sections": ["56.18002"],
     "expected_facts": ["competent person", "at least once each shift"]},
    {"id": "reg-07", "type": "reg",
     "question": "Who has to inspect self-propelled mobile equipment before a shift?",
     "expected_sections": ["56.14100"],
     "expected_facts": ["inspected by the equipment operator"]},
    {"id": "reg-08", "type": "reg",
     "question": "Which equipment needs roll-over protective structures (ROPS)?",
     "expected_sections": ["56.14130"],
     "expected_facts": ["Crawler tractors", "Graders", "Wheel loaders"]},
    {"id": "reg-09", "type": "reg",
     "question": "When are hard hats required?",
     "expected_sections": ["56.15002"],
     "expected_facts": ["falling objects"]},
    {"id": "reg-10", "type": "reg",
     "question": "What does Part 56 require for respirators when miners are exposed to dust?",
     "expected_sections": ["56.5005T"], "forbidden_sections": ["56.5005"],
     "expected_facts": ["written respiratory protection program", "ASTM F3387-19"],
     "notes": "56.5005 expired April 7, 2026; 56.5005T is in force."},
    # --- agg: answer is a SQL aggregate over accidents ---
    {"id": "agg-01", "type": "agg",
     "question": "How many fatal accidents were recorded in 2023?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023"},
    {"id": "agg-02", "type": "agg",
     "question": "How many fatalities were there in total from 2021 to 2024?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal'"},
    {"id": "agg-03", "type": "agg",
     "question": "Which year had the most fatal accidents?",
     "sql": "SELECT CAL_YR FROM accidents WHERE SEVERITY='fatal' "
            "GROUP BY CAL_YR ORDER BY COUNT(*) DESC LIMIT 1"},
    {"id": "agg-04", "type": "agg",
     "question": "What accident classification caused the most fatalities in 2023?",
     "sql": "SELECT CLASSIFICATION FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023 "
            "GROUP BY CLASSIFICATION ORDER BY COUNT(*) DESC LIMIT 1"},
    {"id": "agg-05", "type": "agg",
     "question": "What is the most common accident classification overall?",
     "sql": "SELECT CLASSIFICATION FROM accidents "
            "GROUP BY CLASSIFICATION ORDER BY COUNT(*) DESC LIMIT 1",
     "notes": GOLDEN_SAMPLE_NOTE, **SAMPLE_CHECKS},
    {"id": "agg-06", "type": "agg",
     "question": "Which state has the most recorded accidents?",
     "sql": "SELECT m.STATE FROM accidents a JOIN mines m USING(MINE_ID) "
            "GROUP BY m.STATE ORDER BY COUNT(*) DESC LIMIT 1",
     "notes": GOLDEN_SAMPLE_NOTE, **SAMPLE_CHECKS},
    {"id": "agg-07", "type": "agg",
     "question": "On average, how many days were lost in lost-time accidents?",
     "sql": "SELECT ROUND(AVG(DAYS_LOST), 1) FROM accidents WHERE SEVERITY='lost_time'",
     "notes": GOLDEN_SAMPLE_NOTE, **SAMPLE_CHECKS},
    # --- hybrid: accident records + the regulation that applies ---
    {"id": "hyb-01", "type": "hybrid",
     "question": "How many fatal accidents involved conveyors, and what rule covers guarding them?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%conveyor%'",
     "expected_sections": ["56.14107"], **MENTION_CHECK},
    {"id": "hyb-02", "type": "hybrid",
     "question": "Were there fatalities involving berms, and what does Part 56 require for them?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%berm%'",
     "expected_sections": ["56.9300"], **MENTION_CHECK},
    {"id": "hyb-03", "type": "hybrid",
     "question": "How many fatal accidents mention a harness or fall protection, "
                 "and which rule applies?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' "
            "AND (NARRATIVE LIKE '%fall protection%' OR NARRATIVE LIKE '%harness%')",
     "expected_sections": ["56.15005"], **MENTION_CHECK},
    {"id": "hyb-04", "type": "hybrid",
     "question": "How many 2023 fatalities were classified as MACHINERY, and what rule covers "
                 "maintenance on machinery?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023 "
            "AND CLASSIFICATION='MACHINERY'",
     "expected_sections": ["56.14105"]},
    # --- partial: one part in scope, one not. Answer the covered part, then "Not covered:" ---
    {"id": "part-01", "type": "partial",
     "question": "Do new quarry workers need hard hats, and what safety training must they "
                 "complete before starting?",
     "expected_sections": ["56.15002"], "notes": "Training is 30 CFR Part 46, not Part 56.",
     **PARTIAL_CHECK},
    {"id": "part-02", "type": "partial",
     "question": "What are the berm requirements on haul roads, and how much does it cost to "
                 "build one?",
     "expected_sections": ["56.9300"], "notes": "Costs are not in any source.", **PARTIAL_CHECK},
    {"id": "part-03", "type": "partial",
     "question": "What does Part 56 say about guarding conveyors, and what does OSHA require for "
                 "conveyors in a warehouse?",
     "expected_sections": ["56.14107"], "notes": "OSHA / non-mine workplaces out of scope.",
     **PARTIAL_CHECK},
    {"id": "part-04", "type": "partial",
     "question": "What must be done before working on electrical equipment, and what is the "
                 "fine if MSHA finds we skipped it?",
     "expected_sections": ["56.12016"], "notes": "Penalties are 30 CFR Part 100.",
     **PARTIAL_CHECK},
    {"id": "part-05", "type": "partial",
     "question": "How many fatal accidents mentioned conveyors from 2021 through 2025?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%conveyor%'",
     "notes": "2025 is outside the data; give the 2021-2024 count.", **PARTIAL_CHECK},
    {"id": "part-06", "type": "partial",
     "question": "How many fatal accidents were there at surface metal/nonmetal mines in 2024, "
                 "and how many at underground mines?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2024",
     "notes": "Underground (Part 57) data not included.", **PARTIAL_CHECK},
    {"id": "part-07", "type": "partial",
     "question": "What is the exposure limit for respirable crystalline silica at my quarry?",
     "expected_sections": ["56.5001T"], "forbidden_sections": ["56.5001"],
     "notes": "56.5001T (in force) defers silica to 30 CFR Part 60, which is not in the corpus.",
     **PARTIAL_CHECK},
    # --- refuse: outside scope, the copilot must decline ---
    {"id": "ref-01", "type": "refuse",
     "question": "What does 30 CFR Part 75 require for roof bolting in underground coal mines?",
     "notes": "Coal / Part 75 out of scope."},
    {"id": "ref-02", "type": "refuse",
     "question": "What are the ventilation requirements for underground metal mines?",
     "notes": "Underground (Part 57) out of scope."},
    {"id": "ref-03", "type": "refuse",
     "question": "How many fatal accidents happened in 2018?",
     "notes": "Year outside 2021-2024 coverage."},
    {"id": "ref-04", "type": "refuse",
     "question": "What's a good stock to buy in the mining sector?",
     "notes": "Off-topic."},
]


def build(db_path: Path = DB_PATH) -> list[dict]:
    con = sqlite3.connect(db_path)
    regs = dict(con.execute("SELECT section_id, text FROM regulations"))
    out = []
    for q in QUESTIONS:
        item = {k: v for k, v in q.items() if k != "sql"}
        item["must_refuse"] = q["type"] == "refuse"
        for sec in q.get("expected_sections", []) + q.get("forbidden_sections", []):
            assert sec in regs, f"{q['id']}: section {sec} not in DB"
        for fact in q.get("expected_facts", []):
            sec_text = " ".join(regs[s] for s in q["expected_sections"])
            assert fact in sec_text, f"{q['id']}: fact {fact!r} not in section text"
        if "sql" in q:
            value = con.execute(q["sql"]).fetchone()[0]
            assert value not in (None, 0), f"{q['id']}: SQL returned {value!r}"
            item["expected_value"] = value
            item["verified_by"] = q["sql"]
        elif q["type"] == "partial":
            item["verified_by"] = ("sections exist; out-of-scope part: " + q["notes"])
        elif q["type"] == "reg":
            item["verified_by"] = "facts found verbatim in " + ", ".join(q["expected_sections"])
        else:
            item["verified_by"] = "scope rule (CLAUDE.md)"
        out.append(item)
    con.close()
    return out


# %% [markdown]
# ## M5 · Rule-based scoring (no LLM judge)
#
# Deterministic scoring: facts by token overlap, numbers by exact value, citations only if grounded, plus the phrase rules. Answers are cached per model so re-scoring is free.
#
# *Source: `evals/eval_agent.py`*


# %%
MIN_CALL_GAP = 13  # seconds between LLM calls: free-tier gemini-2.5-flash allows 5 requests/minute
FACT_THRESHOLD = 0.6  # share of a fact's content words that must appear in the answer
FACT_STOPWORDS = {"a", "an", "the", "of", "to", "be", "is", "are", "at", "on", "in", "and", "or",
             "for", "by", "with", "shall", "must", "that", "this", "it"}


# --- scoring (pure; unit-tested) ------------------------------------------------------------------

def _words(text: str) -> set[str]:
    # Crude stemming: "pulleys"/"pulley", "locked"/"lock" should match.
    return {re.sub(r"(es|s|ed|ing)$", "", w) for w in re.findall(r"[a-z]+", text.lower())
            if w not in FACT_STOPWORDS}


def fact_present(fact: str, answer: str) -> bool:
    """Token-overlap match, so 'must be guarded' counts for 'shall be guarded'."""
    want = _words(fact)
    return not want or len(want & _words(answer)) / len(want) >= FACT_THRESHOLD


def value_present(expected, answer: str) -> bool:
    if isinstance(expected, (int, float)):
        nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", answer)}
        return float(expected) in nums
    return re.search(rf"\b{re.escape(str(expected))}\b", answer, re.IGNORECASE) is not None


def lead(answer: str) -> str:
    """The opening paragraph: where a caveat has to be to count as 'up front'."""
    return answer.strip().split("\n\n")[0]


def phrase_checks(item: dict, answer: str) -> list[str]:
    """Case-insensitive phrase rules from manual review; returns what's missing or forbidden."""
    low, head = answer.lower(), lead(answer).lower()
    miss = [f"phrase: {p!r}" for p in item.get("required_phrases", []) if p.lower() not in low]
    miss += [f"phrase up front: {p!r}" for p in item.get("lead_phrases", []) if p.lower() not in head]
    # "not (be considered) representative" is the honest caveat, so a negated phrase is fine.
    miss += [f"forbidden: {p!r}" for p in item.get("forbidden_phrases", [])
             if any(not re.search(r"\bnot\b", low[max(0, m.start() - 30):m.start()])
                    for m in re.finditer(rf"\b{re.escape(p.lower())}", low))]
    return miss


def score_item(item: dict, result: dict) -> dict:
    """Per-question scores. None means the metric doesn't apply to this question type."""
    answer, refused = result["answer"], result["refused"]
    expected_secs = set(item.get("expected_sections", []))
    cited = set(result["citations"]["sections"])
    s = {"refusal_ok": refused == item["must_refuse"],
         "citation_recall": len(expected_secs & cited) / len(expected_secs) if expected_secs else None,
         "grounded": not any(result["ungrounded"].values()),
         "missing": []}

    if item["must_refuse"]:
        s["correct"] = refused
        return s
    checks = []
    if "expected_facts" in item:
        for f in item["expected_facts"]:
            checks.append(ok := fact_present(f, answer))
            if not ok:
                s["missing"].append(f"fact: {f}")
    if "expected_value" in item:
        checks.append(ok := value_present(item["expected_value"], answer))
        if not ok:
            s["missing"].append(f"value: {item['expected_value']}")
    if expected_secs:  # hybrid questions carry sections but no facts: the section is the answer
        checks.append(bool(expected_secs & cited))
        if not expected_secs & cited:
            s["missing"].append(f"section: {sorted(expected_secs)}")
    phrases = phrase_checks(item, answer)
    # Expired versions of dated sections (56.5001 vs 56.5001T) must not be cited.
    phrases += [f"cited expired: {sec}" for sec in item.get("forbidden_sections", []) if sec in cited]
    s["missing"] += phrases
    s["correct"] = not refused and all(checks) and not phrases
    return s


def _mean(xs: list) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(items: list[dict], scores: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    for kind in ["reg", "agg", "hybrid", "partial", "refuse", "all"]:
        ids = [i["id"] for i in items if (kind == "all" or i["type"] == kind) and i["id"] in scores]
        if not ids:
            continue
        sc = [scores[i] for i in ids]
        rows[kind] = {"n": len(ids),
                      "answer_acc": _mean([s["correct"] for s in sc]),
                      "citation_recall": _mean([s["citation_recall"] for s in sc]),
                      "refusal_acc": _mean([s["refusal_ok"] for s in sc]),
                      "grounded": _mean([s["grounded"] for s in sc])}
    answerable = [scores[i["id"]] for i in items if not i["must_refuse"] and i["id"] in scores]
    rows["all"]["false_refusal"] = _mean([not s["refusal_ok"] for s in answerable])
    return rows


# --- run + report -------------------------------------------------------------------------------

class Throttled:
    """Wraps a Provider so consecutive calls are at least `gap` seconds apart."""

    def __init__(self, provider, gap: float):
        self.provider, self.gap, self.last = provider, gap, 0.0
        self.calls = 0
        self.tokens = {"input": 0, "output": 0}

    def chat(self, system, messages, tools):
        time.sleep(max(0.0, self.last + self.gap - time.time()))
        try:
            reply = self.provider.chat(system, messages, tools)
        finally:
            self.last, self.calls = time.time(), self.calls + 1
        for k in self.tokens:
            self.tokens[k] += reply.usage.get(k, 0)
        return reply


def run_items(items: list[dict], model: str, run_dir: Path, gap: float) -> None:

    provider = Throttled(GeminiProvider(model), gap)
    todo = [i for i in items if not (run_dir / f"{i['id']}.json").exists()]
    print(f"{len(items) - len(todo)} cached, {len(todo)} to run with {model}")
    for item in todo:
        t0, before = time.time(), dict(provider.tokens)
        result = run(item["question"], provider).to_dict()
        result["seconds"] = round(time.time() - t0, 1)
        result["tokens"] = {k: provider.tokens[k] - before[k] for k in before}
        (run_dir / f"{item['id']}.json").write_text(json.dumps(result, indent=1, default=str))
        print(f"  {item['id']}: {len(result['trace'])} tool calls, {result['seconds']}s "
              f"({provider.calls} LLM calls, tokens so far {provider.tokens})")


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def report(items: list[dict], scores: dict, results: dict, model: str) -> str:
    rows = summarize(items, scores)
    tool_calls = sum(len(r["trace"]) for r in results.values())
    tok = {k: sum(r.get("tokens", {}).get(k, 0) for r in results.values()) for k in ("input", "output")}
    lines = [f"# Agent eval — {model}", "",
             f"{rows['all']['n']} golden questions · run {date.today()} · {tool_calls} tool calls "
             f"· {tok['input']:,} input / {tok['output']:,} output tokens "
             f"· false-refusal rate {_fmt(rows['all']['false_refusal'])}", "",
             "| type | n | answer acc | citation recall | refusal acc | grounded |",
             "|---|---|---|---|---|---|"]
    for kind, r in rows.items():
        lines.append(f"| {kind} | {r['n']} | {_fmt(r['answer_acc'])} | {_fmt(r['citation_recall'])} "
                     f"| {_fmt(r['refusal_acc'])} | {_fmt(r['grounded'])} |")
    lines += ["", "**Metrics.** Answer acc: expected facts (token overlap ≥ "
              f"{FACT_THRESHOLD}), value and section all present. Citation recall: expected sections "
              "among *grounded* citations. Grounded: no ungrounded numbers/sections in the answer. "
              "Phrase rules (from manual review): sample stats must say “sample” in the opening "
              "paragraph and never “representative”; keyword counts must say “mention”; partial "
              "answers must include “Not covered:”.",
              "", f"**Caveat.** {len(items)} questions written alongside the system; one miss moves "
              "a type's score by 0.11–0.25. Rule-based scoring checks that facts are present, not that "
              "everything else in the answer is right. Treat this as a regression gate, not a benchmark.",
              "", "## Failures", ""]
    fails = []
    for item in items:
        s = scores.get(item["id"])
        if s is None or (s["correct"] and s["grounded"]):
            continue
        r = results[item["id"]]
        why = "; ".join(s["missing"]) or ("wrong refusal" if not s["refusal_ok"] else "")
        ung = {k: v for k, v in r["ungrounded"].items() if v}
        fails.append(f"- **{item['id']}** {item['question']}\n  - missing: {why or '–'}"
                     + (f"\n  - ungrounded: {ung}" if ung else "")
                     + f"\n  - answer: {r['answer'][:300].replace(chr(10), ' ')}")
    return "\n".join(lines + (fails or ["None."])) + "\n"


# %% [markdown]
# ## Run it
#
# API key and model.


# %%
import getpass
import sys

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")


# %% [markdown]
# Download the data and build the DB and search index (first run only).


# %%
if __name__ == "__main__":
    if not DB_PATH.exists():
        print(build_db(fetch_raw()))
    if not (INDEX_DIR / "embeddings.npy").exists():
        build_index()


# %% [markdown]
# `ask()` prints the answer, the citations that passed the grounding check, and the tool calls. `evaluate()` runs the golden set.


# %%
def ask(question: str, provider: Provider | None = None) -> AgentResult:
    """Ask the copilot and print the answer, verified citations and the tools it used."""
    result = run(question, provider or GeminiProvider(MODEL))
    print(result.answer, "\n")
    for t in result.trace:
        print(f"  [tool] {t['tool']}({json.dumps(t['args'])})")
    print(f"  verified citations: {result.citations['sections'] or '-'}")
    if any(result.ungrounded.values()):
        print(f"  UNGROUNDED (not in tool output): {result.ungrounded}")
    return result


def evaluate(model: str = MODEL, only: list[str] | None = None) -> str:
    """Run (or reuse cached) golden-set answers, score them, return the markdown report."""
    items = [i for i in build() if not only or i["id"] in only]
    run_dir = ROOT / "eval_runs" / model
    run_dir.mkdir(parents=True, exist_ok=True)
    run_items(items, model, run_dir, gap=1)
    results = {i["id"]: json.loads((run_dir / f"{i['id']}.json").read_text()) for i in items}
    for r in results.values():
        r["citations"], r["ungrounded"] = ground(r["answer"], r["question"], r["trace"])
    scores = {k: score_item(next(i for i in items if i["id"] == k), r) for k, r in results.items()}
    return report(items, scores, results, model)


# %% [markdown]
# One example of each kind: regulation, statistics, partial answer, refusal. Try your own with `ask("...")`.


# %%
EXAMPLES = [
    "How high must berms be on haul roads?",                                   # regulation
    "How many fatal accidents were recorded in 2023?",                         # statistics
    "Do new quarry workers need hard hats, and what training must they complete?",  # partial
    "What does Part 75 require for roof bolting in underground coal mines?",   # out of scope
]


# %%
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    for q in sys.argv[1:] or EXAMPLES:  # python mine_safety_copilot.py "your question"
        print("=" * 100, "\nQ:", q, "\n")
        ask(q)


# %% [markdown]
# ## Optional: the golden-set exam


# %%
RUN_EVAL = False  # the full golden set is ~160k input tokens; set True to run it
if __name__ == "__main__" and RUN_EVAL:
    print(evaluate())
