"""The agent's three tools. Each returns a JSON-able dict; errors come back as {"error": ...}
so the model can correct itself instead of the loop crashing.

accident_stats never takes SQL from the model: filters and group-by fields are allow-listed
and every value is bound as a parameter.
"""

import os
import re
import sqlite3
from datetime import date, datetime
from functools import cached_property
from pathlib import Path

from dotenv import load_dotenv

from mine_copilot.config import DB_PATH, INDEX_DIR

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
            from mine_copilot.retrieval.index import RegIndex

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
