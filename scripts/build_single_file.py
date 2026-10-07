"""Assemble the whole copilot into ONE self-contained file: mine_safety_copilot.py + .ipynb.

The package stays the source of truth. This script inlines its modules in dependency order, hoists
third-party imports, drops intra-package imports and CLI blocks, and renames the few top-level
names that two modules both define (so one doesn't silently overwrite the other).
tests/test_single_file.py fails if the committed files drift from this output.

Usage: python scripts/build_single_file.py
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mine_copilot"
OUT_PY = ROOT / "mine_safety_copilot.py"
OUT_NB = ROOT / "mine_safety_copilot.ipynb"
DEPS = "google-genai python-dotenv pandas pyyaml rank-bm25 numpy"

# (path, section title, plain-English intro, {old name: new name} for top-level clashes)
MODULES = [
    (SRC / "config.py", "Configuration",
     "Paths and pinned settings. Everything is written under `COPILOT_HOME` "
     "(default `./mine_copilot_workspace`).", {}),
    (SRC / "ingest" / "accidents.py", "M1 · Accident data: filter, severity, stratified sample",
     "Keeps surface metal/nonmetal accidents from 2021–2024 that have a narrative, maps MSHA's 11 "
     "injury codes to 6 `SEVERITY` values, then keeps **every fatality** plus a seeded random "
     "sample (stratified sampling) up to 3,000 rows.", {"YEARS": "ACCIDENT_YEARS"}),
    (SRC / "ingest" / "regulations.py", "M1 · Regulations: parse eCFR XML, structure-aware chunking",
     "Parses 30 CFR Part 56 into one record per section. Long sections are split on paragraph "
     "boundaries (≤1,500 chars) and every chunk keeps its section ID, so a search hit can always "
     "be cited and the full section fetched.", {}),
    (SRC / "ingest" / "build.py", "M1 · Download and build the SQLite database",
     "Downloads the MSHA files and the pinned eCFR snapshot (skipped if already present) and "
     "loads three tables: `accidents`, `mines`, `regulations`.", {}),
    (SRC / "retrieval" / "index.py", "M3 · Retrieval index: BM25, dense, RRF hybrid",
     "Three search modes over the chunks. Dense (embeddings + cosine similarity on a NumPy "
     "matrix, no vector DB) is the default because it measured best on the golden set; section "
     "IDs named in a query are pinned to the top.", {}),
    (SRC / "retrieval" / "embed.py", "M3 · Gemini embeddings",
     "Batched, L2-normalised embeddings with a retry when the per-minute quota is hit.",
     {"RETRY_WAIT": "EMBED_RETRY_WAIT"}),
    (SRC / "retrieval" / "build.py", "M3 · Build the index",
     "Chunks every section, embeds the chunks once and saves `chunks.json` + `embeddings.npy`.",
     {"main": "build_index", "embed": "use_embeddings"}),
    (SRC / "llm.py", "M4 · LLM provider interface",
     "Every model call goes through `Provider.chat`, so swapping Gemini for another model is a "
     "config change. `FakeProvider` replays scripted replies for offline tests.", {}),
    (SRC / "agent" / "tools.py", "M4 · The agent's three tools (and their guardrails)",
     "`search_regulations`, `get_regulation` and `accident_stats`. Stats use allow-listed "
     "filters and bound parameters on a read-only DB (no text-to-SQL). Expired dated sections "
     "(56.5001 vs 56.5001T) are hidden and redirected here, in code, because the model ignored "
     "prompt rules and labels about them.", {}),
    (SRC / "agent" / "loop.py", "M4 · Agent loop and grounding check",
     "The system prompt (scope, sample caveats, partial answers with `Not covered:`), a loop of "
     "at most 5 tool turns, and a post-answer check that every cited section and number appears "
     "in the tool output.", {"SECTION_RE": "CITATION_RE"}),
    (ROOT / "evals" / "build_golden.py", "M2 · Golden set (the exam)",
     "32 questions. Expected facts are checked verbatim against section text; expected numbers "
     "come from SQL at build time; phrase rules come from the manual review.",
     {"SAMPLE_NOTE": "GOLDEN_SAMPLE_NOTE"}),
    (ROOT / "evals" / "eval_agent.py", "M5 · Rule-based scoring (no LLM judge)",
     "Deterministic scoring: facts by token overlap, numbers by exact value, citations only if "
     "grounded, plus the phrase rules. Answers are cached per model so re-scoring is free.",
     {"STOPWORDS": "FACT_STOPWORDS"}),
]
DROP_NAMES = {"FIXTURES_DIR", "OUT", "HERE", "DEFAULT_MODEL", "main"}  # CLI/test-only top-levels
ROOT_LINE = 'ROOT = Path(os.environ.get("COPILOT_HOME", "mine_copilot_workspace")).resolve()'

INTRO = """# Mine Safety Copilot — the whole project in one file

Grounded safety Q&A for US **surface metal/nonmetal** mines, answering only from **30 CFR Part 56**
and **MSHA accident records (2021–2024)**. Every cited section and number is checked against tool
output; out-of-scope questions are refused, half-in-scope ones get a partial answer.

**To run:** set `GEMINI_API_KEY` (you'll be prompted otherwise) and run every cell. The first run
downloads ~60 MB of public government data and embeds 459 regulation chunks (a few minutes, a few
cents of API usage). Later runs reuse everything.

This file is generated from the repo's `src/` package by `scripts/build_single_file.py`."""

SETUP = '''import getpass
import sys

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")'''

BUILD = '''if __name__ == "__main__":
    if not DB_PATH.exists():
        print(build_db(fetch_raw()))
    if not (INDEX_DIR / "embeddings.npy").exists():
        build_index()'''

ASK = '''def ask(question: str, provider: Provider | None = None) -> AgentResult:
    """Ask the copilot and print the answer, verified citations and the tools it used."""
    result = run(question, provider or GeminiProvider(MODEL))
    print(result.answer, "\\n")
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
    return report(items, scores, results, model)'''

EXAMPLES = '''EXAMPLES = [
    "How high must berms be on haul roads?",                                   # regulation
    "How many fatal accidents were recorded in 2023?",                         # statistics
    "Do new quarry workers need hard hats, and what training must they complete?",  # partial
    "What does Part 75 require for roof bolting in underground coal mines?",   # out of scope
]'''

RUN_EXAMPLES = '''if __name__ == "__main__" and "ipykernel" in sys.modules:
    for q in EXAMPLES:
        print("=" * 100, "\\nQ:", q, "\\n")
        ask(q)'''

RUN_EVAL = '''RUN_EVAL = False  # the full golden set is ~160k input tokens; set True to run it
if __name__ == "__main__" and RUN_EVAL:
    print(evaluate())'''

CLI = '''if __name__ == "__main__" and "ipykernel" not in sys.modules:
    for q in sys.argv[1:] or EXAMPLES:  # python mine_safety_copilot.py "your question"
        print("=" * 100, "\\nQ:", q, "\\n")
        ask(q)'''


def _rename(code: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        code = re.sub(rf"(?<![\w.]){old}\b", new, code)
    return code


def transform(path: Path, renames: dict[str, str]) -> tuple[str, list[str]]:
    """Module body without imports/CLI blocks, plus its top-level third-party imports."""
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    drop, imports, repl = set(), [], {}
    for node in tree.body:
        span = range(node.lineno - 1, node.end_lineno)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node is tree.body[0]:
            drop.update(span)  # module docstring -> becomes the markdown intro
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(span)
            mod = getattr(node, "module", None) or ""
            if not mod.startswith("mine_copilot"):
                imports.append(ast.unparse(node))
        elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            drop.update(span)
        elif isinstance(node, (ast.FunctionDef, ast.Assign)) and (
                ({getattr(node, "name", None)}
                 | {getattr(t, "id", None) for t in getattr(node, "targets", [])})
                & (DROP_NAMES - set(renames))):
            drop.update(span)
        elif isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "ROOT":
            repl[node.lineno - 1] = ROOT_LINE
            drop.update(range(node.lineno, node.end_lineno))
    for node in ast.walk(tree):  # lazy in-function imports of our own modules
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mine_copilot") \
                and node.col_offset > 0:
            pad = " " * node.col_offset
            aliased = [f"{pad}{a.asname} = {a.name}" for a in node.names if a.asname]
            repl[node.lineno - 1] = "\n".join(aliased) if aliased else None
            drop.update(range(node.lineno, node.end_lineno))
    out = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        if i in repl:
            if repl[i] is not None:
                out.append(repl[i])
        else:
            out.append(line)
    # "embed as gemini_embed" must keep pointing at the function after `embed` the flag is renamed.
    code = _rename("\n".join(out), renames).replace("gemini_embed = use_embeddings",
                                                    "gemini_embed = embed")
    code = re.sub(r"\bconfig\.", "", code)
    return re.sub(r"\n{3,}", "\n\n\n", code).strip("\n"), imports


def _needed(imp: str, code: str) -> bool:
    """Drop imports only the removed CLI blocks used (e.g. argparse)."""
    names = re.sub(r"^(from \S+ )?import ", "", imp).split(",")
    bound = [n.split(" as ")[-1].strip().split(".")[0] for n in names]
    return any(re.search(rf"\b{re.escape(b)}\b", code) for b in bound)


def merge_imports(imports: list[str]) -> list[str]:
    plain, froms = set(), {}
    for imp in imports:
        if m := re.match(r"from (\S+) import (.+)", imp):
            froms.setdefault(m.group(1), set()).update(n.strip() for n in m.group(2).split(","))
        else:
            plain.add(imp)
    plain.add("import os")
    out = sorted(plain) + [f"from {mod} import {', '.join(sorted(names))}"
                           for mod, names in sorted(froms.items())]
    return sorted(out, key=lambda s: (s.split()[1].split(".")[0] not in STDLIB, s.split()[1]))


STDLIB = {"argparse", "ast", "asyncio", "dataclasses", "datetime", "functools", "gzip", "json",
          "os", "pathlib", "re", "sqlite3", "sys", "time", "typing", "urllib", "xml", "zipfile"}


def cells() -> list[tuple[str, str, str]]:
    """(kind, source, target) where target is 'both', 'py' or 'nb'."""
    bodies, imports = [], []
    for path, title, intro, renames in MODULES:
        code, imps = transform(path, renames)
        bodies.append((title, intro, path.relative_to(ROOT), code))
        imports += imps
    used = "\n".join(b[3] for b in bodies) + SETUP + BUILD + ASK
    out = [("markdown", INTRO, "both"),
           ("code", f"%pip install -q {DEPS}", "nb"),
           ("markdown", "## Imports", "both"),
           ("code", "\n".join(i for i in merge_imports(imports) if _needed(i, used)), "both")]
    for title, intro, rel, code in bodies:
        out += [("markdown", f"## {title}\n\n{intro}\n\n*Source: `{rel}`*", "both"),
                ("code", code, "both")]
    out += [("markdown", "## Run it\n\nAPI key and model.", "both"), ("code", SETUP, "both"),
            ("markdown", "Download the data and build the DB and search index "
                         "(first run only).", "both"), ("code", BUILD, "both"),
            ("markdown", "`ask()` prints the answer, the citations that passed the grounding "
                         "check, and the tool calls. `evaluate()` runs the golden set.", "both"),
            ("code", ASK, "both"),
            ("markdown", "One example of each kind: regulation, statistics, partial answer, "
                         "refusal. Try your own with `ask(\"...\")`.", "both"),
            ("code", EXAMPLES, "both"), ("code", RUN_EXAMPLES, "nb"), ("code", CLI, "py"),
            ("markdown", "## Optional: the golden-set exam", "both"), ("code", RUN_EVAL, "both")]
    return out


def to_py(cs) -> str:
    parts = ['"""Mine Safety Copilot: the whole project in one runnable file.\n\n'
             'Usage: python mine_safety_copilot.py ["your question" ...]\n'
             'Generated from src/ by scripts/build_single_file.py; do not edit by hand.\n"""']
    for kind, src, target in cs:
        if target == "nb":
            continue
        if kind == "markdown":
            parts.append("# %% [markdown]\n" + "\n".join(f"# {ln}".rstrip() for ln in src.splitlines()))
        else:
            parts.append("# %%\n" + src)
    return "\n\n\n".join(parts) + "\n"


def to_notebook(cs) -> dict:
    nb_cells = []
    for kind, src, target in cs:
        if target == "py":
            continue
        cell = {"cell_type": kind, "id": f"cell-{len(nb_cells):02d}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    cs = cells()
    OUT_PY.write_text(to_py(cs))
    OUT_NB.write_text(json.dumps(to_notebook(cs), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PY.name} ({len(OUT_PY.read_text().splitlines())} lines) and {OUT_NB.name}")
