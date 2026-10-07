# Mine Safety Copilot

Grounded safety Q&A for US **surface metal/nonmetal** mines. It answers only from
**30 CFR Part 56** (the federal safety standards) and **MSHA accident records (2021–2024)**.
It cites the exact section or record behind every claim, checks those citations in code, and
says plainly when a question is fully or partly outside what it knows.

**Why:** in safety, a confident wrong answer is worse than no answer. A general chatbot will invent
a plausible regulation number. This project optimizes for *verifiable* answers: the model can
only look things up through narrow tools, and code checks its citations afterwards. Its
behaviour was then tested by hand against outside sources, not just by its own exam.

## How it works

```mermaid
flowchart LR
    subgraph Ingest["Ingest (offline)"]
        A[MSHA Accidents.zip<br/>+ Mines.zip] --> D[(SQLite<br/>accidents · mines · regulations)]
        B[eCFR Part 56 XML<br/>pinned 2026-10-01] --> D
    end
    subgraph Index["Retrieval index"]
        D --> C[459 chunks]
        C --> E[BM25]
        C --> F[Gemini embeddings<br/>.npy matrix]
    end
    subgraph Agent["Agent loop (≤5 tool turns)"]
        Q[Question] --> L[LLM<br/>gemini-3.8-flash]
        L -- search_regulations --> E & F
        L -- get_regulation --> D
        L -- accident_stats<br/>allow-listed, read-only --> D
        L --> G{Grounding check}
    end
    G --> O[Answer + verified citations<br/>partial answer, or refusal]
    O --> UI[Streamlit demo / CLI]
```

1. **Data (ETL).** It downloads 275k MSHA accident records and the eCFR Part 56 XML, then filters
   to surface metal/nonmetal mines from 2021 to 2024 and loads the result into SQLite: 3,000
   accidents, 1,410 mines and 422 regulation sections. **Stratified sampling** keeps all 80
   fatalities, the rare and high-value rows, and fills the rest with a seeded random sample,
   so results are reproducible. MSHA's 11 cryptic injury codes become a 6-value `SEVERITY`
   column the tools can filter on.
2. **Retrieval (RAG).** **Structure-aware chunking:** sections stay whole when they're short, and
   long ones are split on paragraph boundaries into chunks of at most 1,500 chars (20 sections
   split, 459 chunks). Every chunk keeps its section ID, so any hit can be cited and the full
   section fetched. Search is dense (Gemini embeddings, cosine similarity on a `.npy` matrix),
   BM25 or an RRF hybrid. **Dense is the default because it measured best**, not by assumption.
3. **Agent (tool calling).** Three tools: `search_regulations`, `get_regulation` and
   `accident_stats`. The stats tool takes typed, **allow-listed** filters with bound parameters on a
   read-only DB, so there's **no text-to-SQL**. Out-of-scope questions get a refusal. Half-in-scope
   questions get a **partial answer** that ends with `Not covered:` and a pointer (e.g. Part 46 for
   training, Part 60 for silica).
4. **Grounding check.** After the model answers, code matches every cited section, document
   number and figure against the tool output. Anything unsupported is stripped and flagged.
5. **Safety-critical facts live in code.** Part 56 contains expired and in-force versions of the
   dust rules (56.5001 vs 56.5001T). The model kept citing the expired ones, even when they were
   labelled EXPIRED, because it doesn't know today's date. The tool layer now hides expired
   sections and redirects lookups to the version in force.

## Evaluation

**Golden set:** 32 questions of six types: regulation, stats, hybrid, partial, refuse, and dated
rules. Expected facts are verified verbatim against the section text and expected numbers are
computed by SQL. Scoring is **rule-based (no LLM judge)**, so it's deterministic and free to re-run.

| Agent, gemini-3.8-flash ([report](evals/reports/agent.md)) | n | answer acc | citation recall | grounded |
|---|---|---|---|---|
| regulation · stats · partial · refuse | 26 | 1.00 | 1.00 | 1.00 |
| hybrid (rule + stats) | 4 | 0.75 | 1.00 | 1.00 |
| **all (30-question run)** | **30** | **0.97** | **1.00** | **1.00** |

False-refusal rate 0.00. The two dated-rule questions were added later and run separately:
reg-10 passes, and part-07 refuses with a correct Part 60 pointer instead of giving a partial
answer. Both remaining misses are left visible rather than prompt-tuned away.

| Retrieval, 13 Qs, k=5 ([report](evals/reports/retrieval.md)) | recall@1 | recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.54 | 0.85 | 0.65 |
| **dense (default)** | **0.92** | **0.92** | **0.92** |
| hybrid (RRF) | 0.85 | 0.85 | 0.85 |

> Caveat: questions written alongside the system make a regression gate, not a benchmark.

## Manual review

The automated eval first scored 24/24. A hand review against outside sources found what it missed:

| Check | Finding | Outcome |
|---|---|---|
| Read every answer | Sample stats lacked a caveat; one called the sample "representative" | New phrase rules: old answers drop to **19/24**, fixed prompt scores 29/30 |
| Keyword-count accidents | Of 12 fatal matches, only 7 had the keyword as the cause | Answers must say "*mentioning* conveyors" |
| Fatal counts vs MSHA | Raw counts reconcile **exactly** with MSHA's published totals (95 = 80 kept + 15 underground) | Filters verified |
| Severity mapping | Sound; `other` mixes illness, natural causes, non-employees | Documented; tool describes it |
| Dated silica rules | Model cited expired rules | Fixed in the tool layer |
| Rule text vs eCFR | 5 sections incl. the longest: word-for-word identical; chunks rebuild losslessly | Ingest verified |

Details: [`docs/DATA_CARD.md`](docs/DATA_CARD.md) · rationale: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Cost and testing

- **Built on the free tier.** All model calls go through one provider interface, so swapping
  models (e.g. to Claude) is a config change. BM25 runs with no API key.
- **Pay once, reuse.** Embeddings are computed once into a 1.4 MB `.npy` file (no vector DB). Eval
  answers are cached per model, so `--rescore` and grounding-check fixes re-score offline, and
  `--only` re-runs just the changed questions. A full eval run is about 156k input and 20k output
  tokens, all tracked.
- **48 offline tests.** Fixtures are real data with deliberate *must-reject* rows (coal, 2019,
  underground, blank narratives) so every filter is proven, and a scripted `FakeProvider`
  drives the agent and the Streamlit UI with no network.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
cp .env.example .env                          # add GEMINI_API_KEY
python -m mine_copilot.ingest.build           # download + build the SQLite DB
python -m mine_copilot.retrieval.build        # chunk + embed (--no-embed for BM25 only)
python -m mine_copilot.agent "How high must berms be on haul roads?"   # --json for the trace
streamlit run src/mine_copilot/app.py         # demo UI
pytest -q                                     # offline tests
python evals/eval_agent.py --model gemini-3.8-flash   # live eval (cached; --rescore offline)
```

## Limitations

- Part 56 only: no underground (Part 57), coal (Parts 70–75), training (Part 46) or silica limits (Part 60).
- Accident stats describe a 3,000-row sample that keeps every fatality, so it over-represents
  severe accidents. Narratives are short (≤384 chars), and half the rows have no equipment field.
- Expired rule text can't be quoted, even for accidents that happened under it.
- The grounding check verifies sections and numbers, not paraphrased wording.
- Small in-house eval with one model; next steps are a held-out question set written by someone
  else and a comparison across models.
- Not legal or compliance advice. Always check the current eCFR.

## Repo layout

```
src/mine_copilot/   ingest/ retrieval/ agent/ llm.py config.py demo.py app.py
evals/              golden_set.yaml, build_golden.py, eval_*.py, reports/
tests/              offline tests + fixtures/
docs/               DATA_CARD.md, DECISIONS.md
```
