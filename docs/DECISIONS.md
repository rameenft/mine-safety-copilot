# Decisions

## Gemini free tier behind a provider interface
Free tier keeps the build at $0. All model calls go through `llm.py`, so moving to Claude
(Anthropic API or Bedrock) is a config change, not a rewrite.

## Lean, incremental dependencies
`pyproject.toml` only lists what the current milestone needs. Keeps installs fast and makes
each dependency's purpose obvious in the git history.

## Model: gemini-3.8-flash
Newest stable (non-preview) Flash model on the free tier as of 2026-10-06; verified with a test call.
Pinned in `.env` via `GEMINI_MODEL`; fallback is `gemini-2.5-flash` if free-tier quota runs out.

## Surface scope = 3 subunits
Kept STRIP/QUARRY/OPEN PIT, MILL/PREP PLANT, DREDGE. Dropped "SURFACE AT UNDERGROUND" because Part 57,
not Part 56, governs it — mixing them would let the bot cite the wrong part.

## Derived SEVERITY column
`DEGREE_INJURY` has 11 cryptic codes ("DYS AWY FRM WRK & RESTRCTD ACT"). A 6-value `SEVERITY`
column gives the tools a small, validatable filter vocabulary.

## Regulations in SQLite too
Sections are stored in a `regulations` table so `get_regulation(section_id)` is an exact lookup,
independent of the vector index. Suffixed IDs (56.5001T) are kept as distinct sections.

## Fixtures include must-reject rows
The accident fixture mixes 100 valid rows with coal/2019/underground/blank-narrative rows, so the
loader test proves each filter works instead of only checking the happy path.

## Retrieval: dense default, BM25 fallback, hybrid measured
Golden-set eval (13 reg/hybrid Qs): BM25 R@5 0.85, dense (gemini-embedding-001, 768-d) 0.92,
equal-weight RRF hybrid only R@1 0.62, because BM25 noise dragged good dense hits down. Dense-weighted (2×)
RRF reaches 0.85 but still trails dense, so dense is the default. BM25 runs with no API key
(tests/CI) and section IDs named in a query ("56.14107") are pinned via exact match. n=13 is small:
re-check once the M4 agent sends decomposed sub-queries instead of mixed hybrid questions.

## Embeddings stored as a .npy matrix, not a vector DB
459 chunks × 768 floats ≈ 1.4 MB; brute-force dot product is instant. A vector DB would add a
dependency with no benefit at this scale. Free tier caps ~100 texts/min, so the build retries on 429.
