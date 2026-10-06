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
