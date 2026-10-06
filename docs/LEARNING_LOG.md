# Learning Log

## Milestone 0 — Setup
- A **virtual environment** (`.venv`) isolates this project's packages from system Python.
- **`.env` + `.gitignore`**: secrets live in a local file that git never tracks; `.env.example` documents the keys.
- **`pyproject.toml`** is the single modern config for package metadata, deps, ruff and pytest.
- **`src/` layout** forces imports to go through the installed package, catching packaging bugs early.
- **CLAUDE.md** is the hand-off file Claude Code reads at the start of every session.
