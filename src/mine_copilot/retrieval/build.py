"""Build the regulation search index from the SQLite `regulations` table.

Usage: python -m mine_copilot.retrieval.build [--no-embed]
"""

import sqlite3
import sys

from mine_copilot import config
from mine_copilot.retrieval.index import RegIndex


def load_sections() -> list[dict]:
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM regulations ORDER BY rowid")]


def main(embed: bool = True) -> RegIndex:
    embed_fn = None
    if embed:
        from mine_copilot.retrieval.embed import embed as gemini_embed

        def embed_fn(texts):
            return gemini_embed(texts, "RETRIEVAL_DOCUMENT")

    index = RegIndex.from_sections(load_sections(), embed_fn)
    index.save(config.INDEX_DIR)
    dims = "none" if index.embeddings is None else index.embeddings.shape
    print(f"indexed {len(index.chunks)} chunks / {len(index.section_ids)} sections, emb={dims}")
    return index


if __name__ == "__main__":
    main(embed="--no-embed" not in sys.argv)
