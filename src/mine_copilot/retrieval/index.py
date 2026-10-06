"""Dense, BM25 or hybrid (RRF-fused) search over Part 56 chunks, fused with Reciprocal Rank Fusion.

Results are merged to one hit per section so citations point at sections, not chunk fragments.
"""

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from mine_copilot.ingest.regulations import chunk_section

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
            from mine_copilot.retrieval.embed import embed_query

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
