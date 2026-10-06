"""Gemini embeddings, L2-normalised so a dot product is cosine similarity."""

import os
import time
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv

from mine_copilot.config import EMBED_DIM, EMBED_MODEL

BATCH = 100  # max texts per request; free tier also caps ~100 texts/minute
RETRY_WAIT = 65  # seconds to wait out a per-minute quota window


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
                print(f"rate-limited at {i}/{len(texts)}, waiting {RETRY_WAIT}s")
                time.sleep(RETRY_WAIT)
        vecs.extend(e.values for e in resp.embeddings)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


@lru_cache(maxsize=512)
def embed_query(query: str) -> np.ndarray:
    return embed([query], "RETRIEVAL_QUERY")[0]
