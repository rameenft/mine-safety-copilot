"""Retrieval eval: recall@1, recall@5 and MRR on golden reg/hybrid questions, per search mode.

Usage: python evals/eval_retrieval.py   (writes evals/reports/retrieval.md)
"""

from pathlib import Path

import yaml

from mine_copilot.config import INDEX_DIR
from mine_copilot.retrieval.index import RegIndex

HERE = Path(__file__).resolve().parent
K = 5


def score(index: RegIndex, items: list[dict], mode: str, qvecs: dict) -> dict:
    r1 = r5 = rr = 0.0
    misses = []
    for item in items:
        expected = set(item["expected_sections"])
        ranked = [h["section_id"] for h in index.search(
            item["question"], k=K, mode=mode, query_vec=qvecs.get(item["id"]))]
        r1 += bool(expected & set(ranked[:1]))
        r5 += len(expected & set(ranked)) / len(expected)
        first = next((i for i, s in enumerate(ranked) if s in expected), None)
        rr += 0 if first is None else 1 / (first + 1)
        if first is None:
            misses.append(f"{item['id']}: want {sorted(expected)}, got {ranked}")
    n = len(items)
    return {"recall@1": r1 / n, f"recall@{K}": r5 / n, "MRR": rr / n, "misses": misses}


def main() -> None:
    items = [i for i in yaml.safe_load((HERE / "golden_set.yaml").read_text())
             if i["type"] in {"reg", "hybrid"}]
    index = RegIndex.load(INDEX_DIR)
    modes = ["bm25"]
    qvecs = {}
    if index.embeddings is not None:
        from mine_copilot.retrieval.embed import embed

        vecs = embed([i["question"] for i in items], "RETRIEVAL_QUERY")  # one batched call
        qvecs = {i["id"]: v for i, v in zip(items, vecs)}
        modes += ["dense", "hybrid"]

    lines = [f"# Retrieval eval ({len(items)} reg/hybrid questions, k={K})", "",
             f"| mode | recall@1 | recall@{K} | MRR |", "|---|---|---|---|"]
    misses = []
    for mode in modes:
        s = score(index, items, mode, qvecs)
        lines.append(f"| {mode} | {s['recall@1']:.2f} | {s[f'recall@{K}']:.2f} | {s['MRR']:.2f} |")
        misses += [f"- **{mode}** {m}" for m in s["misses"]]
    lines += ["", "## Misses (no expected section in top k)", ""] + (misses or ["None."])
    report = "\n".join(lines) + "\n"
    (HERE / "reports" / "retrieval.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
