"""CLI: python -m mine_copilot.agent "question" [--json]"""

import argparse
import json

from mine_copilot.agent.loop import run
from mine_copilot.llm import GeminiProvider


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = p.parse_args()

    result = run(args.question, GeminiProvider())
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return
    print(result.answer, "\n")
    for t in result.trace:
        print(f"  [tool] {t['tool']}({json.dumps(t['args'])})")
    print(f"  citations: {result.citations}")
    if any(result.ungrounded.values()):
        print(f"  UNGROUNDED: {result.ungrounded}")


if __name__ == "__main__":
    main()
