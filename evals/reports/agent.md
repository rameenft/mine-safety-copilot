# Agent eval — gemini-3.8-flash

24 golden questions · run 2026-10-06 · 46 tool calls · 90,306 input / 11,598 output tokens · false-refusal rate 0.00

| type | n | answer acc | citation recall | refusal acc | grounded |
|---|---|---|---|---|---|
| reg | 9 | 1.00 | 1.00 | 1.00 | 1.00 |
| agg | 7 | 1.00 | – | 1.00 | 1.00 |
| hybrid | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| refuse | 4 | 1.00 | – | 1.00 | 1.00 |
| all | 24 | 1.00 | 1.00 | 1.00 | 1.00 |

**Metrics.** Answer acc: expected facts (token overlap ≥ 0.6), value and section all present. Citation recall: expected sections among *grounded* citations. Grounded: no ungrounded numbers/sections in the answer.

**Caveat.** 24 questions written alongside the system; one miss moves a type's score by 0.11–0.25. Rule-based scoring checks that facts are present, not that everything else in the answer is right. Treat this as a regression gate, not a benchmark.

## Failures

None.
