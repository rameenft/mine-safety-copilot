# Agent eval — gemini-3.8-flash

30 golden questions · run 2026-10-07 · 66 tool calls · 156,183 input / 19,564 output tokens · false-refusal rate 0.00

| type | n | answer acc | citation recall | refusal acc | grounded |
|---|---|---|---|---|---|
| reg | 9 | 1.00 | 1.00 | 1.00 | 1.00 |
| agg | 7 | 1.00 | – | 1.00 | 1.00 |
| hybrid | 4 | 0.75 | 1.00 | 1.00 | 1.00 |
| partial | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| refuse | 4 | 1.00 | – | 1.00 | 1.00 |
| all | 30 | 0.97 | 1.00 | 1.00 | 1.00 |

**Metrics.** Answer acc: expected facts (token overlap ≥ 0.6), value and section all present. Citation recall: expected sections among *grounded* citations. Grounded: no ungrounded numbers/sections in the answer. Phrase rules (from manual review): sample stats must say “sample” in the opening paragraph and never “representative”; keyword counts must say “mention”; partial answers must include “Not covered:”.

**Caveat.** 30 questions written alongside the system; one miss moves a type's score by 0.11–0.25. Rule-based scoring checks that facts are present, not that everything else in the answer is right. Treat this as a regression gate, not a benchmark.

## Failures

- **hyb-01** How many fatal accidents involved conveyors, and what rule covers guarding them?
  - missing: phrase: 'mention'
  - answer: Between 2021 and 2024, there were **6 fatal accidents** involving conveyors in MSHA surface metal/nonmetal mining records (e.g., Document Nos. 220241150031, 220231730009, and 220231310014).  ### Applicable Guarding Regulations  * **§ 56.14107 (Moving machine parts):** Requires moving machine parts—s
