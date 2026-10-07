# Agent eval — gemini-3.8-flash

32 golden questions · run 2026-10-07 · 71 tool calls · 169,584 input / 24,280 output tokens · false-refusal rate 0.04

| type | n | answer acc | citation recall | refusal acc | grounded |
|---|---|---|---|---|---|
| reg | 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| agg | 7 | 1.00 | – | 1.00 | 1.00 |
| hybrid | 4 | 0.75 | 1.00 | 1.00 | 1.00 |
| partial | 7 | 0.86 | 0.80 | 0.86 | 1.00 |
| refuse | 4 | 1.00 | – | 1.00 | 1.00 |
| all | 32 | 0.94 | 0.95 | 0.97 | 1.00 |

**Metrics.** Answer acc: expected facts (token overlap ≥ 0.6), value and section all present. Citation recall: expected sections among *grounded* citations. Grounded: no ungrounded numbers/sections in the answer. Phrase rules (from manual review): sample stats must say “sample” in the opening paragraph and never “representative”; keyword counts must say “mention”; partial answers must include “Not covered:”.

**Caveat.** 32 questions written alongside the system; one miss moves a type's score by 0.11–0.25. Rule-based scoring checks that facts are present, not that everything else in the answer is right. Treat this as a regression gate, not a benchmark.

## Failures

- **hyb-01** How many fatal accidents involved conveyors, and what rule covers guarding them?
  - missing: phrase: 'mention'
  - answer: Between 2021 and 2024, there were **6 fatal accidents** involving conveyors in MSHA surface metal/nonmetal mining records (e.g., Document Nos. 220241150031, 220231730009, and 220231310014).  ### Applicable Guarding Regulations  * **§ 56.14107 (Moving machine parts):** Requires moving machine parts—s
- **part-07** What is the exposure limit for respirable crystalline silica at my quarry?
  - missing: section: ['56.5001T']; phrase: 'Not covered:'
  - answer: Out of scope: I only cover safety standards under 30 CFR Part 56 and MSHA accident data (2021–2024) for US surface metal and nonmetal mines. Respirable crystalline silica limits are governed by 30 CFR Part 60.
