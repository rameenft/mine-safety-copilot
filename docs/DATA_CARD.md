# Data Card

All sources are US federal government works (public domain). Raw and processed data are
git-ignored. Rebuild with `python -m mine_copilot.ingest.build`.

## Sources
| Source | URL | Snapshot |
|---|---|---|
| MSHA Accidents | `arlweb.msha.gov/OpenGovernmentData/DataSets/Accidents.zip` (+ definition file) | downloaded 2026-10-06 |
| MSHA Mines | `.../DataSets/Mines.zip` | downloaded 2026-10-06 |
| 30 CFR Part 56 | eCFR versioner API, `full/2026-10-01/title-30.xml?part=56` | pinned 2026-10-01 |

Format: pipe-delimited, ASCII (read as latin-1), CRLF line endings. The eCFR API requires `Accept-Encoding: gzip`.

## Accidents: filters and row counts
| Step | Rows |
|---|---|
| Raw file | 275,094 |
| `COAL_METAL_IND = 'M'` and `CAL_YR` 2021–2024 | 15,169 |
| Surface subunits only (see below), non-empty `NARRATIVE` | 12,872 (80 fatal) |
| **Final: all fatalities + seeded sample (seed 42), cap 3,000** | **3,000** |

- **Surface subunits kept:** `STRIP, QUARY, OPEN PIT` (1,641), `MILL OPERATION/PREPARATION PLANT` (1,274), `DREDGE` (85).
- **Excluded:** `UNDERGROUND`, plus `SURFACE AT UNDERGROUND`, because Part 57 covers it. Office, independent-shop and "other" subunits are also excluded.
- **By year:** 2021: 724 · 2022: 728 · 2023: 780 · 2024: 768.
- **Derived `SEVERITY`** (from `DEGREE_INJURY`): fatal 80, lost_time 987, restricted_duty 857, no_lost_time 798, permanent_disability 23, other 255.
- **Normalization:** `ACCIDENT_DT` becomes ISO `YYYY-MM-DD`, `DAYS_LOST` becomes an integer, and `'?'`, `'No Value Found'` (any case) and blanks become NULL.

## Missing values (final 3,000 rows)
`MINING_EQUIP` 50.5% · `DAYS_LOST` 6.9% · `OCCUPATION` / `ACTIVITY` / `INJURY_SOURCE` / `NATURE_INJURY` 2.3% · all other kept columns 0%.

## Mines
1,410 rows: one per `MINE_ID` that appears in the sampled accidents (0 orphans).
Columns: `MINE_ID, CURRENT_MINE_NAME, STATE, CURRENT_MINE_TYPE, PRIMARY_SIC`.

## Regulations
- **Size:** 422 sections across 20 subparts, stored as one row per section (`section_id, subpart, heading, text, source_url`).
- **Suffixed versions:** §56.5001T and §56.5005T (silica rule) are newer versions printed next to the originals. Both are kept as separate IDs.
- **Not stored:** editorial `XREF` amendment notes and source citations.
- **Section length:** 20 sections are over 1,500 characters (longest 15,559). These get split by paragraph at indexing time, and every chunk keeps its `section_id` and heading.

## Fixtures (`tests/fixtures/`, committed)
- `accidents_sample.txt`
  - 100 real rows that pass the filters, 10 of them fatal.
  - 15 real rows that must be rejected: 5 coal, 5 from 2019 and 5 underground.
  - 2 synthetic rows: real rows with the narrative blanked (`DOCUMENT_NO` ends in `X`).
- `mines_sample.txt`: mines for those rows.
- `part56_sample.xml`: the real eCFR XML trimmed to 20 sections.
- Regenerate with `python -m mine_copilot.ingest.fixtures`.

## Manual review: regulation text vs eCFR (2026-10-07)
Five sections (56.2, the longest; 56.14101; 56.14130; 56.9300; 56.14107) were diffed word by word
against the official eCFR rendering for 2026-10-01. They're identical apart from the trailing Federal Register
source note, which is deliberately not stored. All 20 sections split for indexing rebuild exactly from their chunks.
Dated versions (56.5001/56.5005 expired April 7, 2026; the `T` versions in force since April 8, 2026)
are handled in the tool layer (see DECISIONS).

## Manual review: SEVERITY mapping (2026-10-07)
Reviewed against the MSHA definition file and sample narratives. The six main buckets hold up.
Code 04 (days away *and* restricted) → `lost_time` is fair: 375 of its 385 rows have days lost.
Keeping code 08 (natural causes) out of `fatal` matches MSHA's chargeable count (see reconciliation below).
`other` (255 rows) is a mix and can't answer questions about any one of its parts:

| Code | Rows | What it contains (examples) |
|---|---|---|
| 07 Occupational illness | 114 | fume exposure, COVID-19 positive, strain pain |
| 00 Accident only | 68 | fire or stuck elevator, nobody hurt |
| 08 Natural causes | 41 | panic attack; a customer driver's fatal medical emergency |
| 10 All other / first aid | 18 | chipped tooth, trip |
| 09 Non-employees | 14 | customer or outside truck drivers |

The `accident_stats` severity description tells the model what `other` contains. Splitting it
into `illness` / `natural_causes` / `non_employee` / `no_injury` is a possible follow-up.

## Manual review: fatal counts reconciled with MSHA (2026-10-07)
Raw metal/nonmetal `FATALITY` records, counted *before* the surface filter, match MSHA's published
[Metal/Nonmetal Fatalities](https://www.msha.gov/metalnonmetal-fatalities) table exactly:

| Year | Raw M/NM fatal records | MSHA published | Kept (surface) | Excluded: underground | Excluded: surface at underground |
|---|---|---|---|---|---|
| 2021 | 27 | 27 | 21 | 6 | 0 |
| 2022 | 19 | 19 | 15 | 4 | 0 |
| 2023 | 31 | 31 | 27 | 3 | 1 |
| 2024 | 18 | 18 | 17 | 1 | 0 |
| **Total** | **95** | **95** | **80** | **14** | **1** |

No fatal record was dropped by the non-empty-narrative filter. The counts are fatal accident
*records*, so an accident with two deaths would count once.

## Manual review: keyword matches (2026-10-07)
The hybrid golden questions count fatal accidents by narrative keyword. All 12 matches were read by hand:

| Keyword | Matches | Keyword is the cause | Mentioned only | Can't tell |
|---|---|---|---|---|
| conveyor | 6 | 3 | 2 (crane load hit power line; manlift hit conveyor canopy) | 1 |
| berm | 3 | 2 | 1 (dozer slid off bench shelf) | 0 |
| harness / fall protection | 3 | 2 | 1 (was wearing it; engulfed in hopper) | 0 |

One accident (220231730009) matches both conveyor and fall protection. The single rule each question
expects doesn't fit every accident: a dump-site overtravel falls under §56.9301 rather than §56.9300,
a power-line contact under §56.12071, and a hopper engulfment under §56.16002. Keyword counts are
therefore reported as accidents *mentioning* the term (enforced since M8).

## Limitations
- **Short narratives:** median 195 characters, max 384 (the source appears to truncate them). They are thin evidence for root-cause questions.
- **Sample, not population:** the sample keeps every fatality, so severity mix in the sample over-represents fatalities. Statistics describe the sample and must say so.
- **Equipment often missing:** `MINING_EQUIP` is null for half the rows, so equipment filters undercount.
- **Mine attributes are current, not historical:** mine name and type reflect today's values, not those at accident time.
- **Scope:** Part 56 only. No Part 57 (underground), no other years.
