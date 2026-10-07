"""Build evals/golden_set.yaml from question specs, verifying every expectation against the DB.

Regulation facts must appear verbatim in the cited section text; aggregate/hybrid numbers
are computed by SQL at build time (never hand-typed). Run: python evals/build_golden.py
"""

import sqlite3
import sys
from pathlib import Path

import yaml

from mine_copilot.config import DB_PATH

OUT = Path(__file__).with_name("golden_set.yaml")
SAMPLE_NOTE = "Non-fatal counts describe the 3,000-row sample, not the population; say so."
# Phrase checks added after manual review (M8): stats that include non-fatal rows must flag the
# sample up front and never call it representative; keyword counts must say "mention", not "cause".
SAMPLE_CHECKS = {"lead_phrases": ["sample"], "forbidden_phrases": ["representative"]}
MENTION_CHECK = {"required_phrases": ["mention"]}
PARTIAL_CHECK = {"required_phrases": ["Not covered:"]}

QUESTIONS = [
    # --- reg: answer lives in one Part 56 section ---
    {"id": "reg-01", "type": "reg",
     "question": "What does MSHA require for moving machine parts like pulleys and flywheels?",
     "expected_sections": ["56.14107"],
     "expected_facts": ["shall be guarded", "pulleys"]},
    {"id": "reg-02", "type": "reg",
     "question": "When must miners wear safety belts and lines?",
     "expected_sections": ["56.15005"],
     "expected_facts": ["danger of falling", "second person shall tend the lifeline"]},
    {"id": "reg-03", "type": "reg",
     "question": "How high must berms be on haul roads?",
     "expected_sections": ["56.9300"],
     "expected_facts": ["mid-axle height"]},
    {"id": "reg-04", "type": "reg",
     "question": "What must be done before mechanical work on electrically powered equipment?",
     "expected_sections": ["56.12016"],
     "expected_facts": ["deenergized", "locked out"]},
    {"id": "reg-05", "type": "reg",
     "question": "What are the rules for doing repairs or maintenance on machinery?",
     "expected_sections": ["56.14105"],
     "expected_facts": ["power is off", "blocked against hazardous motion"]},
    {"id": "reg-06", "type": "reg",
     "question": "How often must working places be examined, and by whom?",
     "expected_sections": ["56.18002"],
     "expected_facts": ["competent person", "at least once each shift"]},
    {"id": "reg-07", "type": "reg",
     "question": "Who has to inspect self-propelled mobile equipment before a shift?",
     "expected_sections": ["56.14100"],
     "expected_facts": ["inspected by the equipment operator"]},
    {"id": "reg-08", "type": "reg",
     "question": "Which equipment needs roll-over protective structures (ROPS)?",
     "expected_sections": ["56.14130"],
     "expected_facts": ["Crawler tractors", "Graders", "Wheel loaders"]},
    {"id": "reg-09", "type": "reg",
     "question": "When are hard hats required?",
     "expected_sections": ["56.15002"],
     "expected_facts": ["falling objects"]},
    {"id": "reg-10", "type": "reg",
     "question": "What does Part 56 require for respirators when miners are exposed to dust?",
     "expected_sections": ["56.5005T"], "forbidden_sections": ["56.5005"],
     "expected_facts": ["written respiratory protection program", "ASTM F3387-19"],
     "notes": "56.5005 expired April 7, 2026; 56.5005T is in force."},
    # --- agg: answer is a SQL aggregate over accidents ---
    {"id": "agg-01", "type": "agg",
     "question": "How many fatal accidents were recorded in 2023?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023"},
    {"id": "agg-02", "type": "agg",
     "question": "How many fatalities were there in total from 2021 to 2024?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal'"},
    {"id": "agg-03", "type": "agg",
     "question": "Which year had the most fatal accidents?",
     "sql": "SELECT CAL_YR FROM accidents WHERE SEVERITY='fatal' "
            "GROUP BY CAL_YR ORDER BY COUNT(*) DESC LIMIT 1"},
    {"id": "agg-04", "type": "agg",
     "question": "What accident classification caused the most fatalities in 2023?",
     "sql": "SELECT CLASSIFICATION FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023 "
            "GROUP BY CLASSIFICATION ORDER BY COUNT(*) DESC LIMIT 1"},
    {"id": "agg-05", "type": "agg",
     "question": "What is the most common accident classification overall?",
     "sql": "SELECT CLASSIFICATION FROM accidents "
            "GROUP BY CLASSIFICATION ORDER BY COUNT(*) DESC LIMIT 1",
     "notes": SAMPLE_NOTE, **SAMPLE_CHECKS},
    {"id": "agg-06", "type": "agg",
     "question": "Which state has the most recorded accidents?",
     "sql": "SELECT m.STATE FROM accidents a JOIN mines m USING(MINE_ID) "
            "GROUP BY m.STATE ORDER BY COUNT(*) DESC LIMIT 1",
     "notes": SAMPLE_NOTE, **SAMPLE_CHECKS},
    {"id": "agg-07", "type": "agg",
     "question": "On average, how many days were lost in lost-time accidents?",
     "sql": "SELECT ROUND(AVG(DAYS_LOST), 1) FROM accidents WHERE SEVERITY='lost_time'",
     "notes": SAMPLE_NOTE, **SAMPLE_CHECKS},
    # --- hybrid: accident records + the regulation that applies ---
    {"id": "hyb-01", "type": "hybrid",
     "question": "How many fatal accidents involved conveyors, and what rule covers guarding them?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%conveyor%'",
     "expected_sections": ["56.14107"], **MENTION_CHECK},
    {"id": "hyb-02", "type": "hybrid",
     "question": "Were there fatalities involving berms, and what does Part 56 require for them?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%berm%'",
     "expected_sections": ["56.9300"], **MENTION_CHECK},
    {"id": "hyb-03", "type": "hybrid",
     "question": "How many fatal accidents mention a harness or fall protection, "
                 "and which rule applies?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' "
            "AND (NARRATIVE LIKE '%fall protection%' OR NARRATIVE LIKE '%harness%')",
     "expected_sections": ["56.15005"], **MENTION_CHECK},
    {"id": "hyb-04", "type": "hybrid",
     "question": "How many 2023 fatalities were classified as MACHINERY, and what rule covers "
                 "maintenance on machinery?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2023 "
            "AND CLASSIFICATION='MACHINERY'",
     "expected_sections": ["56.14105"]},
    # --- partial: one part in scope, one not. Answer the covered part, then "Not covered:" ---
    {"id": "part-01", "type": "partial",
     "question": "Do new quarry workers need hard hats, and what safety training must they "
                 "complete before starting?",
     "expected_sections": ["56.15002"], "notes": "Training is 30 CFR Part 46, not Part 56.",
     **PARTIAL_CHECK},
    {"id": "part-02", "type": "partial",
     "question": "What are the berm requirements on haul roads, and how much does it cost to "
                 "build one?",
     "expected_sections": ["56.9300"], "notes": "Costs are not in any source.", **PARTIAL_CHECK},
    {"id": "part-03", "type": "partial",
     "question": "What does Part 56 say about guarding conveyors, and what does OSHA require for "
                 "conveyors in a warehouse?",
     "expected_sections": ["56.14107"], "notes": "OSHA / non-mine workplaces out of scope.",
     **PARTIAL_CHECK},
    {"id": "part-04", "type": "partial",
     "question": "What must be done before working on electrical equipment, and what is the "
                 "fine if MSHA finds we skipped it?",
     "expected_sections": ["56.12016"], "notes": "Penalties are 30 CFR Part 100.",
     **PARTIAL_CHECK},
    {"id": "part-05", "type": "partial",
     "question": "How many fatal accidents mentioned conveyors from 2021 through 2025?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND NARRATIVE LIKE '%conveyor%'",
     "notes": "2025 is outside the data; give the 2021-2024 count.", **PARTIAL_CHECK},
    {"id": "part-06", "type": "partial",
     "question": "How many fatal accidents were there at surface metal/nonmetal mines in 2024, "
                 "and how many at underground mines?",
     "sql": "SELECT COUNT(*) FROM accidents WHERE SEVERITY='fatal' AND CAL_YR=2024",
     "notes": "Underground (Part 57) data not included.", **PARTIAL_CHECK},
    {"id": "part-07", "type": "partial",
     "question": "What is the exposure limit for respirable crystalline silica at my quarry?",
     "expected_sections": ["56.5001T"], "forbidden_sections": ["56.5001"],
     "notes": "56.5001T (in force) defers silica to 30 CFR Part 60, which is not in the corpus.",
     **PARTIAL_CHECK},
    # --- refuse: outside scope, the copilot must decline ---
    {"id": "ref-01", "type": "refuse",
     "question": "What does 30 CFR Part 75 require for roof bolting in underground coal mines?",
     "notes": "Coal / Part 75 out of scope."},
    {"id": "ref-02", "type": "refuse",
     "question": "What are the ventilation requirements for underground metal mines?",
     "notes": "Underground (Part 57) out of scope."},
    {"id": "ref-03", "type": "refuse",
     "question": "How many fatal accidents happened in 2018?",
     "notes": "Year outside 2021-2024 coverage."},
    {"id": "ref-04", "type": "refuse",
     "question": "What's a good stock to buy in the mining sector?",
     "notes": "Off-topic."},
]


def build(db_path: Path = DB_PATH) -> list[dict]:
    con = sqlite3.connect(db_path)
    regs = dict(con.execute("SELECT section_id, text FROM regulations"))
    out = []
    for q in QUESTIONS:
        item = {k: v for k, v in q.items() if k != "sql"}
        item["must_refuse"] = q["type"] == "refuse"
        for sec in q.get("expected_sections", []) + q.get("forbidden_sections", []):
            assert sec in regs, f"{q['id']}: section {sec} not in DB"
        for fact in q.get("expected_facts", []):
            sec_text = " ".join(regs[s] for s in q["expected_sections"])
            assert fact in sec_text, f"{q['id']}: fact {fact!r} not in section text"
        if "sql" in q:
            value = con.execute(q["sql"]).fetchone()[0]
            assert value not in (None, 0), f"{q['id']}: SQL returned {value!r}"
            item["expected_value"] = value
            item["verified_by"] = q["sql"]
        elif q["type"] == "partial":
            item["verified_by"] = ("sections exist; out-of-scope part: " + q["notes"])
        elif q["type"] == "reg":
            item["verified_by"] = "facts found verbatim in " + ", ".join(q["expected_sections"])
        else:
            item["verified_by"] = "scope rule (CLAUDE.md)"
        out.append(item)
    con.close()
    return out


if __name__ == "__main__":
    items = build(Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH)
    OUT.write_text(yaml.safe_dump(items, sort_keys=False, width=100))
    print(f"wrote {len(items)} questions to {OUT}")
