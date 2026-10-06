"""Parse 30 CFR Part 56 eCFR XML into one record per section, and chunk long sections."""

import re
import xml.etree.ElementTree as ET

# XREF holds editorial "link to an amendment" notes, not regulatory text.
SKIP_TAGS = {"HEAD", "CITA", "AUTH", "SOURCE", "XREF"}
# IDs can carry a suffix, e.g. 56.5001T (a newer version printed alongside the old one).
HEAD_RE = re.compile(r"^§\s*(?P<id>56\.\d+[A-Z]?)\s*(?P<heading>.*?)\.?$")


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_part56(xml: bytes | str, ecfr_date: str) -> list[dict]:
    """Return [{section_id, subpart, heading, text, source_url}] for every non-reserved section."""
    root = ET.fromstring(xml)
    records = []
    for subpart in root.iter("DIV6"):
        subpart_head = subpart.find("HEAD")
        subpart_name = _text(subpart_head) if subpart_head is not None else ""
        for sec in subpart.iter("DIV8"):
            if sec.get("TYPE") != "SECTION":
                continue
            match = HEAD_RE.match(_text(sec.find("HEAD")))
            if not match or "[Reserved]" in match["heading"]:
                continue
            paragraphs = [_text(child) for child in sec if child.tag not in SKIP_TAGS]
            text = "\n\n".join(p for p in paragraphs if p)
            if not text:
                continue
            sid = match["id"]
            records.append({
                "section_id": sid,
                "subpart": subpart_name,
                "heading": match["heading"].strip(),
                "text": text,
                "source_url": f"https://www.ecfr.gov/on/{ecfr_date}/title-30/section-{sid}",
            })
    return records


def chunk_section(record: dict, max_chars: int = 1500) -> list[dict]:
    """One chunk per section; long sections are split on paragraph boundaries.

    Every chunk keeps the section_id and heading so citations survive splitting.
    """
    paragraphs = record["text"].split("\n\n")
    groups, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            groups.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    groups.append(current)
    return [
        {**record, "chunk_id": f"{record['section_id']}#{i}", "text": text}
        for i, text in enumerate(groups)
    ]
