import re

from src.ingestion.models import ContentType, IngestionItem

_ITEM_PATTERN = re.compile(
    r"^item\s+(\d+[a-z]?|\d+\.\d+)\.?\s*[-–—:]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_PART_PATTERN = re.compile(r"^part\s+(i{1,3}|iv)\b", re.IGNORECASE | re.MULTILINE)
_TABLE_MARKER_PATTERN = re.compile(r"\[\[TABLE_MARKER:([^\]]+)\]\]")



# SEC's item numbers have fixed, regulation-mandated meanings
_TENK_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules", "16": "Form 10-K Summary",
}
_TENQ_TITLES = {
    "I": {
        "1": "Financial Statements",
        "2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "3": "Quantitative and Qualitative Disclosures About Market Risk",
        "4": "Controls and Procedures",
    },
    "II": {
        "1": "Legal Proceedings", "1A": "Risk Factors",
        "2": "Unregistered Sales of Equity Securities and Use of Proceeds",
        "3": "Defaults Upon Senior Securities", "4": "Mine Safety Disclosures",
        "5": "Other Information", "6": "Exhibits",
    },
}

def _canonical_title(filing_type: str | None , part: str | None , item_num: str) -> str | None:
    if filing_type == "10-K":
        return _TENK_TITLES.get(item_num)
    if filing_type == "10-Q" and part in _TENQ_TITLES:
        return _TENQ_TITLES[part].get(item_num)
    return None

def split_sec_sections(items: list[IngestionItem], filing_type: str | None) -> list[IngestionItem]:
    result: list[IngestionItem] = []
    table_section: dict[str , str] = {}

    for item in items:
        if item.content_type != ContentType.TEXT:
            continue
        for sub in _split_text_item(item , filing_type):
            section = sub.metadata.get("section")
            if section:
                for table_id in _TABLE_MARKER_PATTERN.findall(sub.content):
                    table_section[table_id] = section
            sub.content = _TABLE_MARKER_PATTERN.sub("", sub.content).strip()
            result.append(sub)

    for item in items:
        if item.content_type == ContentType.TEXT:
            continue
        section = table_section.get(item.id)
        metadata = {**item.metadata, "section" : section} if section else item.metadata
        result.append(IngestionItem(
            id=item.id , content=item.content, content_type=item.content_type, metadata=metadata,
        ))

    return result

def _split_text_item(item: IngestionItem, filing_type: str | None) -> list[IngestionItem]:
    text = item.content
    item_matches = list(_ITEM_PATTERN.finditer(text))
    if not item_matches:
        return [item]
    part_matches = list(_PART_PATTERN.finditer(text))

    boundaries = _real_section_boundaries(text, item_matches, part_matches, filing_type)
    if not boundaries:
        return [item]

    sub_items = []
    if boundaries[0][0] > 0:
        preamble = text[: boundaries[0][0]].strip()
        if preamble:
            sub_items.append(IngestionItem(
                id=f"{item.id}:preamble", content=preamble,
                content_type=ContentType.TEXT, metadata=dict(item.metadata),
            ))

    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        sub_items.append(IngestionItem(
            id=f"{item.id}:{i}", content=section_text, content_type=ContentType.TEXT,
            metadata={**item.metadata, "section": label},
        ))
    return sub_items

def _real_section_boundaries(
    text: str,
    item_matches: list[re.Match],
    part_matches: list[re.Match],
    filing_type: str | None,
) -> list[tuple[int, str]]:
    """Reduces raw regex matches down to genuine section boundaries. A single Part-aware linear
    scan tracks which "Part" is active as we walk through the document; for each distinct
    (part, item_num) pair, only the LAST match is kept, since SEC filings' Table of Contents
    always lists every item once, near the top, before the real heading appears later. No
    minimum-gap filter: some real sections (Item 6 "[Reserved]", Item 10/12/14 when satisfied
    by "incorporated by reference to the proxy statement") are legitimately short, and a gap
    filter can't tell those apart from Table-of-Contents noise — it would just as happily eat
    Item 7 (MD&A) if it happens to follow a short Item 6.
    """
    events = [(m.start(), "part", m.group(1).upper()) for m in part_matches]
    events += [(m.start(), "item", (m.group(1).upper(), m.group(2).strip())) for m in item_matches]
    events.sort(key=lambda e: e[0])

    current_part: str | None = None
    last_for_key: dict[str, tuple[int, str]] = {}
    for pos, kind, value in events:
        if kind == "part":
            current_part = value
            continue
        item_num, captured_title = value
        key = f"{current_part or ''}|{item_num}"
        title = _canonical_title(filing_type, current_part, item_num) or captured_title
        part_prefix = f"Part {current_part} " if current_part else ""
        label = f"{part_prefix}Item {item_num}" + (f" - {title}" if title else "")
        last_for_key[key] = (pos, label)

    return sorted(last_for_key.values(), key=lambda pair: pair[0])
