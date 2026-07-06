#!/usr/bin/env python3
"""
1_parser.py — CUAD v1 Dataset Parser
======================================
Parses the SQuAD-style CUAD v1 JSON (510 commercial legal contracts),
extracts the full contract body, and prepends a structured metadata
header containing the contract title, parties, date, and type.

Returns a list of dicts:
    {
        "metadata":      str  – standalone structured header (for indexing)
        "text":          str  – header + full contract body (for embedding / LLM input)
        "contract_type": str  – normalised category label
        "raw_title":     str  – original CUAD filename-style title
        "char_count":    int  – body character count (pre-header)
    }

Corpus size: ~100–200 contracts via stratified sampling across the 25
contract types, preserving type-level proportions for representativeness.

Usage (CLI):
    python 1_parser.py                          # default: 150 contracts, saves JSON
    python 1_parser.py --n 200 --output out.json
    python 1_parser.py --no-sample              # all 510 contracts
"""

import json
import re
import random
import logging
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
CUAD_JSON_PATH = Path("/mnt/user-data/uploads/CUAD_v1.json")
OUTPUT_PATH    = Path("/mnt/user-data/outputs/parsed_contracts.json")

N_CONTRACTS    = 150    # target corpus size  (100–200 range)
RANDOM_SEED    = 42     # reproducible sampling
MIN_CHAR_COUNT = 500    # drop near-empty or malformed entries
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Title Parsing
# ═══════════════════════════════════════════════════════════════════════════════
#
# CUAD titles follow three patterns sourced from SEC EDGAR filenames:
#
# Pattern A  —  old-style, uppercase:
#   LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
#   Company  : LIMEENERGYCO
#   Date     : 09/09/1999  (MM_DD_YYYY)
#   Type     : DISTRIBUTOR AGREEMENT
#
# Pattern B  —  CamelCase with 8-digit date:
#   DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_Promotion Agreement
#   Company  : DovaPharmaceuticalsInc
#   Date     : 11/08/2018  (YYYYMMDD)
#   Type     : Promotion Agreement  (last underscore-token)
#
# Pattern C  —  free-text with " - " separator (no encoded date):
#   MetLife, Inc. - Remarketing Agreement
#   VIVINT SOLAR, INC. - NON-COMPETITION AGREEMENT
#   Company  : everything left of " - "
#   Date     : unknown
#   Type     : everything right of " - "


# Regex for Pattern A date  (MM_DD_YYYY inside an underscored title)
_RE_DATE_A = re.compile(r'(?<![A-Z\d])(\d{2})[_](\d{2})[_](\d{4})(?!\d)')
# Regex for Pattern B date  (8-digit YYYYMMDD after first underscore)
_RE_DATE_B = re.compile(r'_(\d{4})(\d{2})(\d{2})_')


def _parse_date(title: str) -> Optional[str]:
    """Return 'MM/DD/YYYY' or None."""
    m = _RE_DATE_A.search(title)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{mm}/{dd}/{yyyy}"
    m = _RE_DATE_B.search(title)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{mm}/{dd}/{yyyy}"
    return None


def _parse_company(title: str) -> str:
    """
    Extract a company identifier from the title.
    Returned as-is from the source; normalisation (camelCase splitting,
    comma insertion, etc.) is left to downstream stages to avoid lossy guessing.
    """
    # Pattern C:  "Company Name - Contract Type"
    if " - " in title:
        return title.split(" - ")[0].strip()

    # Patterns A & B: company is the segment before the first date
    m_a = _RE_DATE_A.search(title)
    if m_a:
        prefix = title[: m_a.start()]          # e.g. "LIMEENERGYCO_"
        return prefix.strip("_- ")

    m_b = _RE_DATE_B.search(title)
    if m_b:
        prefix = title[: m_b.start()]          # e.g. "DovaPharmaceuticalsInc"
        return prefix.strip("_- ")

    # Fallback: use the whole title (rare; catches edge cases like XACCT)
    return title.strip()


def _trim_to_agreement(text: str) -> str:
    """
    Extract the contract-type phrase from a text fragment, ending at the
    LAST occurrence of 'agreement', 'contract', or 'statement'.

    Using a greedy `.*` means the regex engine backtracks from the right,
    always landing on the last occurrence – which correctly handles both:
      • single-type  "Cooperation Agreement of 50MWp…"   → "Cooperation Agreement"
      • multi-type   "Manufacturing Agreement_ Supply Agreement" → same (full span)

    Examples (→ expected output):
      "Cooperation Agreement of 50MWp Photovoltaic…"    → "Cooperation Agreement"
      "INTELLECTUAL PROPERTY AGREEMENT between THE…"    → "Intellectual Property Agreement"
      "Manufacturing Agreement_ Supply Agreement"        → "Manufacturing Agreement, Supply Agreement"
    """
    # Greedy .*  →  finds the LAST occurrence of the keyword word
    m = re.search(r'(?is)(.*(?:agreement|contract|statement)s?)', text)
    if m:
        result = m.group(1).strip()
        # Normalise underscore-joined multi-type titles
        result = re.sub(r'\s*_\s*', ', ', result)
        # Title-case for readability when source is ALL-CAPS
        if result.upper() == result and len(result) > 3:
            result = result.title()
        return result.strip(", ")
    # Fallback: return cleaned text, capped at 80 chars
    return text.strip("_- ")[:80]


def _parse_contract_type(title: str) -> str:
    """
    Extract a human-readable contract type string from the raw title.

    Strategy (in order of precedence):
    1. Pattern C:  right-hand side of ' - '.
    2. Pattern A:  text immediately after the exhibit reference marker.
    3. Pattern B:  last '_'-delimited non-numeric, non-exhibit token.
    4. Fallback:   'Agreement'.

    In all cases the result is trimmed to end at the first occurrence of
    the word 'agreement' / 'contract' / 'statement', so long appended
    descriptions (party names, locations, project names) are stripped.
    """
    # Pattern C:  "Company Name - CONTRACT TYPE [long description]"
    if " - " in title:
        raw = title.split(" - ", maxsplit=1)[-1].strip().strip("_")
        return _trim_to_agreement(raw)

    # Pattern A:  COMPANY_MM_DD_YYYY-EX-NUM-CONTRACT TYPE [description]
    if _RE_DATE_A.search(title):
        # Grab everything after the last exhibit-number token  (e.g. -EX-10.3-)
        m_ex = re.search(r'-EX[\-\.\d\w]*-(.*)', title, re.I)
        if m_ex:
            rest = m_ex.group(1).strip()
            if rest:                           # non-empty after the exhibit ref
                return _trim_to_agreement(rest)
        # Fallback: last '-' token that is not an exhibit marker
        for part in reversed(title.split("-")):
            p = part.strip()
            if p and not re.fullmatch(r'EX[\-\d\.]+', p, re.I):
                return _trim_to_agreement(p)

    # Pattern B:  CompanyCamelCase_YYYYMMDD_FORM_EX-N_ID_EX-N_Contract Type [_ Type2]
    if _RE_DATE_B.search(title):
        # Take everything after the last exhibit-number token.
        # This naturally captures multi-type titles such as
        #   "…_EX-2.6_Manufacturing Agreement_ Supply Agreement"
        #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
        m_last_ex = re.search(
            r'.*_(?:EX[\-\.\d\w]+|EX\d[A-Z][\w\-]+)_(.*)',
            title, re.I
        )
        if m_last_ex:
            rest = m_last_ex.group(1).strip()
            if rest:
                return _trim_to_agreement(rest)

        # Fallback: reverse-scan underscore tokens, skip exhibit / numeric ids
        _form_re = re.compile(
            r'^(?:EX[\-\.\d]+|F-\d+|S-\d+|10-|8-K|20-F|1-A|N-\d|POS|DRS)',
            re.I,
        )
        tokens = [t for t in title.split("_") if t and not re.fullmatch(r'\d+', t)]
        for part in reversed(tokens):
            p = part.strip()
            if p and not _form_re.match(p):
                return _trim_to_agreement(p)

    return "Agreement"


def build_metadata_header(raw_title: str) -> str:
    """
    Compose a structured metadata block to prepend to every contract.
    This preserves global document context even after chunking.

    Format (8 lines, human- and machine-readable):

        ══════════════════════════════════
        CONTRACT METADATA
        ══════════════════════════════════
        Contract Type : Distributor Agreement
        Company       : LIME ENERGY CO
        Date          : 09/09/1999
        Source File   : LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
        ══════════════════════════════════
    """
    divider  = "═" * 50
    ctype    = _parse_contract_type(raw_title)
    company  = _parse_company(raw_title)
    date     = _parse_date(raw_title) or "Not specified"

    lines = [
        divider,
        "CONTRACT METADATA",
        divider,
        f"Contract Type : {ctype}",
        f"Company       : {company}",
        f"Date          : {date}",
        f"Source File   : {raw_title}",
        divider,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Contract-Type Detection (for stratified sampling)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Maps each of the 25 CUAD contract categories to a keyword pattern.
# Longest/most-specific patterns are checked first to avoid false matches
# (e.g. "co-branding" should not also match plain "brand").

_TYPE_MAP: list[tuple[str, str]] = [
    # (keyword substring to find in lower-case title, canonical label)
    ("non-competition",          "Non-Compete Agreement"),
    ("non-compete",              "Non-Compete Agreement"),
    ("non-solicit",              "Non-Compete Agreement"),
    ("non-disparagement",        "Non-Compete Agreement"),
    ("strategic alliance",       "Strategic Alliance Agreement"),
    ("strategic sales",          "Strategic Alliance Agreement"),
    ("joint venture",            "Joint Venture Agreement"),
    ("joint filing",             "Joint Venture Agreement"),        # minor alias
    ("co-branding",              "Co-Branding Agreement"),
    ("co-hosting",               "Co-Branding Agreement"),
    ("collaboration",            "Collaboration Agreement"),
    ("cooperation",              "Collaboration Agreement"),
    ("development",              "Development Agreement"),
    ("distributor",              "Distributor Agreement"),
    ("distribution",             "Distributor Agreement"),
    ("outsourcing",              "Outsourcing Agreement"),
    ("franchise",                "Franchise Agreement"),
    ("endorsement",              "Endorsement Agreement"),
    ("sponsorship",              "Sponsorship Agreement"),
    ("sponsor",                  "Sponsorship Agreement"),
    ("promotion",                "Promotion Agreement"),
    ("reseller",                 "Reseller Agreement"),
    ("affiliate",                "Affiliate Agreement"),
    ("consulting",               "Consulting Agreement"),
    ("agency",                   "Agency Agreement"),
    ("transportation",           "Transportation Agreement"),
    ("manufacturing",            "Manufacturing Agreement"),
    ("supply",                   "Supply Agreement"),
    ("maintenance",              "Maintenance Agreement"),
    ("hosting",                  "Hosting Agreement"),
    ("marketing",                "Marketing Agreement"),
    ("intellectual property",    "IP Agreement"),
    ("ip agreement",             "IP Agreement"),
    ("license",                  "License Agreement"),
    ("service",                  "Service Agreement"),
    ("servicing",                "Service Agreement"),
    ("outsourc",                 "Outsourcing Agreement"),
]


def detect_contract_type(raw_title: str) -> str:
    """Return a normalised contract-type label for stratified sampling."""
    lower = raw_title.lower()
    for keyword, label in _TYPE_MAP:
        if keyword in lower:
            return label
    return "Other"


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Core Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_cuad(path: Path) -> list[dict]:
    """
    Load the CUAD v1 JSON and return the list of SQuAD-style articles.

    Expected top-level structure:
        {"version": "1.0", "data": [ <article>, ... ]}

    Each article:
        {"title": <str>, "paragraphs": [{"context": <str>, "qas": [...]}]}
    """
    log.info(f"Loading {path.name}  ({path.stat().st_size / 1_048_576:.1f} MB) …")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    version  = payload.get("version", "unknown")
    articles = payload.get("data", [])
    log.info(f"  version={version}   articles={len(articles)}")
    return articles


def extract_contracts(articles: list[dict]) -> list[dict]:
    """
    Iterate over CUAD articles, build the metadata header, and return
    a flat list of contract records.

    Key design decisions
    ────────────────────
    • Each CUAD article has exactly ONE paragraph whose 'context' field
      holds the *full* contract text (verified: 0 multi-paragraph articles).

    • The metadata header is both stored separately (for downstream indexing /
      filtering) and prepended to the full text (so any chunking strategy
      automatically inherits global document context).

    • The QA annotations (41 clause categories × up to N answers per contract)
      are intentionally discarded here; they belong in a separate labels file
      and should not pollute the raw text corpus.
    """
    records: list[dict] = []
    n_skipped = 0

    for article in articles:
        raw_title  = article.get("title", "").strip()
        paragraphs = article.get("paragraphs", [])

        if not paragraphs:
            log.warning(f"  Skipped (no paragraphs): {raw_title[:80]}")
            n_skipped += 1
            continue

        # CUAD invariant: one paragraph per article  →  paragraphs[0]
        body = paragraphs[0].get("context", "").strip()

        if len(body) < MIN_CHAR_COUNT:
            log.warning(f"  Skipped (too short, {len(body)} chars): {raw_title[:80]}")
            n_skipped += 1
            continue

        header    = build_metadata_header(raw_title)
        full_text = f"{header}\n\n{body}"

        records.append({
            "metadata":      header,             # structured header only
            "text":          full_text,           # header + body (primary field)
            "contract_type": detect_contract_type(raw_title),
            "raw_title":     raw_title,
            "char_count":    len(body),
        })

    log.info(f"  Extracted {len(records)} valid contracts  ({n_skipped} skipped)")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Stratified Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_sample(
    records: list[dict],
    n:       int,
    seed:    int,
) -> list[dict]:
    """
    Proportional stratified sample over contract types.

    Each type gets max(1, round(n × |type| / |total|)) slots, so the
    sampled corpus mirrors the original type distribution while staying
    close to the target size n.  Rounding drift is corrected by a final
    trim / random top-up step.

    Args:
        records: full list of extracted contracts
        n:       target corpus size
        seed:    RNG seed for reproducibility

    Returns:
        Shuffled list of sampled contract records (len ≈ n).
    """
    rng = random.Random(seed)

    # Group by type
    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(rec["contract_type"], []).append(rec)

    total   = len(records)
    sampled: list[dict] = []

    log.info(f"\nStratified sampling  (target={n}, seed={seed}):")
    log.info(f"  {'Contract Type':<45}  {'Available':>9}  {'Sampled':>7}")
    log.info(f"  {'-'*45}  {'-'*9}  {'-'*7}")

    for ctype in sorted(buckets):
        group = buckets[ctype]
        quota = max(1, round(n * len(group) / total))
        chosen = rng.sample(group, min(quota, len(group)))
        sampled.extend(chosen)
        log.info(f"  {ctype:<45}  {len(group):>9}  {len(chosen):>7}")

    # Correct rounding drift
    rng.shuffle(sampled)
    if len(sampled) > n:
        sampled = sampled[:n]
    elif len(sampled) < n:
        sampled_ids  = {id(r) for r in sampled}
        remaining    = [r for r in records if id(r) not in sampled_ids]
        top_up_count = min(n - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, top_up_count))

    log.info(f"  {'─'*63}")
    log.info(f"  {'TOTAL':<45}  {total:>9}  {len(sampled):>7}")
    return sampled


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_cuad(
    input_path:  Path = CUAD_JSON_PATH,
    n_contracts: int  = N_CONTRACTS,
    seed:        int  = RANDOM_SEED,
    sample:      bool = True,
) -> list[dict]:
    """
    End-to-end CUAD parsing pipeline.

    Parameters
    ----------
    input_path  : path to CUAD_v1.json
    n_contracts : target corpus size (ignored when sample=False)
    seed        : RNG seed for stratified sampling
    sample      : True → stratified subset; False → all 510 contracts

    Returns
    -------
    List of dicts with keys:
        metadata      – structured header block
        text          – header + full contract body
        contract_type – normalised category (e.g. 'Distributor Agreement')
        raw_title     – original CUAD title string
        char_count    – body length in characters
    """
    articles    = load_cuad(input_path)
    all_records = extract_contracts(articles)

    if not sample:
        log.info("sample=False → returning all valid contracts")
        return all_records

    return stratified_sample(all_records, n=n_contracts, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(corpus: list[dict]) -> None:
    """Print a human-readable corpus summary to stdout."""
    sep = "─" * 60

    # Per-type breakdown
    type_counts: dict[str, int] = {}
    for r in corpus:
        type_counts[r["contract_type"]] = type_counts.get(r["contract_type"], 0) + 1

    total_chars = sum(r["char_count"] for r in corpus)
    avg_chars   = total_chars // len(corpus)
    min_chars   = min(r["char_count"] for r in corpus)
    max_chars   = max(r["char_count"] for r in corpus)

    print(f"\n{sep}")
    print(f"  CORPUS SUMMARY  ({len(corpus)} contracts)")
    print(sep)
    print(f"  {'Contract Type':<45}  {'Count':>5}")
    print(f"  {'-'*45}  {'-'*5}")
    for ct, cnt in sorted(type_counts.items()):
        print(f"  {ct:<45}  {cnt:>5}")
    print(f"\n  {'Char stats (body only)':}")
    print(f"    total : {total_chars:>12,}")
    print(f"    avg   : {avg_chars:>12,}")
    print(f"    min   : {min_chars:>12,}")
    print(f"    max   : {max_chars:>12,}")

    # Sample record
    first = corpus[0]
    print(f"\n{sep}")
    print("  SAMPLE RECORD  [index 0]")
    print(sep)
    print(f"  raw_title    : {first['raw_title']}")
    print(f"  contract_type: {first['contract_type']}")
    print(f"  char_count   : {first['char_count']:,}")
    print(f"\n  ── metadata ──\n{first['metadata']}")
    print(f"\n  ── text[:400] ──\n{first['text'][:400]}")
    print(f"{sep}\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse CUAD v1 SQuAD JSON → structured contract corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",     default=str(CUAD_JSON_PATH), help="Path to CUAD_v1.json")
    ap.add_argument("--output",    default=str(OUTPUT_PATH),    help="Output JSON path")
    ap.add_argument("--n",         type=int, default=N_CONTRACTS, help="Target corpus size")
    ap.add_argument("--seed",      type=int, default=RANDOM_SEED, help="Random seed")
    ap.add_argument("--no-sample", action="store_true",
                    help="Return all valid contracts instead of sampling")
    args = ap.parse_args()

    corpus = parse_cuad(
        input_path  = Path(args.input),
        n_contracts = args.n,
        seed        = args.seed,
        sample      = not args.no_sample,
    )

    _print_summary(corpus)

    # Persist
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(corpus, fh, indent=2, ensure_ascii=False)

    size_kb = out.stat().st_size / 1024
    log.info(f"Saved → {out}  ({size_kb:,.1f} KB)")