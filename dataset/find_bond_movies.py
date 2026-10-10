#!/usr/bin/env python3
"""
find_bond_movies.py -- collect every James Bond film from the actor filmography
files in this folder and write them to a single CSV.

The filmography files (sean-connery_filmography.csv, ...) were built from Letterboxd,
so each row carries a Letterboxd `slug`. That slug is what we match on: it is exact
and unambiguous.

Deliberately NOT a "grep for bond" script. A keyword search over these files also
hits ~40 documentaries, TV specials and making-ofs -- "The Sound of 007",
"Top Gear: 50 Years of Bond Cars", "Being James Bond", "Happy Anniversary 007" --
none of which are Bond films. Instead we match against the canonical list below and
report the near-misses separately so you can see what was skipped and why.

Reads:  *_filmography.csv (falls back to *_filmography.json if a CSV is missing)
Writes: james_bond_movies.csv  (UTF-8, CRLF, same column style as the source files)

Usage:
    python find_bond_movies.py
    python find_bond_movies.py --folder . --out james_bond_movies.csv
    python find_bond_movies.py --only-found      # drop films absent from the folder
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# The canon. (official_number, title, year, slug, series, lead actor)
# official_number is the Eon Productions numbering; None = not an Eon film.
# --------------------------------------------------------------------------
EON = "Eon"
NON_EON = "non-Eon"

BOND_FILMS: list[tuple[int | None, str, int, str, str, str]] = [
    (1,    "Dr. No",                            1962, "dr-no",                              EON,     "Sean Connery"),
    (2,    "From Russia with Love",             1963, "from-russia-with-love",              EON,     "Sean Connery"),
    (3,    "Goldfinger",                        1964, "goldfinger",                         EON,     "Sean Connery"),
    (4,    "Thunderball",                       1965, "thunderball",                        EON,     "Sean Connery"),
    (5,    "You Only Live Twice",               1967, "you-only-live-twice",                EON,     "Sean Connery"),
    (None, "Casino Royale",                     1967, "casino-royale-1967",                 NON_EON, "David Niven"),
    (6,    "On Her Majesty's Secret Service",   1969, "on-her-majestys-secret-service",     EON,     "George Lazenby"),
    (7,    "Diamonds Are Forever",              1971, "diamonds-are-forever",               EON,     "Sean Connery"),
    (8,    "Live and Let Die",                  1973, "live-and-let-die",                   EON,     "Roger Moore"),
    (9,    "The Man with the Golden Gun",       1974, "the-man-with-the-golden-gun",        EON,     "Roger Moore"),
    (10,   "The Spy Who Loved Me",              1977, "the-spy-who-loved-me",               EON,     "Roger Moore"),
    (11,   "Moonraker",                         1979, "moonraker",                          EON,     "Roger Moore"),
    (12,   "For Your Eyes Only",                1981, "for-your-eyes-only",                 EON,     "Roger Moore"),
    (13,   "Octopussy",                         1983, "octopussy",                          EON,     "Roger Moore"),
    (None, "Never Say Never Again",             1983, "never-say-never-again",              NON_EON, "Sean Connery"),
    (14,   "A View to a Kill",                  1985, "a-view-to-a-kill",                   EON,     "Roger Moore"),
    (15,   "The Living Daylights",              1987, "the-living-daylights",               EON,     "Timothy Dalton"),
    (16,   "Licence to Kill",                   1989, "licence-to-kill",                    EON,     "Timothy Dalton"),
    (17,   "GoldenEye",                         1995, "goldeneye",                          EON,     "Pierce Brosnan"),
    (18,   "Tomorrow Never Dies",               1997, "tomorrow-never-dies",                EON,     "Pierce Brosnan"),
    (19,   "The World Is Not Enough",           1999, "the-world-is-not-enough",            EON,     "Pierce Brosnan"),
    (20,   "Die Another Day",                   2002, "die-another-day",                    EON,     "Pierce Brosnan"),
    (21,   "Casino Royale",                     2006, "casino-royale-2006",                 EON,     "Daniel Craig"),
    (22,   "Quantum of Solace",                 2008, "quantum-of-solace",                  EON,     "Daniel Craig"),
    (23,   "Skyfall",                           2012, "skyfall",                            EON,     "Daniel Craig"),
    (24,   "Spectre",                           2015, "spectre-2015",                       EON,     "Daniel Craig"),
    (25,   "No Time to Die",                    2021, "no-time-to-die-2021",                EON,     "Daniel Craig"),
]

OUT_COLUMNS = [
    "official_number", "title", "year", "slug", "url",
    "series", "lead_actor", "found_in", "source_file",
    "rating", "genres", "cast", "directors", "poster",
]

# Rows whose title mentions Bond but that are not Bond films (documentaries etc.).
NEAR_MISS_RE = re.compile(r"\bbond\b|\b007\b|royale", re.IGNORECASE)


@dataclass
class Entry:
    """A row found in one of the filmography files."""
    title: str
    year: str
    slug: str
    url: str
    row: dict
    source_file: str
    actor: str


@dataclass
class Match:
    """A canonical Bond film, plus whatever we found for it in the folder."""
    number: int | None
    title: str
    year: int
    slug: str
    series: str
    lead_actor: str
    entry: Entry | None = None
    matched_by: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.entry is not None


def norm(text: str) -> str:
    """Lowercase, strip punctuation/accents-ish, collapse whitespace."""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def actor_from_filename(path: str) -> str:
    """'sean-connery_filmography.csv' -> 'Sean Connery'."""
    stem = os.path.basename(path)
    stem = re.sub(r"_filmography\.(csv|json)$", "", stem, flags=re.IGNORECASE)
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", stem) if w)


def load_entries(folder: str) -> tuple[list[Entry], list[str]]:
    """Read every *_filmography.csv in `folder`. Fall back to the .json twin."""
    entries: list[Entry] = []
    csv_paths = sorted(glob.glob(os.path.join(folder, "*_filmography.csv")))
    used: list[str] = []
    missing_json: list[str] = []

    for path in csv_paths:
        actor = actor_from_filename(path)
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                entries.append(Entry(
                    title=(row.get("title") or "").strip(),
                    year=(row.get("year") or "").strip(),
                    slug=(row.get("slug") or "").strip(),
                    url=(row.get("url") or "").strip(),
                    row=row,
                    source_file=os.path.basename(path),
                    actor=actor,
                ))
        used.append(os.path.basename(path))

    # If a filmography exists only as JSON, read it too so nothing is silently skipped.
    for path in sorted(glob.glob(os.path.join(folder, "*_filmography.json"))):
        csv_twin = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(csv_twin):
            continue
        actor = actor_from_filename(path)
        try:
            with open(path, encoding="utf-8-sig") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! could not read {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            entries.append(Entry(
                title=str(row.get("title") or "").strip(),
                year=str(row.get("year") or "").strip(),
                slug=str(row.get("slug") or "").strip(),
                url=str(row.get("url") or "").strip(),
                row={k: row.get(k) for k in OUT_COLUMNS},
                source_file=os.path.basename(path),
                actor=actor,
            ))
        missing_json.append(os.path.basename(path))

    return entries, used + missing_json


def build_index(entries: list[Entry]) -> tuple[dict[str, Entry], dict[tuple[str, str], Entry]]:
    """Index entries by slug, and by (normalised title, year) as a fallback."""
    by_slug: dict[str, Entry] = {}
    by_title_year: dict[tuple[str, str], Entry] = {}
    for e in entries:
        if e.slug:
            by_slug.setdefault(e.slug, e)
        if e.title:
            by_title_year.setdefault((norm(e.title), e.year), e)
    return by_slug, by_title_year


def resolve(film: tuple, by_slug: dict, by_title_year: dict) -> Match:
    number, title, year, slug, series, lead = film
    m = Match(number=number, title=title, year=year, slug=slug, series=series, lead_actor=lead)

    entry = by_slug.get(slug)
    if entry is not None:
        m.entry, m.matched_by = entry, "slug"
    else:
        entry = by_title_year.get((norm(title), str(year)))
        if entry is not None:
            m.entry, m.matched_by = entry, "title+year"

    if m.entry is not None:
        m.sources.append(m.entry.source_file)
    return m


def to_row(m: Match, extra_sources: list[str]) -> dict:
    e = m.entry
    return {
        "official_number": "" if m.number is None else m.number,
        "title": e.title if e else m.title,
        "year": e.year if e and e.year else m.year,
        "slug": m.slug,
        "url": e.url if e else "",
        "series": m.series,
        "lead_actor": m.lead_actor,
        "found_in": e.actor if e else "",
        "source_file": "; ".join([m.entry.source_file] + extra_sources) if e else "",
        "rating": (e.row.get("rating") or "") if e else "",
        "genres": (e.row.get("genres") or "") if e else "",
        "cast": (e.row.get("cast") or "") if e else "",
        "directors": (e.row.get("directors") or "") if e else "",
        "poster": (e.row.get("poster") or "") if e else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect all James Bond films into one CSV.")
    ap.add_argument("--folder", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder holding *_filmography.csv files (default: script's folder)")
    ap.add_argument("--out", default="james_bond_movies.csv", help="output CSV path")
    ap.add_argument("--only-found", action="store_true",
                    help="omit canon films that are absent from the folder")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f"error: no such folder: {args.folder}", file=sys.stderr)
        return 2
    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.folder, args.out)

    entries, files_read = load_entries(args.folder)
    if not entries:
        print(f"error: no *_filmography.csv or .json files found in {args.folder}", file=sys.stderr)
        return 1

    print(f"Read {len(entries)} rows from {len(files_read)} file(s):")
    for name in files_read:
        print(f"  - {name}")

    by_slug, by_title_year = build_index(entries)

    # Extra rows for the same slug living in a second filmography (e.g. a cameo).
    slug_sources: dict[str, list[str]] = {}
    for e in entries:
        slug_sources.setdefault(e.slug, [])
        if e.source_file not in slug_sources[e.slug]:
            slug_sources[e.slug].append(e.source_file)

    matches = [resolve(f, by_slug, by_title_year) for f in BOND_FILMS]
    rows, absent = [], []
    for m in matches:
        if m.found:
            extra = [s for s in slug_sources.get(m.slug, []) if s != m.entry.source_file]
            rows.append(to_row(m, extra))
        else:
            absent.append(m)
            if not args.only_found:
                rows.append(to_row(m, []))

    rows.sort(key=lambda r: (int(r["year"]) if str(r["year"]).isdigit() else 0,
                             r["official_number"] if r["official_number"] != "" else 999))

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)

    found = len(matches) - len(absent)
    eon_found = sum(1 for m in matches if m.found and m.series == EON)
    print(f"\nMatched {found}/{len(BOND_FILMS)} canon films "
          f"({eon_found}/{sum(1 for m in matches if m.series == EON)} of the Eon series)")
    print(f"  Eon series films in CSV : {sum(1 for r in rows if r['series'] == EON)}")
    print(f"  non-Eon films in CSV    : {sum(1 for r in rows if r['series'] == NON_EON)}")
    if absent:
        print("  absent from folder (kept, metadata blank):")
        for m in absent:
            print(f"    - {m.title} ({m.year}) — no Eon Bond actor was in it")

    # Show what a keyword search would have wrongly pulled in.
    near = {(e.slug, e.title, e.year, e.source_file) for e in entries
            if NEAR_MISS_RE.search(e.title) and e.slug not in {f[3] for f in BOND_FILMS}}
    if near:
        print(f"\nIgnored {len(near)} Bond-adjacent non-films (documentaries, TV specials, ...):")
        for _, title, year, src in sorted(near, key=lambda t: t[2] or ""):
            print(f"    - {title} ({year or '?'})  [{src}]")

    print(f"\nWrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
