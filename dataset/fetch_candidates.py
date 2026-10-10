#!/usr/bin/env python
"""Fetch candidate Bond actors from Wikidata.

Pool definition
---------------
Male actors who are citizens of the UK or Ireland, born within
[MIN_BIRTH_YEAR, MAX_BIRTH_YEAR], with at least MIN_SITELINKS Wikipedia language
editions.  Sitelink count is a cheap proxy for prominence -- without it the
query returns thousands of working actors with no public profile.

This is a *nationality* filter, which mirrors the real casting constraint: Bond
is canonically a British character and 5 of the 6 precedent actors are from the
British Isles (Lazenby, Australian, is the lone outlier).

Writes dataset/candidates.csv in the same schema as dataset/actors.csv, with
isBond=0.  Precedent Bond rows live in dataset/actors.csv and are not touched.

Note: actor_id is the Wikidata Q-id (lowercased), which is unique and
traceable.  Precedent rows use name slugs; both are fine, actor_id just needs
to be unique.

Usage
-----
    python fetch_candidates.py --out candidates.csv
    python fetch_candidates.py --min-year 1981 --max-year 1996 --min-sitelinks 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "JamesBondAI/0.1 (research; python-urllib)"

# Citizenship: United Kingdom, England, Scotland, Wales, Northern Ireland, Ireland
CITIZENSHIPS = ["Q145", "Q21", "Q22", "Q25", "Q26", "Q27"]
# When an actor holds several, prefer the most specific homeland over generic UK.
CITIZEN_PRIORITY = ["Q22", "Q25", "Q21", "Q26", "Q27", "Q145"]
MALE = "Q6581097"
ACTOR = "Q33999"
HUMAN = "Q5"

OUT_COLUMNS = [
    "actor_id", "firstName", "secondName", "gender", "birthYear",
    "nationality", "heightCm", "firstBondFilm", "firstBondYear",
    "ageAtCasting", "bondFilms", "isBond", "source",
]

QUERY_TEMPLATE = """
SELECT ?actor ?actorLabel ?birth ?height ?citizenship ?citizenshipLabel ?links WHERE {
  ?actor wdt:P31 wd:%(human)s ;
         wdt:P21 wd:%(male)s ;
         wdt:P106 wd:%(actor)s ;
         wdt:P27 ?citizenship ;
         wdt:P569 ?birth ;
         wikibase:sitelinks ?links .
  VALUES ?citizenship { %(citizenships)s }
  OPTIONAL { ?actor wdt:P2048 ?height . }
  FILTER( ?birth >= "%(min)s-01-01T00:00:00Z"^^xsd:dateTime &&
          ?birth <= "%(max)s-12-31T00:00:00Z"^^xsd:dateTime )
  FILTER( ?links >= %(min_links)d )
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def build_query(min_year: int, max_year: int, min_links: int) -> str:
    return QUERY_TEMPLATE % {
        "human": HUMAN,
        "male": MALE,
        "actor": ACTOR,
        "citizenships": " ".join("wd:" + c for c in CITIZENSHIPS),
        "min": min_year,
        "max": max_year,
        "min_links": min_links,
    }


def run_query(query: str, retries: int = 5) -> list[dict]:
    """POST the query (WDQS prefers POST for long queries) with backoff on 429."""
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    delay = 5.0
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(ENDPOINT, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                print(f"  HTTP {e.code}, retrying in {delay:.0f}s "
                      f"({attempt}/{retries}) ...", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []


def parse_year(iso: str) -> int | None:
    try:
        return int(iso[:4])
    except (ValueError, TypeError):
        return None


def parse_height_cm(node: dict | None) -> int | None:
    """P2048 is a quantity in metres; convert to cm, rejecting implausible units."""
    if not node:
        return None
    try:
        value = float(node["value"])  # e.g. "+1.88"
    except (KeyError, ValueError, TypeError):
        return None
    cm = round(value * 100)
    return cm if 140 <= cm <= 220 else None  # otherwise a unit we don't expect


def split_name(label: str) -> tuple[str, str]:
    parts = label.split()
    if len(parts) <= 1:
        return label, ""
    return parts[0], " ".join(parts[1:])


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def collect(bindings: list[dict]) -> list[dict]:
    """Collapse multi-citizenship rows to one record per actor."""
    by_actor: dict[str, dict] = {}
    for b in bindings:
        aid = qid(b["actor"]["value"])
        rec = by_actor.setdefault(aid, {
            "qid": aid,
            "label": b["actorLabel"]["value"],
            "birth_year": parse_year(b["birth"]["value"]),
            "height_cm": None,
            "citizens": set(),
        })
        rec["citizens"].add(qid(b["citizenship"]["value"]))
        if rec["height_cm"] is None:
            rec["height_cm"] = parse_height_cm(b.get("height"))
    return list(by_actor.values())


def pick_nationality(citizens: set[str], labels: dict[str, str]) -> str:
    for c in CITIZEN_PRIORITY:
        if c in citizens:
            return labels.get(c, "")
    return labels.get(next(iter(citizens)), "")


def to_row(rec: dict, labels: dict[str, str]) -> dict:
    first, second = split_name(rec["label"])
    return {
        "actor_id": rec["qid"].lower(),
        "firstName": first,
        "secondName": second,
        "gender": "male",
        "birthYear": rec["birth_year"],
        "nationality": pick_nationality(rec["citizens"], labels),
        "heightCm": rec["height_cm"] if rec["height_cm"] else "",
        "firstBondFilm": "",
        "firstBondYear": "",
        "ageAtCasting": "",
        "bondFilms": "",
        "isBond": 0,
        "source": f"https://www.wikidata.org/entity/{rec['qid']}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="candidates.csv")
    ap.add_argument("--min-year", type=int, default=1980,
                    help="earliest birth year (1980 -> age 46 in 2026)")
    ap.add_argument("--max-year", type=int, default=1996,
                    help="latest birth year (1996 -> age 30 in 2026)")
    ap.add_argument("--min-sitelinks", type=int, default=5,
                    help="prominence filter; raise to shrink the pool")
    args = ap.parse_args()

    query = build_query(args.min_year, args.max_year, args.min_sitelinks)
    print(f"querying Wikidata: born {args.min_year}-{args.max_year}, "
          f"sitelinks >= {args.min_sitelinks} ...", file=sys.stderr)
    bindings = run_query(query)
    print(f"  -> {len(bindings)} raw rows", file=sys.stderr)

    labels = {qid(b["citizenship"]["value"]): b["citizenshipLabel"]["value"]
              for b in bindings}

    rows = [to_row(r, labels) for r in collect(bindings)]
    rows.sort(key=lambda r: (r["secondName"], r["firstName"]))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    with_height = sum(1 for r in rows if r["heightCm"] != "")
    print(f"wrote {len(rows)} candidates to {args.out} "
          f"({with_height} with height, {len(rows)-with_height} without)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
