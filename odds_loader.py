#!/usr/bin/env python3
"""
odds_loader.py — Daily NFL snapshot from The Odds API (props-compatible)

- Standard markets (h2h, spreads, totals) fetched with /v4/sports/:sport/odds
- Player props fetched PER-EVENT using /v4/sports/:sport/events/:eventId/odds
  (this avoids HTTP 422 errors)

Outputs a CSV compatible with parlay_builder:
  id, description, odds, p_true, game_id

Usage:
  python odds_loader.py --api-key YOUR_KEY --out nfl_daily.csv
"""

import argparse
import csv
import datetime as dt
import json
import math
from typing import Dict, List, Optional, Tuple
import requests

NFL_SPORT_KEY = "americanfootball_nfl"

# ---------- Odds utilities ----------

def american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    return (-odds) / ((-odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)

def devig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5
    return p1 / s, p2 / s

def devig_three_way(p1: float, p2: float, p3: float) -> Tuple[float, float, float]:
    s = p1 + p2 + p3
    if s <= 0:
        return (1/3, 1/3, 1/3)
    return p1/s, p2/s, p3/s

def median(xs: List[float]) -> float:
    ys = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not ys:
        return float("nan")
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])

# ---------- API fetch helpers ----------

def fetch_sports_odds(api_key: str, sport_key: str, markets: List[str], region: str = "us",
                      odds_format: str = "american", books: Optional[List[str]] = None) -> Tuple[List[dict], Dict[str, Optional[str]]]:
    """Fetch standard markets (h2h, spreads, totals)."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if books:
        params["bookmakers"] = ",".join(books)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    headers = {
        "x-requests-remaining": r.headers.get("x-requests-remaining"),
        "x-requests-used": r.headers.get("x-requests-used"),
    }
    return data, headers

def fetch_events(api_key: str, sport_key: str) -> List[dict]:
    """Get list of upcoming NFL events (ids, teams)."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    r = requests.get(url, params={"apiKey": api_key}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_event_props(api_key: str, sport_key: str, event_id: str, prop_markets: List[str],
                      region: str = "us", odds_format: str = "american", books: Optional[List[str]] = None) -> Tuple[dict, Dict[str, Optional[str]]]:
    """Fetch props for one event."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": ",".join(prop_markets),
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if books:
        params["bookmakers"] = ",".join(books)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    headers = {
        "x-requests-remaining": r.headers.get("x-requests-remaining"),
        "x-requests-used": r.headers.get("x-requests-used"),
    }
    return r.json(), headers

# ---------- Consensus & best price per outcome ----------

def best_price_and_consensus_p_true(outcomes_by_book: List[dict]) -> List[dict]:
    by_book: Dict[str, List[dict]] = {}
    for oc in outcomes_by_book:
        bm = oc.get("_bookmaker", "unknown")
        by_book.setdefault(bm, []).append(oc)

    best_price: Dict[Tuple[str, Optional[float], Optional[str]], int] = {}
    ptrue_samples: Dict[Tuple[str, Optional[float], Optional[str]], List[float]] = {}

    for bm, ocs in by_book.items():
        per_point: Dict[Tuple[Optional[float], Optional[str]], List[dict]] = {}
        for oc in ocs:
            key = (oc.get("point"), oc.get("participant"))
            per_point.setdefault(key, []).append(oc)

        for (point, participant), group in per_point.items():
            for oc in group:
                key_out = (oc.get("name"), point, participant)
                price = int(oc["price"])
                best_price[key_out] = price if key_out not in best_price else max(price, best_price[key_out])

            names = list({g.get("name") for g in group})
            prices_by_name = {n: [int(g["price"]) for g in group if g.get("name") == n] for n in names}
            if len(names) == 2:
                p1 = median([american_to_implied_prob(p) for p in prices_by_name[names[0]]])
                p2 = median([american_to_implied_prob(p) for p in prices_by_name[names[1]]])
                d1, d2 = devig_two_way(p1, p2)
                for n, d in zip([names[0], names[1]], [d1, d2]):
                    ptrue_samples.setdefault((n, point, participant), []).append(d)
            elif len(names) == 3:
                ps = [median([american_to_implied_prob(p) for p in prices_by_name[n]]) for n in names]
                d = devig_three_way(*ps)
                for n, dv in zip(names, d):
                    ptrue_samples.setdefault((n, point, participant), []).append(dv)
            else:
                for n in names:
                    pv = median([american_to_implied_prob(p) for p in prices_by_name[n]])
