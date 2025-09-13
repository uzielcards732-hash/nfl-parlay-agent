
#!/usr/bin/env python3
"""
odds_loader.py — Daily NFL snapshot from The Odds API (free tier friendly)

- Fetches odds for NFL:
  * Moneyline (h2h), spreads, totals
  * Player props (configurable; default set below)
- Computes for each outcome:
  * Best available American odds across books
  * Consensus P_true via median of de‑vigged implied probabilities across books
- Outputs a tidy CSV for the parlay builder:
  id, description, odds, p_true, game_id

Usage:
  python odds_loader.py --api-key YOUR_KEY --out nfl_daily.csv
Optional:
  --region us --books "draftkings,fanduel,betmgm" --markets "h2h,spreads,totals,player_pass_tds,..."
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

# ---------- API fetch ----------

def fetch_market(api_key: str, sport_key: str, market: str, region: str = "us",
                 odds_format: str = "american", books: Optional[List[str]] = None) -> Tuple[List[dict], Dict[str, Optional[str]]]:
    """
    Calls The Odds API v4 for a single market and returns (events_json, rate_limit_headers).
    """
    base = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": market,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    if books:
        params["bookmakers"] = ",".join(books)
    url = base.format(sport=sport_key)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Capture rate-limit headers for logging
    headers = {
        "x-requests-remaining": r.headers.get("x-requests-remaining"),
        "x-requests-used": r.headers.get("x-requests-used"),
    }
    return data, headers

# ---------- Consensus & best price per outcome ----------

def best_price_and_consensus_p_true(outcomes_by_book: List[dict], market: str) -> List[dict]:
    """
    Given a list of *all bookmakers'* outcomes for an event+market, compute per-outcome:
      - best available American odds across books
      - consensus p_true via median of *de‑vigged* implied probabilities across books

    We do approximate devig per bookmaker by pairing outcomes (2-way or 3-way) using their 'name' and 'point'.
    Finally we take the median of those de‑vigged probabilities across books.
    """
    # Build structure: per bookmaker -> outcomes list
    # Input outcomes_by_book are flattened; each item must include 'bookmaker' name.
    by_book: Dict[str, List[dict]] = {}
    for oc in outcomes_by_book:
        bm = oc.get("_bookmaker", "unknown")
        by_book.setdefault(bm, []).append(oc)

    # Aggregate best odds and collect book-level p_true estimates
    # Key outcomes by (name, point, participant) — for player props, 'description' may include player
    # but standardized fields are typically name (e.g., "Over") and participant (player/team)
    # We'll carry 'participant' if present.
    best_price: Dict[Tuple[str, Optional[float], Optional[str]], int] = {}
    ptrue_samples: Dict[Tuple[str, Optional[float], Optional[str]], List[float]] = {}

    for bm, ocs in by_book.items():
        # group by point and participant to identify two-way or three-way within this bookmaker
        # For spreads/totals, participant may be None; for props, participant is the player name.
        per_point: Dict[Tuple[Optional[float], Optional[str]], List[dict]] = {}
        for oc in ocs:
            key = (oc.get("point"), oc.get("participant"))
            per_point.setdefault(key, []).append(oc)

        for (point, participant), group in per_point.items():
            # update best prices
            for oc in group:
                key_out = (oc.get("name"), point, participant)
                price = int(oc["price"])
                if key_out not in best_price:
                    best_price[key_out] = price
                else:
                    # choose bettor-favorable price (less negative or more positive)
                    if price > best_price[key_out]:
                        best_price[key_out] = price

            # compute devigged probs for this bookmaker slice
            names = list({g.get("name") for g in group})
            # map name -> list of prices (though usually one per name per book)
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
                # Unexpected: fall back to raw implied prob median (no devig)
                for n in names:
                    pv = median([american_to_implied_prob(p) for p in prices_by_name[n]])
                    ptrue_samples.setdefault((n, point, participant), []).append(pv)

    # finalize median p_true and best odds per outcome
    out = []
    for key, best_odds in best_price.items():
        name, point, participant = key
        p_list = ptrue_samples.get(key, [])
        p_true = median(p_list) if p_list else float("nan")
        out.append({
            "name": name,
            "point": point,
            "participant": participant,
            "best_odds": int(best_odds),
            "p_true": float(p_true),
        })
    return out

# ---------- Parse one event into parlay legs ----------

def parse_event_to_legs(event: dict, market: str) -> List[dict]:
    """
    Consolidate all bookmakers' outcomes for a single event+market,
    derive best odds + consensus p_true per outcome, and convert to leg rows.
    """
    game_id = f"{event.get('away_team','?')}_@_{event.get('home_team','?')}"
    # Flatten outcomes across all bookmakers for this event/market
    flat_outcomes = []
    for bm in event.get("bookmakers", []):
        bm_name = bm.get("title") or bm.get("key") or "unknown"
        for mk in bm.get("markets", []):
            if mk.get("key") != market:
                continue
            for oc in mk.get("outcomes", []):
                oc_copy = dict(oc)  # shallow copy
                oc_copy["_bookmaker"] = bm_name
                # normalize possible fields
                if "participant" not in oc_copy:
                    # For team markets, participant often absent; for props it's the player name
                    oc_copy["participant"] = oc_copy.get("description") or None
                flat_outcomes.append(oc_copy)

    oc_summary = best_price_and_consensus_p_true(flat_outcomes, market)
    legs = []

    def fmt_odds(o: int) -> str:
        return f"{o:+d}"

    for item in oc_summary:
        name = item["name"]               # e.g., "Over", "Under", or team name
        point = item["point"]             # number for spreads/totals/props
        participant = item["participant"] # player name for props (if present)
        odds = int(item["best_odds"])
        p_true = float(item["p_true"])
        if math.isnan(p_true):
            continue

        # Construct id/description based on market
        if market == "h2h":
            leg_id = f"{name}_ML@{game_id}"
            desc = f"{name} ML ({fmt_odds(odds)})"
        elif market == "spreads":
            leg_id = f"{name}_{point:+g}@{game_id}"
            desc = f"{name} {point:+g} ({fmt_odds(odds)})"
        elif market == "totals":
            ou = "O" if name.lower().startswith("over") else "U"
            leg_id = f"{ou}{point:g}@{game_id}"
            desc = f"Game {ou}{point:g} ({fmt_odds(odds)})"
        else:
            # treat as player prop
            label = market.replace("player_", "")
            side = name  # typically "Over"/"Under"
            player = participant or "Player"
            leg_id = f"{player}_{label}_{point:g}_{side}@{game_id}"
            desc = f"{player} {label.replace('_',' ')} {side} {point:g} ({fmt_odds(odds)})"

        legs.append({
            "id": leg_id,
            "description": desc,
            "odds": odds,
            "p_true": round(p_true, 4),
            "game_id": game_id
        })
    return legs

# ---------- Snapshot orchestration ----------

DEFAULT_MARKETS = [
    "h2h",
    "spreads",
    "totals",
    # Player props (edit this list as you like)
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_anytime_td"
]

def snapshot_daily(api_key: str, region: str = "us", books: Optional[List[str]] = None,
                   markets: Optional[List[str]] = None) -> Tuple[List[dict], Dict[str, Optional[str]]]:
    markets = markets or DEFAULT_MARKETS
    all_legs: List[dict] = []
    last_headers: Dict[str, Optional[str]] = {}
    for m in markets:
        events, headers = fetch_market(api_key, NFL_SPORT_KEY, m, region=region, books=books)
        last_headers = headers or last_headers
        for ev in events:
            legs = parse_event_to_legs(ev, m)
            all_legs.extend(legs)
    return all_legs, last_headers

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Daily NFL odds snapshot -> legs CSV")
    parser.add_argument("--api-key", required=True, help="The Odds API key")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--region", default="us", help="Region code (default: us)")
    parser.add_argument("--books", default="", help="Comma-separated bookmaker keys (optional)")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                        help="Comma-separated markets to fetch (default includes h2h, spreads, totals, 5 props)")
    args = parser.parse_args()

    books = [b.strip() for b in args.books.split(",") if b.strip()] or None
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    print(f"[{dt.datetime.utcnow().isoformat()}Z] Fetching markets: {markets} (region={args.region})")
    legs, headers = snapshot_daily(api_key=args.api_key, region=args.region, books=books, markets=markets)
    print(f"Got {len(legs)} legs. Rate-limit headers: {json.dumps(headers)}")

    # Write CSV in parlay_builder format
    fieldnames = ["id", "description", "odds", "p_true", "game_id"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in legs:
            w.writerow({
                "id": row["id"],
                "description": row["description"],
                "odds": row["odds"],
                "p_true": row["p_true"],
                "game_id": row.get("game_id", "")
            })
    print(f"Wrote {len(legs)} rows to {args.out}")

if __name__ == "__main__":
    main()
