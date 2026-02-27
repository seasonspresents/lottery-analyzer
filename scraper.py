"""
Multi-State Lottery Scratcher Scraper
Scrapes data from lottery.net (summary) and official state sites (detailed prize tiers).
"""
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# On Vercel (read-only filesystem), use /tmp for data storage
if os.environ.get("VERCEL"):
    DATA_DIR = Path("/tmp/data")
else:
    DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

# ---------------------------------------------------------------------------
# State configuration — add new states here
# ---------------------------------------------------------------------------
STATE_CONFIG = {
    "az": {
        "name": "Arizona",
        "abbreviation": "AZ",
        "lottery_net_url": "https://www.lottery.net/arizona/scratchers",
        "official_base_url": "https://www.arizonalottery.com",
        "scratchers_url": "https://www.arizonalottery.com/scratchers/",
        "game_url_pattern": r"/scratchers/(\d+)-",
        "scratchcards_net_base": "https://scratchcards.net/arizona",
        "second_chance_url": "azplayersclub.com",
        "second_chance_text": "Arizona Lottery has <strong style=\"color:#374151;\">Second Chance drawings</strong>. Enter losing tickets at <strong style=\"color:var(--pop-purple);\">azplayersclub.com</strong> for chances to win cash, trips, and cars.",
        "map_center": [33.4484, -112.0740],
        "map_zoom": 11,
        "geocode_suffix": "Arizona",
        "mascot_name": "Spike the Scratcher Cactus",
        "mascot_emoji": "&#127797;",
        "summary_columns": 6,
        "has_retailer_api": True,
        "retailer_api_url": "https://api.arizonalottery.com/v2/retailers",
        "data_prefix": "az",
        "helpline": "1-800-NEXT-STEP",
    },
    "ca": {
        "name": "California",
        "abbreviation": "CA",
        "lottery_net_url": "https://www.lottery.net/california/scratchers",
        "official_base_url": "https://www.calottery.com",
        "scratchers_url": "https://www.calottery.com/scratchers",
        "game_url_pattern": r"/scratchers/(\d+)-",
        "scratchcards_net_base": "https://scratchcards.net/california",
        "second_chance_url": "calottery.com/second-chance",
        "second_chance_text": "California Lottery has <strong style=\"color:#374151;\">2nd Chance</strong> drawings. Enter losing tickets at <strong style=\"color:var(--pop-purple);\">calottery.com/second-chance</strong> for bonus chances to win.",
        "map_center": [34.0522, -118.2437],
        "map_zoom": 10,
        "geocode_suffix": "California",
        "mascot_name": "Goldie the Golden Bear",
        "mascot_emoji": "&#129528;",
        "summary_columns": 7,
        "has_retailer_api": False,
        "retailer_api_url": None,
        "data_prefix": "ca",
        "helpline": "1-800-GAMBLER",
    },
}

DEFAULT_STATE = "az"

# Backward-compatible module-level constants (point to AZ defaults)
SUMMARY_FILE = DATA_DIR / "az_scratchers_summary.json"
DETAIL_FILE = DATA_DIR / "az_scratchers_detail.json"
LOTTERY_NET_URL = STATE_CONFIG["az"]["lottery_net_url"]
AZ_LOTTERY_BASE = STATE_CONFIG["az"]["official_base_url"]
AZ_SCRATCHERS_URL = STATE_CONFIG["az"]["scratchers_url"]


def get_data_paths(state_code="az"):
    """Return (summary_file, detail_file, analyzed_file) for a given state."""
    prefix = STATE_CONFIG[state_code]["data_prefix"]
    return (
        DATA_DIR / f"{prefix}_scratchers_summary.json",
        DATA_DIR / f"{prefix}_scratchers_detail.json",
        DATA_DIR / f"{prefix}_scratchers_analyzed.json",
    )


def scrape_summary(state_code="az"):
    """Scrape summary data (name, price, top prize, prizes remaining, odds) from lottery.net."""
    cfg = STATE_CONFIG[state_code]
    url = cfg["lottery_net_url"]
    num_cols = cfg["summary_columns"]

    print(f"Scraping {cfg['name']} summary data from lottery.net ...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    games = []
    table = soup.find("table")
    if not table:
        print("WARNING: Could not find scratchers table on lottery.net")
        return games

    rows = table.find_all("tr")
    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < num_cols:
            continue
        try:
            name = cells[0].get_text(strip=True)
            game_num = cells[1].get_text(strip=True)
            price_text = cells[2].get_text(strip=True).replace("$", "").replace(",", "")
            top_prize_text = cells[3].get_text(strip=True)

            # Column layout differs by state
            if num_cols == 7:
                # CA: 0=name, 1=num, 2=price, 3=top_prize, 4=top_prizes_available, 5=remaining, 6=odds
                top_prizes_available_text = cells[4].get_text(strip=True).replace(",", "")
                prizes_remaining = cells[5].get_text(strip=True).replace(",", "")
                odds_text = cells[6].get_text(strip=True)
            else:
                # AZ (6 cols): 0=name, 1=num, 2=price, 3=top_prize, 4=remaining, 5=odds
                top_prizes_available_text = None
                prizes_remaining = cells[4].get_text(strip=True).replace(",", "")
                odds_text = cells[5].get_text(strip=True)

            # Parse price
            price = float(price_text) if price_text else 0

            # Parse top prize
            top_prize_clean = top_prize_text.replace("$", "").replace(",", "")
            if "million" in top_prize_clean.lower():
                top_prize = float(re.sub(r"[^0-9.]", "", top_prize_clean)) * 1_000_000
            else:
                top_prize = float(re.sub(r"[^0-9.]", "", top_prize_clean)) if top_prize_clean else 0

            # Parse odds (e.g., "1 in 3.44")
            odds_match = re.search(r"1\s+in\s+([\d.]+)", odds_text)
            odds = float(odds_match.group(1)) if odds_match else 0

            game = {
                "name": name,
                "game_num": game_num,
                "price": price,
                "top_prize": top_prize,
                "top_prizes_remaining": int(prizes_remaining) if prizes_remaining.isdigit() else 0,
                "odds_1_in": odds,
                "win_probability": round(1 / odds, 4) if odds > 0 else 0,
            }

            # CA bonus field
            if top_prizes_available_text is not None:
                game["top_prizes_available"] = int(top_prizes_available_text) if top_prizes_available_text.isdigit() else 0

            games.append(game)
        except (ValueError, IndexError) as e:
            print(f"  Skipping row: {e}")
            continue

    print(f"  Found {len(games)} games")
    return games


def scrape_game_urls(state_code="az"):
    """Get individual game page URLs from the official state lottery site."""
    cfg = STATE_CONFIG[state_code]
    base_url = cfg["official_base_url"]
    scratchers_url = cfg["scratchers_url"]
    pattern = cfg["game_url_pattern"]

    print(f"Fetching game URLs from {base_url} ...")
    resp = requests.get(scratchers_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = {}
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.match(pattern, href)
        if match:
            game_num = match.group(1)
            full_url = href if href.startswith("http") else base_url + href
            urls[game_num] = full_url
    print(f"  Found {len(urls)} game URLs")
    return urls


def scrape_detail_with_playwright(game_urls, max_games=None):
    """Use Playwright to scrape detailed prize tier data from each game page."""
    from playwright.sync_api import sync_playwright

    details = {}
    urls_to_scrape = list(game_urls.items())
    if max_games:
        urls_to_scrape = urls_to_scrape[:max_games]

    print(f"Scraping detailed prize data for {len(urls_to_scrape)} games (this takes a minute) ...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, (game_num, url) in enumerate(urls_to_scrape):
            try:
                print(f"  [{i+1}/{len(urls_to_scrape)}] Scraping game {game_num} ...")
                page.goto(url, wait_until="networkidle", timeout=20000)
                # Wait for prize table to render
                page.wait_for_timeout(2000)

                # Try to find the prizes table
                prize_rows = page.query_selector_all("table tr, .prize-row, [class*='prize']")

                # Extract game info from the page
                content = page.content()
                soup = BeautifulSoup(content, "html.parser")

                game_detail = {
                    "game_num": game_num,
                    "url": url,
                    "prize_tiers": [],
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }

                # Look for prize tables
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        texts = [c.get_text(strip=True) for c in cells]
                        # Look for rows with prize data (dollar amount, odds, remaining count)
                        if len(texts) >= 2 and "$" in texts[0]:
                            tier = {"raw": texts}
                            # Parse prize amount
                            prize_text = texts[0].replace("$", "").replace(",", "")
                            try:
                                tier["prize"] = float(prize_text)
                            except ValueError:
                                tier["prize"] = 0
                            if len(texts) >= 2:
                                tier["odds_text"] = texts[1]
                            if len(texts) >= 3:
                                tier["remaining_text"] = texts[2]
                                try:
                                    tier["remaining"] = int(texts[2].replace(",", ""))
                                except ValueError:
                                    tier["remaining"] = None
                            game_detail["prize_tiers"].append(tier)

                details[game_num] = game_detail
                time.sleep(0.5)  # Be polite

            except Exception as e:
                print(f"  ERROR scraping game {game_num}: {e}")
                details[game_num] = {"game_num": game_num, "url": url, "error": str(e), "prize_tiers": []}

        browser.close()

    return details


# ---------------------------------------------------------------------------
# Prize tier scraping from scratchcards.net
# ---------------------------------------------------------------------------

def _parse_money(text):
    """Parse a dollar string like '$1,000' or '$5.00' into a float."""
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _parse_int(text):
    """Parse an integer string like '1,234' into an int."""
    cleaned = re.sub(r"[^\d]", "", text.replace(",", ""))
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return 0


def _find_prize_table(soup):
    """Find the prize tier table in a scratchcards.net game page.

    Returns a list of dicts: [{prize, total, claimed, remaining}, ...]
    """
    tiers = []
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        # Check header for prize-related keywords
        header_text = rows[0].get_text(strip=True).lower()
        if "prize" not in header_text and "remaining" not in header_text:
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) < 3:
                continue
            # Look for rows that start with a dollar amount
            if not texts[0].startswith("$") and "$" not in texts[0]:
                continue
            try:
                prize = _parse_money(texts[0])
                if prize <= 0:
                    continue
                # Columns vary but common patterns:
                # [Prize, Total, Claimed, Remaining] or [Prize, Total, Remaining]
                if len(texts) >= 4:
                    total = _parse_int(texts[1])
                    claimed = _parse_int(texts[2])
                    remaining = _parse_int(texts[3])
                else:
                    total = _parse_int(texts[1])
                    remaining = _parse_int(texts[2])
                    claimed = total - remaining
                tiers.append({
                    "prize": prize,
                    "total": total,
                    "claimed": claimed,
                    "remaining": remaining,
                })
            except (ValueError, IndexError):
                continue
        if tiers:
            break  # Found the right table
    return tiers


def scrape_prize_tiers(state_code="az", game_nums=None):
    """Scrape full prize tier data from scratchcards.net for each game.

    Returns {game_num: {prize_tiers: [...], total_remaining_prizes: N}}
    """
    cfg = STATE_CONFIG[state_code]
    base = cfg.get("scratchcards_net_base")
    if not base:
        print(f"  No scratchcards.net base URL for {cfg['name']} — skipping tier scrape")
        return {}

    if game_nums is None:
        game_nums = []

    print(f"Scraping prize tiers from scratchcards.net for {len(game_nums)} {cfg['name']} games ...")
    results = {}

    for i, gnum in enumerate(game_nums):
        url = f"{base}/{gnum}"
        try:
            if i > 0:
                time.sleep(0.3)
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                print(f"  [{i+1}/{len(game_nums)}] Game {gnum}: 404 — skipped")
                continue
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            tiers = _find_prize_table(soup)
            if tiers:
                total_remaining = sum(t["remaining"] for t in tiers)
                results[gnum] = {
                    "prize_tiers": tiers,
                    "total_remaining_prizes": total_remaining,
                }
                print(f"  [{i+1}/{len(game_nums)}] Game {gnum}: {len(tiers)} tiers, {total_remaining} prizes remaining")
            else:
                print(f"  [{i+1}/{len(game_nums)}] Game {gnum}: no prize table found")
        except Exception as e:
            print(f"  [{i+1}/{len(game_nums)}] Game {gnum}: ERROR — {e}")

    print(f"  Got tier data for {len(results)}/{len(game_nums)} games")
    return results


def analyze_games(summary_games, detail_data=None, tier_data=None):
    """Calculate analysis metrics for each game."""
    analyzed = []

    for game in summary_games:
        g = dict(game)
        price = g["price"]
        odds = g["odds_1_in"]
        top_prize = g["top_prize"]

        # Prize-to-price ratio (how many ticket costs the top prize covers)
        g["top_prize_to_price_ratio"] = round(top_prize / price, 1) if price > 0 else 0

        if odds > 0:
            g["win_rate_pct"] = round((1 / odds) * 100, 2)
        else:
            g["win_rate_pct"] = 0

        # --- EV from prize tier data (scratchcards.net) ---
        game_num = g["game_num"]
        g["prize_tiers"] = None
        g["total_remaining_prizes"] = None
        g["est_remaining_tickets"] = None
        g["ev_per_ticket"] = None
        g["ev_per_dollar"] = None
        g["ev_net"] = None
        g["payout_pct"] = None
        g["positive_ev"] = False

        if tier_data and game_num in tier_data:
            td = tier_data[game_num]
            tiers = td.get("prize_tiers", [])
            total_remaining = td.get("total_remaining_prizes", 0)
            if tiers and total_remaining > 0 and g["win_probability"] > 0:
                g["prize_tiers"] = tiers
                g["total_remaining_prizes"] = total_remaining
                # Estimate remaining tickets from total remaining prizes / win probability
                est_remaining_tickets = total_remaining / g["win_probability"]
                g["est_remaining_tickets"] = round(est_remaining_tickets)
                # Total remaining value across all tiers
                total_remaining_value = sum(t["prize"] * t["remaining"] for t in tiers)
                # EV per ticket
                ev_per_ticket = total_remaining_value / est_remaining_tickets if est_remaining_tickets > 0 else 0
                g["ev_per_ticket"] = round(ev_per_ticket, 2)
                g["ev_per_dollar"] = round(ev_per_ticket / price, 4) if price > 0 else 0
                g["ev_net"] = round(ev_per_ticket - price, 2)
                g["payout_pct"] = round((ev_per_ticket / price) * 100, 1) if price > 0 else 0
                g["positive_ev"] = ev_per_ticket > price

        # Fall back to old detail_data if no tier data
        if g["ev_per_dollar"] is None and detail_data and game_num in detail_data:
            detail = detail_data[game_num]
            tiers = detail.get("prize_tiers", [])
            if tiers:
                g["prize_tiers"] = tiers
                total_ev = 0
                has_ev = False
                for tier in tiers:
                    if tier.get("prize") and tier.get("remaining") is not None and tier.get("odds_text"):
                        try:
                            odds_val = float(tier["odds_text"].replace(",", "").replace(" ", ""))
                            if odds_val > 0:
                                total_ev += tier["prize"] / odds_val
                                has_ev = True
                        except (ValueError, ZeroDivisionError):
                            pass
                if has_ev:
                    g["ev_per_ticket"] = round(total_ev, 2)
                    g["ev_per_dollar"] = round(total_ev / price, 4) if price > 0 else 0

        # --- Composite score: 45/45/10 + EV bonus ---
        score = 0
        if odds > 0:
            score += (1 / odds) * 22.5           # Win probability component
            score += min(g["top_prizes_remaining"], 10) * 2.25  # Prizes remaining
            score += min(g["top_prize_to_price_ratio"] / 10000, 5)  # Prize ratio
            if g["ev_per_dollar"] is not None:
                score += min(g["ev_per_dollar"] * 6, 5)  # EV bonus (0-5 range)
        g["score"] = round(score, 2)

        analyzed.append(g)

    # Sort by score descending
    analyzed.sort(key=lambda x: x["score"], reverse=True)

    # Add rank
    for i, g in enumerate(analyzed):
        g["rank"] = i + 1

    # --- Smart Pick: "safety score" balancing win rate + prizes remaining ---
    # Group by price to normalize within price tiers
    price_groups = {}
    for g in analyzed:
        p = int(g["price"])
        price_groups.setdefault(p, []).append(g)

    # Compute normalized safety score per price group
    for p, group in price_groups.items():
        max_win = max(g["win_rate_pct"] for g in group) or 1
        max_remaining = max(g["top_prizes_remaining"] for g in group) or 1
        # For payout normalization, find max payout_pct in group
        payout_vals = [g["payout_pct"] for g in group if g.get("payout_pct") is not None]
        max_payout = max(payout_vals) if payout_vals else 0
        has_ev_data = max_payout > 0

        for g in group:
            # Normalize win rate within price tier (0-1)
            norm_win = g["win_rate_pct"] / max_win
            # Normalize remaining within price tier (0-1)
            norm_remaining = g["top_prizes_remaining"] / max_remaining
            # Penalty: games with only 1 top prize left are risky
            remaining_penalty = 0
            if g["top_prizes_remaining"] <= 1:
                remaining_penalty = 0.3
            elif g["top_prizes_remaining"] <= 2:
                remaining_penalty = 0.1

            if has_ev_data and g.get("payout_pct") is not None:
                # EV-aware safety: 40% win, 35% remaining, 15% payout, 10% base
                norm_payout = g["payout_pct"] / max_payout if max_payout > 0 else 0
                safety = (norm_win * 0.40) + (norm_remaining * 0.35) + (norm_payout * 0.15) + 0.10 - remaining_penalty
            else:
                # Fallback: original blend
                safety = (norm_win * 0.50) + (norm_remaining * 0.40) + 0.10 - remaining_penalty
            g["safety_score"] = round(max(safety, 0) * 100, 1)

    # --- Buy recommendation: how many tickets for ~80% win chance ---
    for g in analyzed:
        odds = g["odds_1_in"]
        price = g["price"]
        if odds > 0:
            p_win = 1 / odds  # single ticket win probability

            # Calculate tickets needed for target probabilities
            targets = {}
            for target_pct in [50, 65, 80, 90, 95]:
                target = target_pct / 100
                n = math.ceil(math.log(1 - target) / math.log(1 - p_win))
                targets[target_pct] = n

            # Sweet spot: 80% chance of winning at least 1 prize
            sweet_spot = targets[80]

            # Budget cap: don't recommend spending more than these limits
            budget_caps = {1: 15, 2: 15, 3: 15, 5: 50, 10: 50, 20: 60, 30: 60, 50: 100}
            max_spend = budget_caps.get(int(price), 50)
            max_tickets = max(1, int(max_spend / price))

            recommended = min(sweet_spot, max_tickets)

            g["buy_recommended"] = recommended
            g["buy_cost"] = round(recommended * price, 2)
            g["buy_win_chance"] = round((1 - (1 - p_win) ** recommended) * 100, 1)
            g["buy_targets"] = {
                str(k): {"tickets": v, "cost": round(v * price, 2)}
                for k, v in targets.items()
            }

            # Build the probability ladder for display (1 through recommended + a couple more)
            ladder = []
            for n in range(1, min(recommended + 3, 16)):
                chance = round((1 - (1 - p_win) ** n) * 100, 1)
                ladder.append({"n": n, "cost": round(n * price, 2), "chance": chance})
            g["buy_ladder"] = ladder
        else:
            g["buy_recommended"] = 1
            g["buy_cost"] = price
            g["buy_win_chance"] = 0
            g["buy_targets"] = {}
            g["buy_ladder"] = []

    # Generate plain-English explanations for top picks
    all_sorted_safety = sorted(analyzed, key=lambda g: g.get("safety_score", 0), reverse=True)
    for i, g in enumerate(all_sorted_safety):
        g["safety_rank"] = i + 1
        g["smart_pick_reason"] = _generate_explanation(g, price_groups.get(int(g["price"]), []))

    return analyzed


def _generate_explanation(game, peers):
    """Generate a 5th-grader-friendly explanation for why this game is a good/bad pick."""
    win_pct = game["win_rate_pct"]
    remaining = game["top_prizes_remaining"]
    price = int(game["price"])
    top_prize = game["top_prize"]
    rec = game.get("buy_recommended", 1)
    rec_cost = game.get("buy_cost", price)
    rec_chance = game.get("buy_win_chance", 0)

    # Compare to peers at same price
    avg_win = sum(g["win_rate_pct"] for g in peers) / len(peers) if peers else 0

    parts = []

    # Win rate context
    if win_pct >= avg_win * 1.05:
        parts.append(f"Out of every 100 tickets, roughly {int(win_pct)} are winners — that's better than most other ${price} games")
    elif win_pct >= avg_win * 0.95:
        parts.append(f"About {int(win_pct)} out of every 100 tickets win something — that's average for a ${price} game")
    else:
        parts.append(f"Only about {int(win_pct)} out of 100 tickets win — that's below average for ${price} games")

    # Remaining context
    if remaining >= 5:
        parts.append(f"There are still {remaining} top prizes up for grabs, so the game hasn't been picked over yet")
    elif remaining >= 2:
        parts.append(f"There are {remaining} top prizes left — not a ton, but still in play")
    else:
        parts.append(f"Only {remaining} top prize left — most of the big wins are already claimed")

    # Top prize context
    if top_prize >= 100000:
        parts.append(f"The top prize is ${top_prize:,.0f}, which is a solid payout for a ${price} ticket")

    # Payout / EV context
    payout = game.get("payout_pct")
    if payout is not None:
        if payout >= 80:
            parts.append(f"The estimated return is {payout:.0f}% — that's unusually strong for a scratcher")
        elif payout >= 65:
            parts.append(f"The estimated return is {payout:.0f}%, which is solid for this price point")
        elif payout >= 50:
            parts.append(f"The estimated return is {payout:.0f}% — typical for scratchers")

    # Buy recommendation
    parts.append(
        f"I'd recommend buying {rec} tickets from the same roll "
        f"(${rec_cost:,.0f} total) — that gives you about a {rec_chance:.0f}% chance "
        f"of winning at least one prize. Ask the clerk for {rec} in a row from the same roll"
    )

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Velocity snapshots — track prize claim speed between runs
# ---------------------------------------------------------------------------

def get_history_path(state_code):
    """Return the path for a state's history snapshot file."""
    prefix = STATE_CONFIG[state_code]["data_prefix"]
    return DATA_DIR / f"{prefix}_history.json"


def save_snapshot(state_code, analyzed_games):
    """Save a snapshot of current remaining counts per game. Max 30 snapshots."""
    path = get_history_path(state_code)
    history = []
    if path.exists():
        try:
            with open(path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "games": {},
    }
    for g in analyzed_games:
        snapshot["games"][g["game_num"]] = {
            "top_prizes_remaining": g["top_prizes_remaining"],
            "total_remaining_prizes": g.get("total_remaining_prizes"),
        }

    history.append(snapshot)
    # Keep last 30 snapshots
    if len(history) > 30:
        history = history[-30:]

    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def compute_velocity(state_code, current_games):
    """Compare current remaining counts to previous snapshot and set velocity fields."""
    path = get_history_path(state_code)
    if not path.exists():
        # First run: set velocity to 0
        for g in current_games:
            g["top_prize_velocity"] = 0
            g["total_prize_velocity"] = 0
        return

    try:
        with open(path) as f:
            history = json.load(f)
    except (json.JSONDecodeError, IOError):
        for g in current_games:
            g["top_prize_velocity"] = 0
            g["total_prize_velocity"] = 0
        return

    if not history:
        for g in current_games:
            g["top_prize_velocity"] = 0
            g["total_prize_velocity"] = 0
        return

    prev = history[-1]["games"]  # Most recent snapshot
    for g in current_games:
        gnum = g["game_num"]
        if gnum in prev:
            prev_top = prev[gnum].get("top_prizes_remaining", 0)
            prev_total = prev[gnum].get("total_remaining_prizes")
            g["top_prize_velocity"] = max(0, prev_top - g["top_prizes_remaining"])
            if prev_total is not None and g.get("total_remaining_prizes") is not None:
                g["total_prize_velocity"] = max(0, prev_total - g["total_remaining_prizes"])
            else:
                g["total_prize_velocity"] = 0
        else:
            g["top_prize_velocity"] = 0
            g["total_prize_velocity"] = 0


def run_full_scrape(skip_detail=False, skip_tiers=False, state_code="az"):
    """Run a complete scrape and save results."""
    cfg = STATE_CONFIG[state_code]
    summary_file, detail_file, analyzed_file = get_data_paths(state_code)

    # Summary data
    summary = scrape_summary(state_code)
    summary_file.parent.mkdir(exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "games": summary,
        }, f, indent=2)
    print(f"Saved summary to {summary_file}")

    # Prize tier data from scratchcards.net (replaces Playwright detail scrape)
    tier_data = None
    if not skip_tiers:
        try:
            game_nums = [g["game_num"] for g in summary]
            tier_data = scrape_prize_tiers(state_code, game_nums)
        except Exception as e:
            print(f"Tier scrape failed: {e}")

    # Detailed data (optional, requires Playwright — kept as fallback)
    detail_data = None
    if not skip_detail and not tier_data:
        try:
            game_urls = scrape_game_urls(state_code)
            detail_data = scrape_detail_with_playwright(game_urls)
            with open(detail_file, "w") as f:
                json.dump({
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "games": detail_data,
                }, f, indent=2)
            print(f"Saved detail to {detail_file}")
        except ImportError:
            print("Playwright not installed -- skipping detailed scrape (run: pip install playwright && playwright install chromium)")
        except Exception as e:
            print(f"Detail scrape failed: {e}")

    # Analyze
    analyzed = analyze_games(summary, detail_data, tier_data)

    # Velocity: compute then snapshot
    compute_velocity(state_code, analyzed)
    save_snapshot(state_code, analyzed)

    with open(analyzed_file, "w") as f:
        json.dump({
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "games": analyzed,
        }, f, indent=2)
    print(f"Saved analysis to {analyzed_file}")
    return analyzed


if __name__ == "__main__":
    import sys
    skip = "--skip-detail" in sys.argv
    skip_tiers = "--skip-tiers" in sys.argv

    # Parse --state=xx argument
    state_code = DEFAULT_STATE
    for arg in sys.argv[1:]:
        if arg.startswith("--state="):
            state_code = arg.split("=", 1)[1].lower()

    if state_code not in STATE_CONFIG:
        print(f"Unknown state: {state_code}. Available: {', '.join(STATE_CONFIG.keys())}")
        sys.exit(1)

    cfg = STATE_CONFIG[state_code]
    results = run_full_scrape(skip_detail=skip, skip_tiers=skip_tiers, state_code=state_code)
    print(f"\n{'='*60}")
    print(f"TOP 10 {cfg['name'].upper()} SCRATCHERS BY SCORE")
    print(f"{'='*60}")
    for g in results[:10]:
        print(f"  #{g['rank']:2d}  {g['name']:<35s} ${g['price']:<6.0f} "
              f"Win: {g['win_rate_pct']:5.1f}%  Top Prize: ${g['top_prize']:>12,.0f}  "
              f"Remaining: {g['top_prizes_remaining']}")
