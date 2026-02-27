"""
Multi-State Lottery Scratcher Analyzer - Local Web Dashboard
Run: python app.py
Visit: http://localhost:5000
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests as http_requests
from flask import Flask, render_template, jsonify, request, make_response

from scraper import (
    run_full_scrape, scrape_summary, scrape_prize_tiers, analyze_games,
    compute_velocity, save_snapshot,
    DATA_DIR, STATE_CONFIG, DEFAULT_STATE, get_data_paths,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def get_current_state():
    """Resolve the active state: ?state= param > cookie > DEFAULT_STATE."""
    state = request.args.get("state", "").lower()
    if state in STATE_CONFIG:
        return state
    state = request.cookies.get("lottery_state", "").lower()
    if state in STATE_CONFIG:
        return state
    return DEFAULT_STATE


def get_analyzed_data(state_code):
    """Load analyzed data from disk for the given state."""
    _, _, analyzed_file = get_data_paths(state_code)
    if analyzed_file.exists():
        with open(analyzed_file) as f:
            return json.load(f)
    return None


def _ensure_data_dir():
    """Ensure DATA_DIR exists (needed on Vercel where /tmp can be wiped)."""
    DATA_DIR.mkdir(exist_ok=True)


def render_with_state(template, state_code, **kwargs):
    """Render a template with state config injected + set the state cookie."""
    resp = make_response(render_template(
        template,
        state=state_code,
        state_cfg=STATE_CONFIG[state_code],
        states=STATE_CONFIG,
        **kwargs,
    ))
    resp.set_cookie("lottery_state", state_code, max_age=60 * 60 * 24 * 365)
    return resp


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    state = get_current_state()
    data = get_analyzed_data(state)
    if not data:
        return render_with_state("loading.html", state)

    games = data["games"]

    # Filter parameters
    min_price = request.args.get("min_price", type=float, default=0)
    max_price = request.args.get("max_price", type=float, default=999)
    sort_by = request.args.get("sort", default="score")

    filtered = [g for g in games if min_price <= g["price"] <= max_price]

    sort_keys = {
        "score": lambda g: g["score"],
        "price": lambda g: g["price"],
        "odds": lambda g: g["win_probability"],
        "top_prize": lambda g: g["top_prize"],
        "remaining": lambda g: g["top_prizes_remaining"],
        "ratio": lambda g: g["top_prize_to_price_ratio"],
        "ev": lambda g: g.get("ev_per_dollar") or 0,
        "payout": lambda g: g.get("payout_pct") or 0,
    }
    sort_fn = sort_keys.get(sort_by, sort_keys["score"])
    filtered.sort(key=sort_fn, reverse=True)

    # Stats
    price_groups = {}
    for g in games:
        p = int(g["price"])
        if p not in price_groups:
            price_groups[p] = []
        price_groups[p].append(g)

    best_per_price = {}
    for p, group in sorted(price_groups.items()):
        best = max(group, key=lambda g: g["score"])
        best_per_price[p] = best

    # Smart picks: top 3 by safety score
    smart_picks = sorted(games, key=lambda g: g.get("safety_score", 0), reverse=True)[:3]

    return render_with_state(
        "dashboard.html",
        state,
        games=filtered,
        all_games=games,
        best_per_price=best_per_price,
        smart_picks=smart_picks,
        analyzed_at=data.get("analyzed_at", "unknown"),
        sort_by=sort_by,
        min_price=min_price,
        max_price=max_price,
        total_games=len(games),
    )


@app.route("/plan")
def plan():
    """Mobile-friendly game plan to take to the store."""
    state = get_current_state()
    data = get_analyzed_data(state)
    if not data:
        return render_with_state("loading.html", state)
    games = data["games"]
    by_safety = sorted(games, key=lambda g: g.get("safety_score", 0), reverse=True)
    smart_picks = by_safety[:3]
    backup_picks = by_safety[3:5]
    return render_with_state("plan.html", state, smart_picks=smart_picks, backup_picks=backup_picks, analyzed_at=data.get("analyzed_at", "unknown"))


@app.route("/tracker")
def tracker():
    """Win/loss session tracker."""
    state = get_current_state()
    data = get_analyzed_data(state)
    game_names = []
    game_odds = {}
    if data:
        game_names = [{"name": g["name"], "price": g["price"], "game_num": g["game_num"]} for g in data["games"]]
        game_names.sort(key=lambda g: g["name"])
        for g in data["games"]:
            game_odds[g["name"]] = {
                "win_rate_pct": g.get("win_rate_pct", 0),
                "odds_1_in": g.get("odds_1_in", 0),
            }
    return render_with_state("tracker.html", state, game_names=game_names, game_odds=game_odds)


@app.route("/go")
def sales():
    """Direct response sales page."""
    return render_template("sales.html")


@app.route("/stores")
def stores():
    state = get_current_state()
    data = get_analyzed_data(state)
    return render_with_state(
        "stores.html",
        state,
        total_games=len(data["games"]) if data else 0,
        analyzed_at=data.get("analyzed_at", "unknown") if data else "unknown",
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/retailers")
def api_retailers():
    """Proxy to state lottery retailer API to avoid CORS issues."""
    state = request.args.get("state", DEFAULT_STATE).lower()
    if state not in STATE_CONFIG:
        state = DEFAULT_STATE
    cfg = STATE_CONFIG[state]

    if not cfg["has_retailer_api"]:
        return jsonify({"error": f"No retailer API available for {cfg['name']}. Visit {cfg['official_base_url']} to find retailers."}), 404

    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if not lat or not lng:
        return jsonify({"error": "lat and lng required"}), 400
    try:
        resp = http_requests.get(
            cfg["retailer_api_url"],
            params={"latitude": lat, "longitude": lng},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/games")
def api_games():
    state = request.args.get("state", get_current_state()).lower()
    if state not in STATE_CONFIG:
        state = DEFAULT_STATE
    data = get_analyzed_data(state)
    if not data:
        return jsonify({"error": "No data yet. Run scraper first."}), 404
    return jsonify(data)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Re-scrape and re-analyze data for a given state."""
    state = request.args.get("state", get_current_state()).lower()
    if state not in STATE_CONFIG:
        state = DEFAULT_STATE
    try:
        _ensure_data_dir()
        summary = scrape_summary(state)
        # Skip tier scraping on Vercel (too slow for serverless 10s timeout)
        # Tier data is populated by daily_refresh.sh or local runs
        tier_data = None
        if not os.environ.get("VERCEL"):
            try:
                game_nums = [g["game_num"] for g in summary]
                tier_data = scrape_prize_tiers(state, game_nums)
            except Exception:
                pass
        analyzed = analyze_games(summary, tier_data=tier_data)
        compute_velocity(state, analyzed)
        save_snapshot(state, analyzed)
        result = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "games": analyzed,
        }
        _, _, analyzed_file = get_data_paths(state)
        with open(analyzed_file, "w") as f:
            json.dump(result, f, indent=2)
        return jsonify({"status": "ok", "games_count": len(analyzed), "analyzed_at": result["analyzed_at"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Auto-scrape default state on first run if no data exists
    _, _, default_analyzed = get_data_paths(DEFAULT_STATE)
    if not default_analyzed.exists():
        print("No data found. Running initial scrape ...")
        try:
            run_full_scrape(skip_detail=True, state_code=DEFAULT_STATE)
        except Exception as e:
            print(f"Initial scrape failed: {e}")
            print("Start the server anyway -- use /api/refresh to retry.")

    import ssl
    cert_dir = Path(__file__).parent / "certs"
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        print("\n  Lottery Scratcher Analyzer")
        print("  https://az-scratchers:5050\n")
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(cert_file), str(key_file))
        app.run(host="127.0.0.1", port=5050, ssl_context=ssl_ctx)
    else:
        print("\n  Lottery Scratcher Analyzer")
        print("  http://localhost:5000\n")
        app.run(host="127.0.0.1", port=5000)
