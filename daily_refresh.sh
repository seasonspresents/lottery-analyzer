#!/bin/bash
cd /Users/mattias/az-lottery-analyzer
source .venv/bin/activate

for state in az ca; do
    python -c "
from scraper import scrape_summary, scrape_prize_tiers, analyze_games, compute_velocity, save_snapshot, get_data_paths, STATE_CONFIG
import json
from datetime import datetime, timezone

state = '$state'
cfg = STATE_CONFIG[state]
print(f'[{datetime.now()}] Refreshing {cfg[\"name\"]}...')
summary = scrape_summary(state)

# Scrape prize tiers for EV calculations
tier_data = None
try:
    game_nums = [g['game_num'] for g in summary]
    tier_data = scrape_prize_tiers(state, game_nums)
except Exception as e:
    print(f'  Tier scrape failed: {e}')

analyzed = analyze_games(summary, tier_data=tier_data)

# Velocity: compute then snapshot
compute_velocity(state, analyzed)
save_snapshot(state, analyzed)

_, _, analyzed_file = get_data_paths(state)
result = {'analyzed_at': datetime.now(timezone.utc).isoformat(), 'games': analyzed}
with open(analyzed_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[{datetime.now()}] Done. {len(analyzed)} {cfg[\"abbreviation\"]} games refreshed.')
"
done
