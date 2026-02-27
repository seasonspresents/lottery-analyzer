#!/bin/bash
cd /Users/mattias/az-lottery-analyzer
source .venv/bin/activate

for state in az ca; do
    python -c "
from scraper import scrape_summary, analyze_games, get_data_paths, STATE_CONFIG
import json
from datetime import datetime, timezone

state = '$state'
cfg = STATE_CONFIG[state]
print(f'[{datetime.now()}] Refreshing {cfg[\"name\"]}...')
summary = scrape_summary(state)
analyzed = analyze_games(summary)
_, _, analyzed_file = get_data_paths(state)
result = {'analyzed_at': datetime.now(timezone.utc).isoformat(), 'games': analyzed}
with open(analyzed_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[{datetime.now()}] Done. {len(analyzed)} {cfg[\"abbreviation\"]} games refreshed.')
"
done
