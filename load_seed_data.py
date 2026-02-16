#!/usr/bin/env python3
"""
Load AAA-Elite seed data into the standings tracker.

Loads game results from a JSON file and imports them via the API.
Can also load from a Spordle export or manually created data file.

Usage:
    # Start the app first: uvicorn app:app --port 8000
    python load_seed_data.py                          # Load from aaa_elite_games.json
    python load_seed_data.py --file my_games.json     # Load from custom file
    python load_seed_data.py --url http://host:8000   # Custom app URL
"""
import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

DEFAULT_URL = "http://localhost:8000"
DEFAULT_FILE = "aaa_elite_games.json"


def load_games(file_path: str) -> list:
    """Load games from a JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "games" in data:
        return data["games"]
    return []


def import_games(games: list, base_url: str, clear_first: bool = False):
    """Import games into the running app via API."""
    if clear_first:
        print(f"Clearing existing games at {base_url}...")
        try:
            resp = requests.delete(f"{base_url}/api/games", timeout=5)
            print(f"  Cleared: {resp.json()}")
        except Exception as e:
            print(f"  Warning: Could not clear games: {e}")

    print(f"\nImporting {len(games)} games to {base_url}...")
    success = 0
    failed = 0

    for i, game in enumerate(games, 1):
        # Ensure required fields
        payload = {
            "team_a": game.get("team_a", ""),
            "team_b": game.get("team_b", ""),
            "goals_a": game.get("goals_a", 0),
            "goals_b": game.get("goals_b", 0),
            "ot": game.get("ot", False),
            "pim_a": game.get("pim_a", 0),
            "pim_b": game.get("pim_b", 0),
            "first_goal_team": game.get("first_goal_team", ""),
            "first_goal_time_sec": game.get("first_goal_time_sec"),
            "franc_jeu_a": game.get("franc_jeu_a", 0),
            "franc_jeu_b": game.get("franc_jeu_b", 0),
            "division": game.get("division", "AAA-Elite"),
        }

        if not payload["team_a"] or not payload["team_b"]:
            print(f"  [{i}] SKIP: Missing team name")
            failed += 1
            continue

        try:
            resp = requests.post(f"{base_url}/api/games", json=payload, timeout=5)
            if resp.status_code == 200:
                print(f"  [{i}/{len(games)}] {payload['team_a']} {payload['goals_a']}-{payload['goals_b']} {payload['team_b']}")
                success += 1
            else:
                print(f"  [{i}/{len(games)}] FAILED ({resp.status_code}): {resp.text[:100]}")
                failed += 1
        except Exception as e:
            print(f"  [{i}/{len(games)}] ERROR: {e}")
            failed += 1

        time.sleep(0.05)

    print(f"\nDone! {success} imported, {failed} failed.")

    # Show standings
    try:
        resp = requests.get(f"{base_url}/api/standings", params={"division": "AAA-Elite"}, timeout=5)
        data = resp.json()
        standings = data.get("standings", [])
        if standings:
            print(f"\n{'#':>3} {'Team':<30} {'GP':>3} {'W':>3} {'L':>3} {'T':>3} {'PTS':>4} {'GF':>4} {'GA':>4} {'FJ':>3}")
            print("-" * 80)
            for s in standings:
                print(f"{s['rank']:>3} {s['team']:<30} {s['gp']:>3} {s['w']:>3} {s['l']:>3} {s['t']:>3} {s['pts']:>4} {s['gf']:>4} {s['ga']:>4} {s['franc_jeu']:>3}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Load AAA-Elite game data into standings tracker")
    parser.add_argument("--file", "-f", default=DEFAULT_FILE, help="JSON file with game data")
    parser.add_argument("--url", "-u", default=DEFAULT_URL, help="App base URL")
    parser.add_argument("--clear", action="store_true", help="Clear existing games before import")
    args = parser.parse_args()

    try:
        games = load_games(args.file)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        print("\nTo get game data, either:")
        print("  1. Run: python scrape_spordle.py")
        print("  2. Create aaa_elite_games.json manually")
        print("  3. Export from Spordle page (browser dev tools)")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {args.file}: {e}")
        sys.exit(1)

    if not games:
        print(f"No games found in {args.file}")
        sys.exit(1)

    import_games(games, args.url, clear_first=args.clear)


if __name__ == "__main__":
    main()
