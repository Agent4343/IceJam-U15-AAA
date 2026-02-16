#!/usr/bin/env python3
"""
Spordle Scraper for Tournoi International M15 du Grand Montréal
AAA-Elite Division

Fetches game results from the Spordle page and outputs them as JSON
that can be imported into the standings tracker via /api/games.

Usage:
    python scrape_spordle.py
    python scrape_spordle.py --schedule-id 186656
    python scrape_spordle.py --import-to http://localhost:8000
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

# Spordle page URL components
SPORDLE_BASE = "https://page.spordle.com"
ORG_SLUG = "tournoi-international-de-hockey-m15-du-grand-montreal"
PAGE_UUID = "0b7dd40b-e3bf-4b03-8a2a-9c7060714a65"
DEFAULT_SCHEDULE_ID = "186656"  # AAA-Elite division

# Known Spordle API patterns to try
API_PATTERNS = [
    "https://page-api.spordle.com/api/v2/page-tree/{org}/schedule/{schedule_id}/games",
    "https://page-api.spordle.com/api/v1/schedules/{schedule_id}/games",
    "https://api.spordle.com/page/api/v1/public/schedule/{schedule_id}/games",
    "https://play-api.spordle.com/api/v1/public/schedules/{schedule_id}/games",
    "https://api.spordle.com/public/api/v1/schedules/{schedule_id}/games",
    "https://page.spordle.com/api/schedules/{schedule_id}/games",
    "https://page-api.spordle.com/api/v2/schedules/{schedule_id}/games",
    "https://page-api.spordle.com/api/v2/schedules/{schedule_id}/results",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": f"{SPORDLE_BASE}/{ORG_SLUG}/schedule-stats-standings/{PAGE_UUID}?scheduleId={DEFAULT_SCHEDULE_ID}",
}


def try_api_endpoints(schedule_id: str) -> dict | None:
    """Try various Spordle API patterns to find game data."""
    session = requests.Session()
    session.headers.update(HEADERS)

    for pattern in API_PATTERNS:
        url = pattern.format(org=ORG_SLUG, schedule_id=schedule_id)
        print(f"  Trying: {url}")
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  SUCCESS! Got data from {url}")
                return data
            print(f"  -> {resp.status_code}")
        except Exception as e:
            print(f"  -> Error: {e}")

    return None


def scrape_page_html(schedule_id: str) -> str | None:
    """Fetch the Spordle page HTML and look for embedded data."""
    url = f"{SPORDLE_BASE}/{ORG_SLUG}/schedule-stats-standings/{PAGE_UUID}?scheduleId={schedule_id}"
    print(f"  Fetching page: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        print(f"  -> {resp.status_code}")
    except Exception as e:
        print(f"  -> Error: {e}")
    return None


def extract_data_from_html(html: str) -> dict | None:
    """Try to extract JSON data embedded in the page HTML."""
    import re

    # Look for __NEXT_DATA__ (Next.js pattern)
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for window.__data or similar patterns
    patterns = [
        r'window\.__(?:INITIAL_STATE|DATA|PRELOADED_STATE)__\s*=\s*({.*?});',
        r'window\.(?:gameData|scheduleData|gamesData)\s*=\s*({.*?});',
        r'"games"\s*:\s*(\[.*?\])',
    ]
    for pat in patterns:
        match = re.search(pat, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def convert_to_app_format(games_data: list) -> list:
    """Convert Spordle game data to our app's GameInput format."""
    results = []
    for game in games_data:
        # Adapt field names based on what Spordle returns
        # These field mappings may need adjustment based on actual API response
        result = {
            "team_a": game.get("homeTeam", {}).get("name", game.get("home_team", "")),
            "team_b": game.get("awayTeam", {}).get("name", game.get("away_team", "")),
            "goals_a": game.get("homeScore", game.get("home_score", 0)),
            "goals_b": game.get("awayScore", game.get("away_score", 0)),
            "ot": game.get("overtime", False),
            "division": "AAA-Elite",
            "franc_jeu_a": 0,
            "franc_jeu_b": 0,
        }
        if result["team_a"] and result["team_b"]:
            results.append(result)
    return results


def import_to_app(games: list, base_url: str):
    """Import games into the running app via API."""
    print(f"\nImporting {len(games)} games to {base_url}...")
    for i, game in enumerate(games, 1):
        try:
            resp = requests.post(f"{base_url}/api/games", json=game, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  [{i}/{len(games)}] Added: {game['team_a']} {game['goals_a']}-{game['goals_b']} {game['team_b']}")
            else:
                print(f"  [{i}/{len(games)}] FAILED ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"  [{i}/{len(games)}] ERROR: {e}")
        time.sleep(0.1)  # Be gentle


def main():
    parser = argparse.ArgumentParser(description="Scrape Spordle for AAA-Elite tournament data")
    parser.add_argument("--schedule-id", default=DEFAULT_SCHEDULE_ID, help="Spordle schedule ID")
    parser.add_argument("--import-to", metavar="URL", help="Import games to app (e.g., http://localhost:8000)")
    parser.add_argument("--output", "-o", default="aaa_elite_games.json", help="Output JSON file")
    args = parser.parse_args()

    print(f"Spordle Scraper - AAA-Elite Division (scheduleId={args.schedule_id})")
    print("=" * 60)

    # Step 1: Try API endpoints
    print("\n[1] Trying Spordle API endpoints...")
    data = try_api_endpoints(args.schedule_id)

    # Step 2: Try page HTML
    if not data:
        print("\n[2] Trying to fetch page HTML for embedded data...")
        html = scrape_page_html(args.schedule_id)
        if html:
            data = extract_data_from_html(html)
            if data:
                print("  Found embedded data!")

    if not data:
        print("\n** Could not fetch data from Spordle.")
        print("   The Spordle API may require browser-based access.")
        print()
        print("   Alternative approaches:")
        print("   1. Open the Spordle page in your browser")
        print("   2. Open Developer Tools (F12) -> Network tab")
        print("   3. Look for XHR/Fetch requests to API endpoints")
        print("   4. Copy the game data JSON and save to aaa_elite_games.json")
        print()
        print(f"   Spordle URL: {SPORDLE_BASE}/{ORG_SLUG}/schedule-stats-standings/{PAGE_UUID}?scheduleId={args.schedule_id}")
        print()
        print("   Or use: python load_seed_data.py to load known game data")
        sys.exit(1)

    # Convert and save
    games = data if isinstance(data, list) else data.get("games", data.get("data", []))
    app_games = convert_to_app_format(games)

    print(f"\nFound {len(app_games)} games")

    with open(args.output, "w") as f:
        json.dump(app_games, f, indent=2)
    print(f"Saved to {args.output}")

    # Import if requested
    if args.import_to:
        import_to_app(app_games, args.import_to)


if __name__ == "__main__":
    main()
