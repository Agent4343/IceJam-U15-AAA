"""
Grand Montreal International U15 Hockey Tournament - Standings Tracker
Tournoi International Laval 2026

Tracks round robin standings for the Greater Montreal International U15
Hockey Tournament. Sanctioned by Hockey Quebec.

Games are entered manually via the API. Tiebreaker rules are applied
automatically to determine rankings.

TIEBREAKER RULES (Article 9.7 - Hockey Quebec):
    a) Highest number of points
    b) Highest number of wins
    c) Least goals against
    d) Most goals for
    e) Quickest goal scored in all games played
    f) Most Franc Jeu (Fair Play) points
    g) By a draw

POINTS SYSTEM:
    Win = 2 points
    Tie = 1 point
    Loss = 0 points
    Franc Jeu (Fair Play) = 1 point

GAME FORMAT:
    Three periods: 12-12-15 min (stop time)
    Pool games: Tie stands
    Elimination (1/8, QF): 5 min 3v3 OT, then shootout
    Semi-finals & Finals: 10 min 3v3 OT, then shootout

MERCY RULE:
    7+ goal lead after 2nd period = running time in 3rd period
    (even if gap drops below 7; penalties still timed)
"""
from __future__ import annotations
import re
import uuid
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from functools import cmp_to_key
from urllib.request import urlopen, Request as URLRequest
from urllib.error import URLError

from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field, field_validator
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOURNAMENT_NAME = "Grand Montreal International U15 Hockey Tournament"
TOURNAMENT_SHORT = "Tournoi International Laval 2026"
TOURNAMENT_WEBSITE = "intm15.com"

# Multiplier for tournament time calculation (ensures game order takes precedence over time within game)
TOURNAMENT_TIME_MULTIPLIER = 100000

# Mercy rule threshold (7+ goal lead after 2nd period = running time)
MERCY_RULE_THRESHOLD = 7

# In-memory game storage
games_db: Dict[str, "Game"] = {}


@dataclass
class Game:
    game_id: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    ot: bool
    pim_a: int = 0
    pim_b: int = 0
    first_goal_team: str = ""  # Which team scored first ("a" or "b")
    first_goal_time_sec: Optional[int] = None  # Time in seconds when first goal was scored
    game_number: int = 0  # Order of game in tournament (for quickest goal tiebreaker)
    franc_jeu_a: int = 0  # Franc Jeu (Fair Play) points for team A (0 or 1)
    franc_jeu_b: int = 0  # Franc Jeu (Fair Play) points for team B (0 or 1)
    division: str = ""  # Division/category (AAA, AA, BB, etc.)


@dataclass
class TeamStats:
    name: str
    gp: int = 0
    w: int = 0
    l: int = 0
    t: int = 0
    pts: int = 0
    gf: int = 0
    ga: int = 0
    pim: int = 0
    franc_jeu: int = 0  # Franc Jeu (Fair Play) points accumulated
    first_goal_time: Optional[int] = None  # Earliest first goal in tournament (game_number * multiplier + seconds)
    division: str = ""


class GameInput(BaseModel):
    team_a: str = Field(..., min_length=1, max_length=100, description="Home team name")
    team_b: str = Field(..., min_length=1, max_length=100, description="Away team name")
    goals_a: int = Field(..., ge=0, le=100, description="Goals scored by team A")
    goals_b: int = Field(..., ge=0, le=100, description="Goals scored by team B")
    ot: bool = False
    pim_a: int = Field(default=0, ge=0, le=500, description="Penalty minutes for team A")
    pim_b: int = Field(default=0, ge=0, le=500, description="Penalty minutes for team B")
    first_goal_team: str = Field(default="", max_length=100, description="'a' or 'b' for which team scored first")
    first_goal_time_sec: Optional[int] = Field(default=None, ge=0, le=3600, description="Time in seconds when first goal was scored")
    franc_jeu_a: int = Field(default=0, ge=0, le=1, description="Franc Jeu point for team A (0 or 1)")
    franc_jeu_b: int = Field(default=0, ge=0, le=1, description="Franc Jeu point for team B (0 or 1)")
    division: str = Field(default="", max_length=50, description="Division (AAA, AA, BB, etc.)")

    @field_validator('team_b')
    @classmethod
    def teams_must_be_different(cls, v, info):
        if 'team_a' in info.data and v.strip().lower() == info.data['team_a'].strip().lower():
            raise ValueError('team_a and team_b must be different teams')
        return v


app = FastAPI(title=TOURNAMENT_NAME)
templates = Jinja2Templates(directory="templates")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def points_for_game(gf: int, ga: int) -> Tuple[int, int, int, int]:
    """
    Returns (pts, w, l, t) for a team.

    Laval 2026 points system:
        Win = 2 points
        Tie = 1 point
        Loss = 0 points
    (Franc Jeu points are tracked separately)
    """
    if gf == ga:
        return (1, 0, 0, 1)
    if gf > ga:
        return (2, 1, 0, 0)
    return (0, 0, 1, 0)


def compare_two_teams(t1: TeamStats, t2: TeamStats) -> int:
    """
    Compare two teams using Hockey Quebec Article 9.7 tiebreaker rules.
    Returns: -1 if t1 ranks higher, 1 if t2 ranks higher

    Tiebreaker order:
        a) Highest number of points
        b) Highest number of wins
        c) Least goals against
        d) Most goals for
        e) Quickest goal scored in all games played
        f) Most Franc Jeu points
        g) By a draw (alphabetical as deterministic fallback)
    """
    # a) Highest number of points
    if t1.pts != t2.pts:
        return -1 if t1.pts > t2.pts else 1

    # b) Highest number of wins
    if t1.w != t2.w:
        return -1 if t1.w > t2.w else 1

    # c) Least goals against
    if t1.ga != t2.ga:
        return -1 if t1.ga < t2.ga else 1

    # d) Most goals for
    if t1.gf != t2.gf:
        return -1 if t1.gf > t2.gf else 1

    # e) Quickest goal scored in all games played
    t1_first = t1.first_goal_time if t1.first_goal_time is not None else float('inf')
    t2_first = t2.first_goal_time if t2.first_goal_time is not None else float('inf')
    if t1_first != t2_first:
        return -1 if t1_first < t2_first else 1

    # f) Most Franc Jeu points
    if t1.franc_jeu != t2.franc_jeu:
        return -1 if t1.franc_jeu > t2.franc_jeu else 1

    # g) By a draw (alphabetical order as deterministic fallback)
    return -1 if t1.name.lower() < t2.name.lower() else 1


def calculate_standings(division: str = "") -> List[dict]:
    """Calculate standings from all games using Hockey Quebec tiebreaker rules."""
    if not games_db:
        return []

    stats: Dict[str, TeamStats] = {}

    # Sort games by game_number to process in order
    sorted_games = sorted(games_db.values(), key=lambda g: g.game_number)

    for game in sorted_games:
        # Filter by division if specified
        if division and game.division.lower() != division.lower():
            continue

        # Initialize teams if not seen
        if game.team_a not in stats:
            stats[game.team_a] = TeamStats(name=game.team_a, division=game.division)
        if game.team_b not in stats:
            stats[game.team_b] = TeamStats(name=game.team_b, division=game.division)

        team_a = stats[game.team_a]
        team_b = stats[game.team_b]

        # Update games played
        team_a.gp += 1
        team_b.gp += 1

        # Goals
        team_a.gf += game.goals_a
        team_a.ga += game.goals_b
        team_b.gf += game.goals_b
        team_b.ga += game.goals_a

        # Update PIM
        team_a.pim += game.pim_a
        team_b.pim += game.pim_b

        # Update Franc Jeu points
        team_a.franc_jeu += game.franc_jeu_a
        team_b.franc_jeu += game.franc_jeu_b

        # Track quickest goal of tournament for each team (tiebreaker e)
        if game.first_goal_team and game.first_goal_time_sec is not None:
            tournament_time = game.game_number * TOURNAMENT_TIME_MULTIPLIER + game.first_goal_time_sec

            if game.first_goal_team.lower() == "a":
                if team_a.first_goal_time is None or tournament_time < team_a.first_goal_time:
                    team_a.first_goal_time = tournament_time
            elif game.first_goal_team.lower() == "b":
                if team_b.first_goal_time is None or tournament_time < team_b.first_goal_time:
                    team_b.first_goal_time = tournament_time

        # Calculate points (Win=2, Tie=1, Loss=0)
        pts_a, w_a, l_a, t_a = points_for_game(game.goals_a, game.goals_b)
        pts_b, w_b, l_b, t_b = points_for_game(game.goals_b, game.goals_a)

        team_a.pts += pts_a
        team_a.w += w_a
        team_a.l += l_a
        team_a.t += t_a

        team_b.pts += pts_b
        team_b.w += w_b
        team_b.l += l_b
        team_b.t += t_b

    # Sort using Hockey Quebec tiebreaker rules
    teams_list = list(stats.values())
    sorted_teams = sorted(teams_list, key=cmp_to_key(compare_two_teams))

    # Build standings with rank
    standings = []
    for i, team in enumerate(sorted_teams, 1):
        standings.append({
            "rank": i,
            "team": team.name,
            "gp": team.gp,
            "w": team.w,
            "l": team.l,
            "t": team.t,
            "pts": team.pts,
            "gf": team.gf,
            "ga": team.ga,
            "gd": team.gf - team.ga,
            "pim": team.pim,
            "franc_jeu": team.franc_jeu,
            "division": team.division,
        })

    return standings


# ============ PAGE ROUTES ============

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/rules", response_class=HTMLResponse)
def rules(request: Request):
    return templates.TemplateResponse("rules.html", {"request": request})


# ============ API ROUTES ============

@app.get("/api/standings")
def standings(
    division: str = Query("", description="Filter by division (AAA, AA, BB, etc.)"),
    team: str = Query("", description="Team to highlight"),
):
    """Get current standings with Hockey Quebec tiebreaker rules applied."""
    all_standings = calculate_standings(division)

    tracked_rank = None
    if team:
        for s in all_standings:
            if team.lower() in s["team"].lower():
                tracked_rank = s["rank"]
                break

    # Get list of divisions
    divisions = sorted(set(g.division for g in games_db.values() if g.division))

    return {
        "ok": True,
        "tournament": TOURNAMENT_SHORT,
        "division_filter": division,
        "divisions": divisions,
        "tracked": {"team": team, "rank": tracked_rank} if team else None,
        "teams_found": len(all_standings),
        "standings": all_standings,
        "games_count": len(games_db),
    }


@app.post("/api/games")
def add_game(game: GameInput):
    """Add a new game result."""
    game_id = str(uuid.uuid4())[:8]
    game_number = len(games_db) + 1

    new_game = Game(
        game_id=game_id,
        team_a=norm(game.team_a),
        team_b=norm(game.team_b),
        goals_a=game.goals_a,
        goals_b=game.goals_b,
        ot=game.ot,
        pim_a=game.pim_a,
        pim_b=game.pim_b,
        first_goal_team=game.first_goal_team.lower() if game.first_goal_team else "",
        first_goal_time_sec=game.first_goal_time_sec,
        game_number=game_number,
        franc_jeu_a=game.franc_jeu_a,
        franc_jeu_b=game.franc_jeu_b,
        division=norm(game.division),
    )
    games_db[game_id] = new_game
    return {"ok": True, "game_id": game_id, "game": asdict(new_game)}


@app.get("/api/games")
def list_games(division: str = Query("", description="Filter by division")):
    """List all games."""
    games = list(games_db.values())
    if division:
        games = [g for g in games if g.division.lower() == division.lower()]

    return {
        "ok": True,
        "count": len(games),
        "games": [asdict(g) for g in games],
    }


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str):
    """Delete a game by ID."""
    if game_id not in games_db:
        raise HTTPException(status_code=404, detail="Game not found")
    del games_db[game_id]
    return {"ok": True, "deleted": game_id}


@app.post("/api/games/bulk")
def add_games_bulk(games: List[GameInput]):
    """Add multiple game results at once (for importing from Spordle or seed data)."""
    results = []
    for game in games:
        game_id = str(uuid.uuid4())[:8]
        game_number = len(games_db) + 1

        new_game = Game(
            game_id=game_id,
            team_a=norm(game.team_a),
            team_b=norm(game.team_b),
            goals_a=game.goals_a,
            goals_b=game.goals_b,
            ot=game.ot,
            pim_a=game.pim_a,
            pim_b=game.pim_b,
            first_goal_team=game.first_goal_team.lower() if game.first_goal_team else "",
            first_goal_time_sec=game.first_goal_time_sec,
            game_number=game_number,
            franc_jeu_a=game.franc_jeu_a,
            franc_jeu_b=game.franc_jeu_b,
            division=norm(game.division),
        )
        games_db[game_id] = new_game
        results.append({"game_id": game_id, "team_a": new_game.team_a, "team_b": new_game.team_b})

    return {"ok": True, "imported": len(results), "games": results}


@app.post("/api/games/upload")
async def upload_games(file: UploadFile = File(...)):
    """Upload a JSON file of game results. Accepts a JSON array of game objects."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    games_list = data if isinstance(data, list) else data.get("games", [])
    if not games_list:
        raise HTTPException(status_code=400, detail="No games found in file")

    results = []
    errors = []
    for i, g in enumerate(games_list):
        try:
            game_input = GameInput(
                team_a=g.get("team_a", ""),
                team_b=g.get("team_b", ""),
                goals_a=g.get("goals_a", 0),
                goals_b=g.get("goals_b", 0),
                ot=g.get("ot", False),
                pim_a=g.get("pim_a", 0),
                pim_b=g.get("pim_b", 0),
                first_goal_team=g.get("first_goal_team", ""),
                first_goal_time_sec=g.get("first_goal_time_sec"),
                franc_jeu_a=g.get("franc_jeu_a", 0),
                franc_jeu_b=g.get("franc_jeu_b", 0),
                division=g.get("division", "AAA-Elite"),
            )
            game_id = str(uuid.uuid4())[:8]
            game_number = len(games_db) + 1
            new_game = Game(
                game_id=game_id,
                team_a=norm(game_input.team_a),
                team_b=norm(game_input.team_b),
                goals_a=game_input.goals_a,
                goals_b=game_input.goals_b,
                ot=game_input.ot,
                pim_a=game_input.pim_a,
                pim_b=game_input.pim_b,
                first_goal_team=game_input.first_goal_team.lower() if game_input.first_goal_team else "",
                first_goal_time_sec=game_input.first_goal_time_sec,
                game_number=game_number,
                franc_jeu_a=game_input.franc_jeu_a,
                franc_jeu_b=game_input.franc_jeu_b,
                division=norm(game_input.division),
            )
            games_db[game_id] = new_game
            results.append(game_id)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return {
        "ok": True,
        "imported": len(results),
        "errors": len(errors),
        "error_details": errors[:10],
    }


SPORDLE_ORG = "tournoi-international-de-hockey-m15-du-grand-montreal"
SPORDLE_PAGE_UUID = "0b7dd40b-e3bf-4b03-8a2a-9c7060714a65"
SPORDLE_AAA_ELITE_SCHEDULE = "186656"


def _spordle_headers(schedule_id: str) -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/html, */*",
        "Referer": f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}",
    }


def _deep_find_lists(obj, path="", max_depth=6) -> list:
    """Recursively find all lists in a nested dict, return (path, length, sample_keys)."""
    results = []
    if max_depth <= 0:
        return results
    if isinstance(obj, dict):
        for k, v in obj.items():
            cp = f"{path}.{k}" if path else k
            if isinstance(v, list):
                sample_keys = []
                if v and isinstance(v[0], dict):
                    sample_keys = list(v[0].keys())[:10]
                results.append({"path": cp, "length": len(v), "sample_keys": sample_keys})
            results.extend(_deep_find_lists(v, cp, max_depth - 1))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):  # only check first 3 items
            results.extend(_deep_find_lists(v, f"{path}[{i}]", max_depth - 1))
    return results


def _extract_api_urls(obj, found=None, depth=0) -> list:
    """Find any URL strings in the data that point to Spordle APIs."""
    if found is None:
        found = []
    if depth > 6:
        return found
    if isinstance(obj, str) and ("spordle" in obj.lower() or "api" in obj.lower()) and obj.startswith("http"):
        found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_api_urls(v, found, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:20]:
            _extract_api_urls(v, found, depth + 1)
    return found


def _navigate(obj, dotpath: str):
    """Navigate a nested dict/list by dot-separated path. Returns None on failure."""
    for part in dotpath.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit():
            idx = int(part)
            obj = obj[idx] if idx < len(obj) else None
        else:
            return None
        if obj is None:
            return None
    return obj


def _try_fetch_json(url: str, headers: dict, timeout: int = 10):
    """Try to fetch JSON from a URL, return (data, error)."""
    try:
        req = URLRequest(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode()), None
    except Exception as e:
        return None, str(e)
    return None, "non-200"


def _parse_spordle_games(games_raw: list, division: str) -> list:
    """Convert raw Spordle game objects to our format."""
    imported = []
    for g in games_raw:
        if not isinstance(g, dict):
            continue
        try:
            # Try many possible field name patterns
            team_a = ""
            team_b = ""
            goals_a = None
            goals_b = None

            # Home team name
            for path in ["homeTeam.short_name", "homeTeam.name", "homeTeam.team_name",
                          "home_team.short_name", "home_team.name", "home.name",
                          "home_team_name", "teamA", "team_a", "home"]:
                val = _navigate(g, path)
                if val and isinstance(val, str):
                    team_a = val
                    break

            # Away team name
            for path in ["awayTeam.short_name", "awayTeam.name", "awayTeam.team_name",
                          "away_team.short_name", "away_team.name", "away.name",
                          "away_team_name", "teamB", "team_b", "away"]:
                val = _navigate(g, path)
                if val and isinstance(val, str):
                    team_b = val
                    break

            # Scores
            for key in ["home_score", "homeScore", "score_home", "goals_a",
                         "home_goals", "homeGoals", "result.home"]:
                val = _navigate(g, key)
                if val is not None:
                    goals_a = int(val)
                    break
            for key in ["away_score", "awayScore", "score_away", "goals_b",
                         "away_goals", "awayGoals", "result.away"]:
                val = _navigate(g, key)
                if val is not None:
                    goals_b = int(val)
                    break

            if not team_a or not team_b:
                continue
            if goals_a is None or goals_b is None:
                continue

            ot = bool(g.get("overtime") or g.get("ot") or g.get("is_overtime")
                      or g.get("period", 0) > 3
                      or (g.get("result", {}) or {}).get("overtime"))

            imported.append({
                "team_a": str(team_a).strip(),
                "team_b": str(team_b).strip(),
                "goals_a": goals_a,
                "goals_b": goals_b,
                "division": division,
                "ot": ot,
            })
        except Exception:
            continue
    return imported


@app.get("/api/spordle/debug")
def debug_spordle(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE),
):
    """Debug endpoint: show what Spordle returns so we can find the game data."""
    headers = _spordle_headers(schedule_id)
    page_url = f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}"

    result = {"page_url": page_url, "next_data": None, "api_urls_found": [], "lists_found": [], "page_props_keys": None}

    try:
        req = URLRequest(page_url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode()
            result["html_size"] = len(html)

            # Extract __NEXT_DATA__
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
            if match:
                next_data = json.loads(match.group(1))
                result["next_data_keys"] = list(next_data.keys())

                # Show pageProps structure
                page_props = _navigate(next_data, "props.pageProps")
                if page_props and isinstance(page_props, dict):
                    result["page_props_keys"] = list(page_props.keys())
                    # Show sub-keys for each pageProps entry
                    for k, v in page_props.items():
                        if isinstance(v, dict):
                            result[f"pageProps.{k}_keys"] = list(v.keys())[:20]
                        elif isinstance(v, list):
                            result[f"pageProps.{k}_len"] = len(v)
                            if v and isinstance(v[0], dict):
                                result[f"pageProps.{k}_sample_keys"] = list(v[0].keys())[:15]
                        elif isinstance(v, str) and len(v) < 200:
                            result[f"pageProps.{k}"] = v

                # Find all lists in the data
                result["lists_found"] = _deep_find_lists(next_data)

                # Find any API URLs
                result["api_urls_found"] = _extract_api_urls(next_data)

                # Look for buildId (useful for Next.js API routes)
                result["buildId"] = next_data.get("buildId")

            # Also look for any fetch/XHR URLs in the JavaScript
            api_matches = re.findall(r'(?:fetch|axios|get|post)\s*\(\s*["\']([^"\']+spordle[^"\']*)["\']', html)
            if api_matches:
                result["js_api_calls"] = api_matches[:10]

            # Look for GraphQL endpoints
            gql_matches = re.findall(r'["\']([^"\']*graphql[^"\']*)["\']', html, re.IGNORECASE)
            if gql_matches:
                result["graphql_endpoints"] = gql_matches[:5]

            # Look for API base URL patterns
            base_matches = re.findall(r'["\']?(https?://[^"\']*(?:api|spordle)[^"\']*)["\']?', html)
            if base_matches:
                # Deduplicate and limit
                unique = list(dict.fromkeys(base_matches))[:20]
                result["api_base_urls_in_html"] = unique

    except Exception as e:
        result["error"] = str(e)

    return result


@app.get("/api/spordle/fetch")
def fetch_spordle(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE, description="Spordle schedule ID"),
    division: str = Query(default="AAA-Elite", description="Division label to assign"),
    auto_import: bool = Query(default=False, description="Auto-import fetched games"),
):
    """
    Fetch game data from Spordle (server-side). Railway can reach Spordle directly.
    Tries multiple strategies to find game data.
    """
    headers = _spordle_headers(schedule_id)
    page_url = f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}"
    tried = []
    games_raw = []
    debug_info = {}

    # ── Strategy 1: Try known API patterns ──
    api_patterns = [
        "https://page-api.spordle.com/api/v2/page-tree/{org}/schedule/{sid}/games",
        "https://page-api.spordle.com/api/v1/schedules/{sid}/games",
        "https://api.spordle.com/page/api/v1/public/schedule/{sid}/games",
        "https://play-api.spordle.com/api/v1/public/schedules/{sid}/games",
        "https://page-api.spordle.com/api/v2/schedules/{sid}/games",
        "https://page-api.spordle.com/api/v1/page-tree/{org}/pages/{page}/components",
        "https://page-api.spordle.com/api/v1/schedules/{sid}",
        "https://page-api.spordle.com/public/api/v1/schedules/{sid}/games",
        "https://page-api.spordle.com/api/v1/page/{page}/schedule/{sid}",
    ]
    for pattern in api_patterns:
        url = pattern.format(org=SPORDLE_ORG, sid=schedule_id, page=SPORDLE_PAGE_UUID)
        tried.append(url)
        data, err = _try_fetch_json(url, {**headers, "Accept": "application/json"})
        if data:
            debug_info["api_hit"] = url
            # Try to extract games from this response
            if isinstance(data, list):
                games_raw = data
            elif isinstance(data, dict):
                # Look for game arrays
                for linfo in _deep_find_lists(data):
                    if any(k in " ".join(linfo.get("sample_keys", [])).lower()
                           for k in ["team", "home", "away", "score", "goal"]):
                        games_raw = _navigate(data, linfo["path"]) or []
                        break
            if games_raw:
                break

    # ── Strategy 2: Parse the Spordle page HTML ──
    if not games_raw:
        tried.append(page_url)
        try:
            req = URLRequest(page_url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                html = resp.read().decode()
                debug_info["html_size"] = len(html)

                # Extract __NEXT_DATA__
                match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
                if match:
                    next_data = json.loads(match.group(1))

                    # Find all lists and check for game-like data
                    all_lists = _deep_find_lists(next_data)
                    debug_info["lists_in_next_data"] = all_lists

                    for linfo in all_lists:
                        keys_str = " ".join(linfo.get("sample_keys", [])).lower()
                        if any(k in keys_str for k in ["team", "home", "away", "score", "goal"]):
                            games_raw = _navigate(next_data, linfo["path"]) or []
                            debug_info["games_path"] = linfo["path"]
                            break

                    # If no game lists found, look for API URLs in the data
                    if not games_raw:
                        api_urls = _extract_api_urls(next_data)
                        debug_info["embedded_api_urls"] = api_urls
                        for api_url in api_urls:
                            tried.append(api_url)
                            data, err = _try_fetch_json(api_url, headers)
                            if data:
                                if isinstance(data, list):
                                    games_raw = data
                                elif isinstance(data, dict):
                                    for li in _deep_find_lists(data):
                                        ks = " ".join(li.get("sample_keys", [])).lower()
                                        if any(k in ks for k in ["team", "home", "away", "score"]):
                                            games_raw = _navigate(data, li["path"]) or []
                                            break
                                if games_raw:
                                    break

                    # Also try the buildId for Next.js data routes
                    build_id = next_data.get("buildId", "")
                    page_props = _navigate(next_data, "props.pageProps") or {}
                    debug_info["page_props_keys"] = list(page_props.keys()) if isinstance(page_props, dict) else str(type(page_props))

                    if build_id and not games_raw:
                        # Next.js data route pattern
                        data_url = f"https://page.spordle.com/_next/data/{build_id}/en/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}.json?scheduleId={schedule_id}"
                        tried.append(data_url)
                        data, err = _try_fetch_json(data_url, headers)
                        if data:
                            for li in _deep_find_lists(data):
                                ks = " ".join(li.get("sample_keys", [])).lower()
                                if any(k in ks for k in ["team", "home", "away", "score"]):
                                    games_raw = _navigate(data, li["path"]) or []
                                    break

                # Look for other embedded JSON data
                if not games_raw:
                    for pattern_re in [
                        r'window\.__(?:INITIAL_STATE|DATA|STORE)__\s*=\s*({.*?});',
                        r'window\.GAMES\s*=\s*(\[.*?\]);',
                    ]:
                        match2 = re.search(pattern_re, html, re.DOTALL)
                        if match2:
                            try:
                                embedded = json.loads(match2.group(1))
                                if isinstance(embedded, list):
                                    games_raw = embedded
                                    break
                            except json.JSONDecodeError:
                                pass

        except Exception as e:
            debug_info["page_error"] = str(e)

    # ── Convert to our format ──
    imported_games = _parse_spordle_games(games_raw, division)

    if not imported_games and not games_raw:
        return {
            "ok": False,
            "error": "Connected to Spordle but couldn't find game data. The game data is likely loaded client-side via JavaScript.",
            "tried": tried,
            "debug": debug_info,
            "hint": "Use the /api/spordle/debug endpoint to explore the data structure, or paste game data manually.",
            "spordle_url": page_url,
        }

    result = {
        "ok": True,
        "games_found": len(imported_games),
        "games": imported_games,
        "debug": debug_info,
    }

    # Auto-import if requested
    if auto_import and imported_games:
        count = 0
        for gd in imported_games:
            try:
                game_id = str(uuid.uuid4())[:8]
                game_number = len(games_db) + 1
                new_game = Game(
                    game_id=game_id,
                    team_a=norm(gd["team_a"]),
                    team_b=norm(gd["team_b"]),
                    goals_a=gd["goals_a"],
                    goals_b=gd["goals_b"],
                    ot=gd.get("ot", False),
                    game_number=game_number,
                    division=norm(division),
                )
                games_db[game_id] = new_game
                count += 1
            except Exception:
                continue
        result["auto_imported"] = count

    return result


@app.delete("/api/games")
def clear_games():
    """Clear all games."""
    games_db.clear()
    return {"ok": True, "message": "All games cleared"}
