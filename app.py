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

# Spordle API endpoint patterns to try (server-side fetch)
SPORDLE_API_URLS = [
    "https://page-api.spordle.com/api/v2/page-tree/{org}/schedule/{sid}/games",
    "https://page-api.spordle.com/api/v1/schedules/{sid}/games",
    "https://api.spordle.com/page/api/v1/public/schedule/{sid}/games",
    "https://play-api.spordle.com/api/v1/public/schedules/{sid}/games",
    "https://page-api.spordle.com/api/v2/schedules/{sid}/games",
]


@app.get("/api/spordle/fetch")
def fetch_spordle(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE, description="Spordle schedule ID"),
    division: str = Query(default="AAA-Elite", description="Division label to assign"),
    auto_import: bool = Query(default=False, description="Auto-import fetched games"),
):
    """
    Fetch game data from Spordle (server-side). Railway can reach Spordle directly.
    Tries multiple API endpoint patterns.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json, text/html, */*",
        "Referer": f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}",
    }

    # Try fetching the Spordle page HTML first to find embedded data or API URLs
    page_url = f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}"
    tried = []
    raw_data = None

    # Try known API patterns
    for pattern in SPORDLE_API_URLS:
        url = pattern.format(org=SPORDLE_ORG, sid=schedule_id)
        tried.append(url)
        try:
            req = URLRequest(url, headers=headers)
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    raw_data = json.loads(resp.read().decode())
                    break
        except Exception as e:
            logger.info(f"Spordle API {url}: {e}")
            continue

    # Try the page HTML for embedded data
    if not raw_data:
        tried.append(page_url)
        try:
            req = URLRequest(page_url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                html = resp.read().decode()
                # Look for __NEXT_DATA__ (Next.js SSR)
                import re as _re
                match = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
                if match:
                    raw_data = json.loads(match.group(1))
                # Look for inline data
                if not raw_data:
                    match = _re.search(r'window\.__(?:INITIAL_STATE|DATA)__\s*=\s*({.*?});', html)
                    if match:
                        raw_data = json.loads(match.group(1))
        except Exception as e:
            logger.info(f"Spordle page fetch: {e}")

    if not raw_data:
        return {
            "ok": False,
            "error": "Could not fetch data from Spordle. The API patterns may have changed.",
            "tried": tried,
            "hint": "Try opening the Spordle page in your browser, copying game data, and pasting it into the Import section.",
            "spordle_url": page_url,
        }

    # Try to extract games from the response
    games_raw = []
    if isinstance(raw_data, list):
        games_raw = raw_data
    elif isinstance(raw_data, dict):
        # Navigate common response structures
        for key in ["games", "data", "results", "items", "props.pageProps.games",
                     "props.pageProps.data.games", "props.pageProps.schedule.games"]:
            obj = raw_data
            try:
                for part in key.split("."):
                    obj = obj[part] if isinstance(obj, dict) else obj
                if isinstance(obj, list) and len(obj) > 0:
                    games_raw = obj
                    break
            except (KeyError, TypeError, IndexError):
                continue

    # Convert to our format
    imported_games = []
    for g in games_raw:
        try:
            # Try common Spordle field names
            team_a = (g.get("homeTeam", {}).get("name", "") or
                      g.get("home_team", {}).get("name", "") or
                      g.get("home_team_name", "") or
                      g.get("team_home", {}).get("name", "") or
                      g.get("teamA", "") or g.get("team_a", ""))
            team_b = (g.get("awayTeam", {}).get("name", "") or
                      g.get("away_team", {}).get("name", "") or
                      g.get("away_team_name", "") or
                      g.get("team_away", {}).get("name", "") or
                      g.get("teamB", "") or g.get("team_b", ""))
            goals_a = (g.get("homeScore", None) or g.get("home_score", None) or
                       g.get("score_home", None) or g.get("goals_a", None) or 0)
            goals_b = (g.get("awayScore", None) or g.get("away_score", None) or
                       g.get("score_away", None) or g.get("goals_b", None) or 0)

            if not team_a or not team_b:
                continue

            game_data = {
                "team_a": str(team_a).strip(),
                "team_b": str(team_b).strip(),
                "goals_a": int(goals_a),
                "goals_b": int(goals_b),
                "division": division,
                "ot": bool(g.get("overtime", g.get("ot", False))),
            }
            imported_games.append(game_data)
        except Exception:
            continue

    result = {
        "ok": True,
        "games_found": len(imported_games),
        "games": imported_games,
        "raw_keys": list(raw_data.keys()) if isinstance(raw_data, dict) else f"array[{len(raw_data)}]",
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
