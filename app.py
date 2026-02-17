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

# Known team names for matching in HTML parsing
KNOWN_TEAMS = [
    "Laval Rocket Jr", "Adirondack Jr Wings", "Sherbrooke Harfangs",
    "Petits Canadiens", "Korea Zenith Avengers", "Montreal National",
    "Nord Selects", "Quebec As", "Quebec Blizzard", "Mauricie Estacades",
    "Lanaudiere Pionniers", "Lac St-Louis Lions", "Outaouais Intrepide",
    "Cascades Bois-Francs", "Mortagne Noir et Or", "Rive-Sud Coll. Francais",
]


def _spordle_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }


def _spordle_api_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "Origin": "https://page.spordle.com",
        "Referer": f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/{SPORDLE_PAGE_UUID}?scheduleId={SPORDLE_AAA_ELITE_SCHEDULE}",
    }


def _navigate(obj, dotpath: str):
    """Navigate a nested dict/list by dot-separated path."""
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


def _deep_find_lists(obj, path="", max_depth=6) -> list:
    """Recursively find all lists in a nested dict."""
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
        for i, v in enumerate(obj[:3]):
            results.extend(_deep_find_lists(v, f"{path}[{i}]", max_depth - 1))
    return results


def _deep_find_strings(obj, prefix="", max_depth=5, filter_fn=None) -> list:
    """Recursively find string values in a nested dict."""
    results = []
    if max_depth <= 0:
        return results
    if isinstance(obj, str):
        if filter_fn is None or filter_fn(obj):
            results.append({"path": prefix, "value": obj[:300]})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            cp = f"{prefix}.{k}" if prefix else k
            results.extend(_deep_find_strings(v, cp, max_depth - 1, filter_fn))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:10]):
            results.extend(_deep_find_strings(v, f"{prefix}[{i}]", max_depth - 1, filter_fn))
    return results


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


def _try_fetch_html(url: str, headers: dict, timeout: int = 15) -> str:
    """Fetch HTML from a URL."""
    try:
        req = URLRequest(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode()
    except Exception:
        pass
    return ""


def _extract_next_data(html: str) -> dict:
    """Extract __NEXT_DATA__ JSON from HTML."""
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _parse_html_tables(html: str) -> list:
    """Parse HTML tables for game/standings data. Returns list of table data."""
    tables = []
    # Find all table elements
    table_matches = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE)
    for table_html in table_matches:
        rows = []
        # Extract header row
        thead = re.search(r'<thead[^>]*>(.*?)</thead>', table_html, re.DOTALL | re.IGNORECASE)
        if thead:
            headers = re.findall(r'<th[^>]*>(.*?)</th>', thead.group(1), re.DOTALL | re.IGNORECASE)
            headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
            if headers:
                rows.append(headers)

        # Extract body rows
        tbody = re.search(r'<tbody[^>]*>(.*?)</tbody>', table_html, re.DOTALL | re.IGNORECASE)
        body_html = tbody.group(1) if tbody else table_html
        tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', body_html, re.DOTALL | re.IGNORECASE)
        for tr in tr_matches:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL | re.IGNORECASE)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if cells:
                rows.append(cells)

        if len(rows) > 1:
            tables.append(rows)
    return tables


def _parse_games_from_html(html: str, division: str) -> list:
    """Try to extract game results from rendered HTML."""
    games = []

    # Strategy 1: Look for score patterns like "Team A 3 - 2 Team B" or "3-2"
    # Spordle often renders game cards with team names and scores
    score_patterns = [
        # "Team A" ... "3" ... "-" ... "2" ... "Team B" in divs
        r'<div[^>]*class="[^"]*(?:home|visitor|team)[^"]*"[^>]*>([^<]+)</div>\s*'
        r'<div[^>]*class="[^"]*score[^"]*"[^>]*>(\d+)\s*[-–]\s*(\d+)</div>\s*'
        r'<div[^>]*class="[^"]*(?:home|visitor|team)[^"]*"[^>]*>([^<]+)</div>',
    ]
    for pattern in score_patterns:
        matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
        for m in matches:
            if len(m) == 4:
                games.append({
                    "team_a": m[0].strip(),
                    "team_b": m[3].strip(),
                    "goals_a": int(m[1]),
                    "goals_b": int(m[2]),
                    "division": division,
                    "ot": False,
                })

    # Strategy 2: Parse tables that look like schedule/results
    tables = _parse_html_tables(html)
    for table in tables:
        if len(table) < 2:
            continue
        headers = [h.lower() for h in table[0]] if table[0] else []

        # Check if this looks like a standings table
        is_standings = any(h in headers for h in ["pts", "points", "gp", "w", "l", "pj"])
        # Check if this looks like a schedule table
        is_schedule = any(h in headers for h in ["score", "result", "résultat"])

        if is_standings or is_schedule:
            for row in table[1:]:
                if len(row) >= 3:
                    # Try to find team name and score columns
                    pass  # Will process below

    # Strategy 3: Look for known team names near score numbers
    for team in KNOWN_TEAMS:
        escaped = re.escape(team)
        # Look for patterns like: "TeamName</...>...<...>3<..."
        matches = re.findall(
            rf'{escaped}[^<]*</[^>]+>\s*(?:<[^>]+>\s*)*(\d+)',
            html, re.IGNORECASE
        )
        if matches:
            logger.info(f"Found {team} with scores: {matches[:5]}")

    # Strategy 4: Extract data from React/Next.js rendered components
    # Spordle may render data in data-* attributes or JSON-LD
    json_ld = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL)
    for jl in json_ld:
        try:
            data = json.loads(jl)
            if isinstance(data, dict) and "game" in str(data).lower():
                logger.info(f"JSON-LD data found: {list(data.keys())}")
        except json.JSONDecodeError:
            pass

    return games


def _find_spordle_api_base(next_data: dict) -> list:
    """Extract API base URLs from Spordle's __NEXT_DATA__ config."""
    api_bases = []

    # Look in spordleClient for API URLs
    spordle_client = _navigate(next_data, "props.pageProps.spordleClient") or {}

    # Find all URL strings
    all_urls = _deep_find_strings(
        spordle_client, "", max_depth=8,
        filter_fn=lambda s: s.startswith("http") and len(s) < 300
    )
    for u in all_urls:
        url = u.get("value", "")
        if "api" in url.lower() or "spordle" in url.lower():
            # Strip trailing path to get base
            api_bases.append(url.rstrip("/"))

    # CSP connect-src domains
    csp = _navigate(spordle_client, "metadata.csp") or {}
    connect_src = csp.get("connect-src", [])
    if isinstance(connect_src, str):
        connect_src = [connect_src]
    for item in connect_src:
        if isinstance(item, str) and "spordle" in item.lower() and item.startswith("http"):
            api_bases.append(item.rstrip("/"))

    # Look in runtimeConfig if present
    for config_path in ["runtimeConfig.publicRuntimeConfig", "runtimeConfig",
                         "props.pageProps.__N_SSP", "props.pageProps"]:
        config = _navigate(next_data, config_path)
        if isinstance(config, dict):
            for k, v in config.items():
                if isinstance(v, str) and v.startswith("http") and "api" in v.lower():
                    api_bases.append(v.rstrip("/"))

    return list(dict.fromkeys(api_bases))  # dedupe preserving order


def _parse_spordle_games(games_raw: list, division: str) -> list:
    """Convert raw Spordle game objects to our format."""
    imported = []
    for g in games_raw:
        if not isinstance(g, dict):
            continue
        try:
            team_a = ""
            team_b = ""
            goals_a = None
            goals_b = None

            # Home team name - try many patterns
            for path in ["homeTeam.short_name", "homeTeam.name", "homeTeam.team_name",
                          "homeTeam.abbreviation", "homeTeam.display_name",
                          "home_team.short_name", "home_team.name", "home.name",
                          "home_team_name", "teamA", "team_a", "home",
                          "home_team.display_name", "teams.home.name",
                          "team_home.name", "team_home.short_name"]:
                val = _navigate(g, path)
                if val and isinstance(val, str):
                    team_a = val
                    break

            # Away team name
            for path in ["awayTeam.short_name", "awayTeam.name", "awayTeam.team_name",
                          "awayTeam.abbreviation", "awayTeam.display_name",
                          "away_team.short_name", "away_team.name", "away.name",
                          "away_team_name", "teamB", "team_b", "away",
                          "away_team.display_name", "teams.away.name",
                          "team_away.name", "team_away.short_name"]:
                val = _navigate(g, path)
                if val and isinstance(val, str):
                    team_b = val
                    break

            # Scores
            for key in ["home_score", "homeScore", "score_home", "goals_a",
                         "home_goals", "homeGoals", "result.home", "score.home",
                         "home_team_score", "homeTeamScore"]:
                val = _navigate(g, key)
                if val is not None:
                    try:
                        goals_a = int(val)
                    except (ValueError, TypeError):
                        continue
                    break
            for key in ["away_score", "awayScore", "score_away", "goals_b",
                         "away_goals", "awayGoals", "result.away", "score.away",
                         "away_team_score", "awayTeamScore"]:
                val = _navigate(g, key)
                if val is not None:
                    try:
                        goals_b = int(val)
                    except (ValueError, TypeError):
                        continue
                    break

            if not team_a or not team_b:
                continue
            if goals_a is None or goals_b is None:
                continue

            ot = bool(g.get("overtime") or g.get("ot") or g.get("is_overtime")
                      or g.get("period", 0) > 3
                      or (g.get("result", {}) or {}).get("overtime")
                      or g.get("went_to_overtime"))

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


@app.get("/api/spordle/raw")
def spordle_raw(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE),
    tab: str = Query(default="standings", description="Tab: standings, schedule, playerstats"),
):
    """
    Dump the FULL __NEXT_DATA__ from the Spordle page.
    Use this to see exactly what data Spordle provides server-side.
    """
    headers = _spordle_headers()
    page_url = (
        f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/"
        f"{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}&tab={tab}"
    )

    result = {"page_url": page_url, "tab": tab}

    try:
        html = _try_fetch_html(page_url, headers)
        if not html:
            return {"ok": False, "error": "Failed to fetch Spordle page"}

        result["html_size"] = len(html)

        # Extract __NEXT_DATA__
        next_data = _extract_next_data(html)
        if next_data:
            result["buildId"] = next_data.get("buildId")
            result["page"] = next_data.get("page")
            result["query"] = next_data.get("query")

            # Full pageProps (this is where the data lives)
            page_props = _navigate(next_data, "props.pageProps") or {}
            result["pageProps_keys"] = list(page_props.keys()) if isinstance(page_props, dict) else str(type(page_props))

            # Show ALL lists found (these are likely data arrays)
            all_lists = _deep_find_lists(next_data)
            result["all_data_lists"] = all_lists

            # spordleClient config
            spordle_client = _navigate(next_data, "props.pageProps.spordleClient") or {}
            if spordle_client:
                result["spordleClient_keys"] = list(spordle_client.keys())
                # Show every key (except metadata which is huge)
                for k, v in spordle_client.items():
                    if k == "metadata":
                        # Just show metadata keys and CSP
                        if isinstance(v, dict):
                            result["metadata_keys"] = list(v.keys())
                            csp = v.get("csp", {})
                            if isinstance(csp, dict):
                                result["csp_connect_src"] = csp.get("connect-src", [])
                        continue
                    # Truncate large values but show structure
                    result[f"sc_{k}"] = _truncate_for_debug(v)

            # Show API base URLs discovered
            api_bases = _find_spordle_api_base(next_data)
            result["discovered_api_bases"] = api_bases

            # Show first 3 items of each list for inspection
            for linfo in all_lists[:10]:
                path = linfo["path"]
                data = _navigate(next_data, path)
                if data and isinstance(data, list) and len(data) > 0:
                    sample = data[0] if isinstance(data[0], dict) else data[:3]
                    result[f"sample_{path}"] = _truncate_for_debug(sample)

        else:
            result["error"] = "No __NEXT_DATA__ found in HTML"

        # Also show any non-Next.js embedded data
        for pattern_name, pattern_re in [
            ("window_state", r'window\.__(?:INITIAL_STATE|DATA|STORE|PRELOADED)__\s*=\s*({.*?});'),
            ("window_config", r'window\.(?:config|CONFIG|APP_CONFIG)\s*=\s*({.*?});'),
        ]:
            match = re.search(pattern_re, html, re.DOTALL)
            if match:
                try:
                    result[pattern_name] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    result[f"{pattern_name}_raw"] = match.group(1)[:500]

        # Show rendered HTML stats (what's actually on the page)
        tables = _parse_html_tables(html)
        result["html_tables_found"] = len(tables)
        for i, table in enumerate(tables[:5]):
            result[f"table_{i}_headers"] = table[0] if table else []
            result[f"table_{i}_rows"] = len(table) - 1
            if len(table) > 1:
                result[f"table_{i}_sample_row"] = table[1]

    except Exception as e:
        result["error"] = str(e)

    return result


def _truncate_for_debug(obj, max_depth=3, max_str=200):
    """Truncate deep objects for debug display."""
    if max_depth <= 0:
        return f"<{type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:max_str] + ("..." if len(obj) > max_str else "")
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        return [_truncate_for_debug(obj[0], max_depth - 1, max_str)] + (
            [f"... +{len(obj)-1} more"] if len(obj) > 1 else []
        )
    if isinstance(obj, dict):
        result = {}
        for k, v in list(obj.items())[:15]:
            result[k] = _truncate_for_debug(v, max_depth - 1, max_str)
        if len(obj) > 15:
            result["__more__"] = f"+{len(obj)-15} keys"
        return result
    return str(obj)[:max_str]


@app.get("/api/spordle/debug")
def debug_spordle(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE),
):
    """Debug endpoint: show what Spordle returns so we can find the game data."""
    headers = _spordle_headers()
    result = {}

    # Fetch all 3 tabs and compare what data is in each
    for tab in ["standings", "schedule"]:
        page_url = (
            f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/"
            f"{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}&tab={tab}"
        )
        try:
            html = _try_fetch_html(page_url, headers)
            if not html:
                result[f"{tab}_error"] = "Failed to fetch"
                continue

            result[f"{tab}_html_size"] = len(html)

            next_data = _extract_next_data(html)
            if next_data:
                result[f"{tab}_buildId"] = next_data.get("buildId")

                page_props = _navigate(next_data, "props.pageProps") or {}
                result[f"{tab}_pageProps_keys"] = list(page_props.keys()) if isinstance(page_props, dict) else "none"

                all_lists = _deep_find_lists(next_data)
                result[f"{tab}_lists"] = all_lists

                # Look for game-related data
                for linfo in all_lists:
                    keys_str = " ".join(linfo.get("sample_keys", [])).lower()
                    if any(k in keys_str for k in ["team", "home", "away", "score", "goal",
                                                    "name", "pts", "points", "gp", "win"]):
                        path = linfo["path"]
                        data = _navigate(next_data, path)
                        if data and isinstance(data, list):
                            result[f"{tab}_data_{path}"] = data[:2]

                # API bases from config
                if tab == "standings":
                    api_bases = _find_spordle_api_base(next_data)
                    result["api_bases"] = api_bases

            # Parse HTML tables
            tables = _parse_html_tables(html)
            result[f"{tab}_tables"] = len(tables)
            for i, table in enumerate(tables[:3]):
                result[f"{tab}_table{i}_headers"] = table[0] if table else []
                result[f"{tab}_table{i}_rows"] = len(table) - 1
                if len(table) > 1:
                    result[f"{tab}_table{i}_sample"] = table[1:3]

        except Exception as e:
            result[f"{tab}_error"] = str(e)

    return result


@app.get("/api/spordle/scan-js")
def scan_spordle_js(
    schedule_id: str = Query(default=SPORDLE_AAA_ELITE_SCHEDULE),
):
    """Scan Spordle JS chunks to find the actual API endpoints used for game data."""
    headers = _spordle_headers()
    page_url = (
        f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/"
        f"{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}"
    )

    result = {"api_endpoints": [], "schedule_patterns": [], "fetch_calls": [], "full_api_urls": []}

    try:
        html = _try_fetch_html(page_url, headers)
        if not html:
            return {"ok": False, "error": "Failed to fetch Spordle page"}

        # Find JS chunk URLs
        js_chunks = re.findall(r'(/_next/static/[^"\']+\.js)', html)
        result["chunks_found"] = len(js_chunks)

        for chunk_path in js_chunks[:15]:
            chunk_url = f"https://page.spordle.com{chunk_path}"
            try:
                js_code = _try_fetch_html(chunk_url, headers, timeout=8)
                if not js_code:
                    continue

                # API endpoint patterns
                for p in re.findall(r'["\'](/(?:api|v[12])/[^"\']{3,60})["\']', js_code):
                    if p not in result["api_endpoints"]:
                        result["api_endpoints"].append(p)

                # Full URLs with api/spordle
                for u in re.findall(r'["\']?(https?://[^"\']*(?:api|spordle)[^"\']{0,100})["\']?', js_code):
                    if u not in result["full_api_urls"]:
                        result["full_api_urls"].append(u)

                # Schedule/game patterns
                for p in re.findall(r'["\']([^"\']*(?:schedule|game|standing|classement|score)[^"\']{0,80})["\']', js_code, re.IGNORECASE):
                    if len(p) < 120 and p not in result["schedule_patterns"]:
                        result["schedule_patterns"].append(p)

                # fetch/axios calls
                for f in re.findall(r'(?:fetch|axios|\.get|\.post)\s*\(\s*[`"\']([^`"\']{5,120})[`"\']', js_code):
                    if f not in result["fetch_calls"]:
                        result["fetch_calls"].append(f)

                # Look for concatenated URL patterns like: baseUrl + "/schedules/" + id + "/games"
                for m in re.findall(r'["\']([/a-z0-9_-]*(?:schedule|game|standing)[/a-z0-9_-]*)["\']', js_code, re.IGNORECASE):
                    if m not in result.get("url_fragments", []):
                        result.setdefault("url_fragments", []).append(m)

            except Exception:
                continue

        # Limit results
        for key in result:
            if isinstance(result[key], list) and len(result[key]) > 30:
                result[key] = result[key][:30]

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
    Fetch game data from Spordle. Tries multiple strategies:
    1. Parse rendered HTML tables from standings/schedule tabs
    2. Extract data from __NEXT_DATA__ server-side props
    3. Discover and call Spordle API endpoints
    4. Next.js _next/data route
    """
    html_headers = _spordle_headers()
    api_headers = _spordle_api_headers()
    tried = []
    games_raw = []
    debug_info = {}

    base_url = (
        f"https://page.spordle.com/{SPORDLE_ORG}/schedule-stats-standings/"
        f"{SPORDLE_PAGE_UUID}?scheduleId={schedule_id}"
    )

    # ── Strategy 1: Fetch schedule tab and parse HTML + __NEXT_DATA__ ──
    for tab in ["schedule", "standings"]:
        if games_raw:
            break

        page_url = f"{base_url}&tab={tab}"
        tried.append(page_url)

        html = _try_fetch_html(page_url, html_headers)
        if not html:
            debug_info[f"{tab}_fetch"] = "failed"
            continue

        debug_info[f"{tab}_html_size"] = len(html)

        # Try HTML table parsing
        html_games = _parse_games_from_html(html, division)
        if html_games:
            debug_info["source"] = f"html_tables_{tab}"
            games_raw = html_games
            break

        # Try __NEXT_DATA__
        next_data = _extract_next_data(html)
        if not next_data:
            continue

        debug_info[f"{tab}_buildId"] = next_data.get("buildId")

        # Search all lists in __NEXT_DATA__ for game-like data
        all_lists = _deep_find_lists(next_data)
        debug_info[f"{tab}_lists_found"] = len(all_lists)

        for linfo in all_lists:
            keys_str = " ".join(linfo.get("sample_keys", [])).lower()
            if any(k in keys_str for k in ["team", "home", "away", "score", "goal"]):
                potential = _navigate(next_data, linfo["path"]) or []
                parsed = _parse_spordle_games(potential, division)
                if parsed:
                    games_raw = potential
                    debug_info["source"] = f"next_data_{tab}"
                    debug_info["data_path"] = linfo["path"]
                    break

        # Also check for broader list patterns (team name, points, etc.)
        if not games_raw:
            for linfo in all_lists:
                keys_str = " ".join(linfo.get("sample_keys", [])).lower()
                if any(k in keys_str for k in ["name", "pts", "points", "gp", "wins",
                                                "short_name", "team_name"]):
                    data = _navigate(next_data, linfo["path"]) or []
                    if data and isinstance(data, list) and len(data) > 0:
                        debug_info[f"{tab}_potential_data_{linfo['path']}"] = [
                            data[0] if isinstance(data[0], dict) else data[:3]
                        ]

    # ── Strategy 2: Try API endpoints discovered from config ──
    if not games_raw:
        # Need to fetch the page once to get the config
        html = _try_fetch_html(base_url, html_headers)
        if html:
            next_data = _extract_next_data(html)
            if next_data:
                api_bases = _find_spordle_api_base(next_data)
                debug_info["discovered_api_bases"] = api_bases

                # Try various API paths on each discovered base
                api_paths = [
                    f"/api/v1/schedules/{schedule_id}/games",
                    f"/api/v2/schedules/{schedule_id}/games",
                    f"/api/v1/public/schedules/{schedule_id}/games",
                    f"/api/v1/schedule/{schedule_id}/games",
                    f"/api/v1/games?scheduleId={schedule_id}",
                    f"/api/v2/page-tree/{SPORDLE_ORG}/schedules/{schedule_id}/games",
                    f"/schedules/{schedule_id}/games",
                    f"/public/schedules/{schedule_id}/games",
                    f"/api/v1/standings?scheduleId={schedule_id}",
                    f"/api/v1/schedule-standings/{schedule_id}",
                    f"/api/v1/results?scheduleId={schedule_id}",
                    f"/api/v1/scores?scheduleId={schedule_id}",
                ]

                for base in api_bases:
                    if games_raw:
                        break
                    for path in api_paths:
                        url = base + path
                        tried.append(url)
                        data, err = _try_fetch_json(url, api_headers)
                        if data:
                            debug_info["api_hit"] = url
                            if isinstance(data, list) and len(data) > 0:
                                games_raw = data
                            elif isinstance(data, dict):
                                debug_info.setdefault("api_responses", []).append({
                                    "url": url,
                                    "keys": list(data.keys())[:20],
                                })
                                for li in _deep_find_lists(data):
                                    ks = " ".join(li.get("sample_keys", [])).lower()
                                    if any(k in ks for k in ["team", "home", "away", "score", "goal", "name"]):
                                        games_raw = _navigate(data, li["path"]) or []
                                        break
                            if games_raw:
                                break

                # ── Strategy 3: Next.js _next/data route ──
                if not games_raw:
                    build_id = next_data.get("buildId", "")
                    if build_id:
                        for lang in ["en", "fr"]:
                            data_url = (
                                f"https://page.spordle.com/_next/data/{build_id}/{lang}/"
                                f"{SPORDLE_ORG}/schedule-stats-standings/"
                                f"{SPORDLE_PAGE_UUID}.json?scheduleId={schedule_id}&tab=schedule"
                            )
                            tried.append(data_url)
                            data, err = _try_fetch_json(data_url, html_headers)
                            if data:
                                debug_info["next_data_hit"] = data_url
                                for li in _deep_find_lists(data):
                                    ks = " ".join(li.get("sample_keys", [])).lower()
                                    if any(k in ks for k in ["team", "home", "away", "score"]):
                                        games_raw = _navigate(data, li["path"]) or []
                                        break
                                if games_raw:
                                    break

    # ── Strategy 4: Try common Spordle API patterns directly ──
    if not games_raw:
        direct_apis = [
            f"https://page-api.spordle.com/api/v2/page-tree/{SPORDLE_ORG}/schedule/{schedule_id}/games",
            f"https://page-api.spordle.com/api/v1/schedules/{schedule_id}/games",
            f"https://api.spordle.com/page/api/v1/public/schedule/{schedule_id}/games",
            f"https://play-api.spordle.com/api/v1/public/schedules/{schedule_id}/games",
        ]
        for url in direct_apis:
            tried.append(url)
            data, err = _try_fetch_json(url, api_headers)
            if data:
                debug_info["direct_api_hit"] = url
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

    # ── Convert to our format ──
    imported_games = _parse_spordle_games(games_raw, division) if games_raw else []

    if not imported_games:
        return {
            "ok": False,
            "error": "Could not extract game data from Spordle. The data may be loaded entirely client-side.",
            "tried_count": len(tried),
            "debug": debug_info,
            "hint": "Visit /api/spordle/raw?tab=schedule to see what Spordle returns, or open the Spordle page in your browser DevTools (Network tab) to find the API endpoint.",
            "spordle_url": base_url,
            "manual_import": "You can paste game data via the web UI or POST to /api/games/bulk",
        }

    result = {
        "ok": True,
        "games_found": len(imported_games),
        "source": debug_info.get("source", "api"),
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
