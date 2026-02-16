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

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field, field_validator

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


@app.delete("/api/games")
def clear_games():
    """Clear all games."""
    games_db.clear()
    return {"ok": True, "message": "All games cleared"}
