from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

BASE = "https://icejam.ca"
SCHEDULE_URL = f"{BASE}/schedule/"
DEFAULT_TEAM = "Eastern Hitmen"

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
    first_goal_team: str = ""
    first_goal_time_sec: Optional[int] = None


@dataclass
class TeamStats:
    name: str
    gp: int = 0
    w: int = 0
    l: int = 0
    t: int = 0
    otw: int = 0
    otl: int = 0
    pts: int = 0
    gf: int = 0
    ga: int = 0
    pim: int = 0
    first_goal_time: Optional[int] = None

    def goal_average(self) -> float:
        """Tournament-specific: GF / (GF + GA), not goals per game."""
        total = self.gf + self.ga
        return (self.gf / total) if total else 0.0


class GameInput(BaseModel):
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    ot: bool = False
    pim_a: int = 0
    pim_b: int = 0


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def points_for_game(gf: int, ga: int, ot: bool):
    """Returns (pts, w, l, t, otw, otl) for a team."""
    if gf == ga:
        return (1, 0, 0, 1, 0, 0)
    if gf > ga:
        return (2, 1, 0, 0, 1 if ot else 0, 0)
    return (1, 0, 1, 0, 0, 1) if ot else (0, 0, 1, 0, 0, 0)


def calculate_standings() -> List[dict]:
    """Calculate standings from all games in the database."""
    if not games_db:
        return []

    stats: Dict[str, TeamStats] = {}

    for game in games_db.values():
        # Initialize teams if not seen
        if game.team_a not in stats:
            stats[game.team_a] = TeamStats(name=game.team_a)
        if game.team_b not in stats:
            stats[game.team_b] = TeamStats(name=game.team_b)

        team_a = stats[game.team_a]
        team_b = stats[game.team_b]

        # Update games played
        team_a.gp += 1
        team_b.gp += 1

        # Apply goal differential cap (+7 max)
        diff = abs(game.goals_a - game.goals_b)
        capped_diff = min(diff, 7)

        if game.goals_a > game.goals_b:
            gf_a = game.goals_b + capped_diff
            ga_a = game.goals_b
            gf_b = game.goals_b
            ga_b = gf_a
        elif game.goals_b > game.goals_a:
            gf_b = game.goals_a + capped_diff
            ga_b = game.goals_a
            gf_a = game.goals_a
            ga_a = gf_b
        else:
            gf_a = ga_a = gf_b = ga_b = game.goals_a

        team_a.gf += gf_a
        team_a.ga += ga_a
        team_b.gf += gf_b
        team_b.ga += ga_b

        # Update PIM
        team_a.pim += game.pim_a
        team_b.pim += game.pim_b

        # Calculate points
        pts_a, w_a, l_a, t_a, otw_a, otl_a = points_for_game(game.goals_a, game.goals_b, game.ot)
        pts_b, w_b, l_b, t_b, otw_b, otl_b = points_for_game(game.goals_b, game.goals_a, game.ot)

        team_a.pts += pts_a
        team_a.w += w_a
        team_a.l += l_a
        team_a.t += t_a
        team_a.otw += otw_a
        team_a.otl += otl_a

        team_b.pts += pts_b
        team_b.w += w_b
        team_b.l += l_b
        team_b.t += t_b
        team_b.otw += otw_b
        team_b.otl += otl_b

    # Sort by: points (desc), wins (desc), goal_average (desc), goals_against (asc)
    sorted_teams = sorted(
        stats.values(),
        key=lambda t: (t.pts, t.w, t.goal_average(), -t.ga),
        reverse=True
    )

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
            "otw": team.otw,
            "otl": team.otl,
            "pts": team.pts,
            "gf": team.gf,
            "ga": team.ga,
            "pim": team.pim,
            "goal_avg": round(team.goal_average(), 3)
        })

    return standings


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_team": DEFAULT_TEAM}
    )


@app.get("/api/standings")
def standings(team: str = Query(DEFAULT_TEAM)):
    all_standings = calculate_standings()

    # Find tracked team rank
    tracked_rank = None
    for s in all_standings:
        if team.lower() in s["team"].lower():
            tracked_rank = s["rank"]
            break

    return {
        "ok": True,
        "tracked": {"team": team, "rank": tracked_rank},
        "standings": all_standings,
        "games_count": len(games_db)
    }


@app.post("/api/games")
def add_game(game: GameInput):
    """Add a new game result."""
    game_id = str(uuid.uuid4())[:8]
    new_game = Game(
        game_id=game_id,
        team_a=norm(game.team_a),
        team_b=norm(game.team_b),
        goals_a=game.goals_a,
        goals_b=game.goals_b,
        ot=game.ot,
        pim_a=game.pim_a,
        pim_b=game.pim_b
    )
    games_db[game_id] = new_game
    return {"ok": True, "game_id": game_id, "game": asdict(new_game)}


@app.get("/api/games")
def list_games():
    """List all games."""
    return {
        "ok": True,
        "count": len(games_db),
        "games": [asdict(g) for g in games_db.values()]
    }


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str):
    """Delete a game by ID."""
    if game_id not in games_db:
        raise HTTPException(status_code=404, detail="Game not found")
    del games_db[game_id]
    return {"ok": True, "deleted": game_id}


@app.delete("/api/games")
def clear_games():
    """Clear all games."""
    games_db.clear()
    return {"ok": True, "message": "All games cleared"}
