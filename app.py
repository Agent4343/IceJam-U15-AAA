from __future__ import annotations
import re
import uuid
import random
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from functools import cmp_to_key

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = "https://icejam.ca"
STANDINGS_URL = f"{BASE}/standings/"
SCHEDULE_URL = f"{BASE}/schedule/"
DEFAULT_TEAM = "Eastern Hitmen"

# All 26 teams from IceJam U15 AAA 2026
TEAMS = [
    "Charlottetown Islanders",
    "Dartmouth Whalers",
    "Dieppe Flyers",
    "Eastern AAA U15 Knights",
    "Eastern Express",
    "Eastern Hitmen",
    "Eastern Thunder",
    "Fredericton Caps",
    "Halifax Wolverines",
    "Harbour Rage",
    "Joneljim Cougars",
    "Martello Wealth Bandits",
    "Mid-Isle Matrix",
    "Moncton Hawks",
    "Northern Rivermen",
    "Ottawa Jr 67's",
    "Prince County Warriors",
    "Southern Rangers",
    "The Gulls",
    "The Novas",
    "The Rangers",
    "Tri-Pen Osprey",
    "Truro Bearcats",
    "Valley Wildcats",
    "WearWell Bombers",
    "Western Hurricanes",
]

# Browser headers to avoid being blocked (no Accept-Encoding to get plain text)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

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
    first_goal_team: str = ""  # Which team scored first
    first_goal_time_sec: Optional[int] = None  # Time in seconds when first goal was scored
    game_number: int = 0  # Order of game in tournament (for first goal tiebreaker)


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
    first_goal_time: Optional[int] = None  # Earliest first goal in tournament (game_number * 10000 + seconds)

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
    first_goal_team: str = ""  # "a" or "b" or team name
    first_goal_time_sec: Optional[int] = None


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def points_for_game(gf: int, ga: int, ot: bool) -> Tuple[int, int, int, int, int, int]:
    """Returns (pts, w, l, t, otw, otl) for a team."""
    if gf == ga:
        return (1, 0, 0, 1, 0, 0)
    if gf > ga:
        return (2, 1, 0, 0, 1 if ot else 0, 0)
    return (1, 0, 1, 0, 0, 1) if ot else (0, 0, 1, 0, 0, 0)


def get_head_to_head(team1: str, team2: str, games: Dict[str, Game]) -> Tuple[int, int, int, str, str]:
    """
    Get head-to-head record between two teams.
    Returns (team1_pts, team2_pts, games_played, first_goal_winner, first_goal_team)
    """
    team1_pts = 0
    team2_pts = 0
    games_played = 0
    first_goal_winner = ""  # Team that scored first in h2h game

    for game in games.values():
        if (game.team_a == team1 and game.team_b == team2) or \
           (game.team_a == team2 and game.team_b == team1):
            games_played += 1

            if game.team_a == team1:
                pts1, _, _, _, _, _ = points_for_game(game.goals_a, game.goals_b, game.ot)
                pts2, _, _, _, _, _ = points_for_game(game.goals_b, game.goals_a, game.ot)
                team1_pts += pts1
                team2_pts += pts2
                # First goal in h2h
                if game.first_goal_team:
                    if game.first_goal_team.lower() == "a" or game.first_goal_team == team1:
                        first_goal_winner = team1
                    elif game.first_goal_team.lower() == "b" or game.first_goal_team == team2:
                        first_goal_winner = team2
            else:
                pts2, _, _, _, _, _ = points_for_game(game.goals_a, game.goals_b, game.ot)
                pts1, _, _, _, _, _ = points_for_game(game.goals_b, game.goals_a, game.ot)
                team1_pts += pts1
                team2_pts += pts2
                # First goal in h2h
                if game.first_goal_team:
                    if game.first_goal_team.lower() == "a" or game.first_goal_team == team2:
                        first_goal_winner = team2
                    elif game.first_goal_team.lower() == "b" or game.first_goal_team == team1:
                        first_goal_winner = team1

    return (team1_pts, team2_pts, games_played, first_goal_winner, first_goal_winner)


def compare_two_teams(t1: TeamStats, t2: TeamStats, games: Dict[str, Game]) -> int:
    """
    Compare two teams using all 9 tiebreaker rules.
    Returns: -1 if t1 ranks higher, 1 if t2 ranks higher, 0 if still tied
    """
    # Primary: Points
    if t1.pts != t2.pts:
        return -1 if t1.pts > t2.pts else 1

    # Tiebreaker 1: Head-to-head
    h2h = get_head_to_head(t1.name, t2.name, games)
    t1_h2h_pts, t2_h2h_pts, games_played, first_goal_winner, _ = h2h
    if games_played > 0 and t1_h2h_pts != t2_h2h_pts:
        return -1 if t1_h2h_pts > t2_h2h_pts else 1

    # Tiebreaker 2: Most wins
    if t1.w != t2.w:
        return -1 if t1.w > t2.w else 1

    # Tiebreaker 3: Goal average
    t1_ga = t1.goal_average()
    t2_ga = t2.goal_average()
    if abs(t1_ga - t2_ga) > 0.0001:
        return -1 if t1_ga > t2_ga else 1

    # Tiebreaker 4: Fewest goals against
    if t1.ga != t2.ga:
        return -1 if t1.ga < t2.ga else 1

    # Tiebreaker 5: Most goals for
    if t1.gf != t2.gf:
        return -1 if t1.gf > t2.gf else 1

    # Tiebreaker 6: Fewest penalty minutes
    if t1.pim != t2.pim:
        return -1 if t1.pim < t2.pim else 1

    # Tiebreaker 7: First goal in head-to-head game
    if first_goal_winner:
        if first_goal_winner == t1.name:
            return -1
        elif first_goal_winner == t2.name:
            return 1

    # Tiebreaker 8: Fastest first goal of tournament
    t1_first = t1.first_goal_time if t1.first_goal_time is not None else float('inf')
    t2_first = t2.first_goal_time if t2.first_goal_time is not None else float('inf')
    if t1_first != t2_first:
        return -1 if t1_first < t2_first else 1

    # Tiebreaker 9: Coin toss (random)
    return random.choice([-1, 1])


def get_record_among_tied(teams: List[TeamStats], games: Dict[str, Game]) -> Dict[str, int]:
    """Get points earned only in games between the tied teams."""
    team_names = {t.name for t in teams}
    points = {t.name: 0 for t in teams}

    for game in games.values():
        if game.team_a in team_names and game.team_b in team_names:
            pts_a, _, _, _, _, _ = points_for_game(game.goals_a, game.goals_b, game.ot)
            pts_b, _, _, _, _, _ = points_for_game(game.goals_b, game.goals_a, game.ot)
            points[game.team_a] += pts_a
            points[game.team_b] += pts_b

    return points


def get_wins_among_tied(teams: List[TeamStats], games: Dict[str, Game]) -> Dict[str, int]:
    """Get wins earned only in games between the tied teams."""
    team_names = {t.name for t in teams}
    wins = {t.name: 0 for t in teams}

    for game in games.values():
        if game.team_a in team_names and game.team_b in team_names:
            if game.goals_a > game.goals_b:
                wins[game.team_a] += 1
            elif game.goals_b > game.goals_a:
                wins[game.team_b] += 1

    return wins


def sort_tied_group(teams: List[TeamStats], games: Dict[str, Game]) -> List[TeamStats]:
    """
    Sort a group of teams that are tied in points.
    Handles both 2-team and 3+ team tiebreaker scenarios.
    """
    if len(teams) <= 1:
        return teams

    if len(teams) == 2:
        # Use 2-team tiebreaker rules
        result = compare_two_teams(teams[0], teams[1], games)
        if result <= 0:
            return teams
        else:
            return [teams[1], teams[0]]

    # 3+ teams tied - use multi-team tiebreaker rules
    # Step 1: Record among tied teams
    h2h_points = get_record_among_tied(teams, games)

    # Check if this separates anyone
    max_h2h = max(h2h_points.values())
    min_h2h = min(h2h_points.values())

    if max_h2h != min_h2h:
        # Sort by h2h points first, then recursively sort remaining ties
        teams_sorted = sorted(teams, key=lambda t: h2h_points[t.name], reverse=True)

        # Group by h2h points and recursively sort
        result = []
        i = 0
        while i < len(teams_sorted):
            current_pts = h2h_points[teams_sorted[i].name]
            group = [teams_sorted[i]]
            j = i + 1
            while j < len(teams_sorted) and h2h_points[teams_sorted[j].name] == current_pts:
                group.append(teams_sorted[j])
                j += 1
            result.extend(sort_tied_group(group, games))
            i = j
        return result

    # Step 2: Most wins among tied teams
    h2h_wins = get_wins_among_tied(teams, games)
    max_wins = max(h2h_wins.values())
    min_wins = min(h2h_wins.values())

    if max_wins != min_wins:
        teams_sorted = sorted(teams, key=lambda t: h2h_wins[t.name], reverse=True)
        result = []
        i = 0
        while i < len(teams_sorted):
            current_wins = h2h_wins[teams_sorted[i].name]
            group = [teams_sorted[i]]
            j = i + 1
            while j < len(teams_sorted) and h2h_wins[teams_sorted[j].name] == current_wins:
                group.append(teams_sorted[j])
                j += 1
            result.extend(sort_tied_group(group, games))
            i = j
        return result

    # Step 3-8: Use remaining criteria (same as 2-team but applied to all)
    def compare_multi(t1: TeamStats, t2: TeamStats) -> int:
        # Step 3: Goal average (all games)
        t1_ga = t1.goal_average()
        t2_ga = t2.goal_average()
        if abs(t1_ga - t2_ga) > 0.0001:
            return -1 if t1_ga > t2_ga else 1

        # Step 4: Fewest goals against
        if t1.ga != t2.ga:
            return -1 if t1.ga < t2.ga else 1

        # Step 5: Most goals for
        if t1.gf != t2.gf:
            return -1 if t1.gf > t2.gf else 1

        # Step 6: Fewest penalty minutes
        if t1.pim != t2.pim:
            return -1 if t1.pim < t2.pim else 1

        # Step 7: Fastest first goal of tournament
        t1_first = t1.first_goal_time if t1.first_goal_time is not None else float('inf')
        t2_first = t2.first_goal_time if t2.first_goal_time is not None else float('inf')
        if t1_first != t2_first:
            return -1 if t1_first < t2_first else 1

        # Step 8: Coin toss
        return random.choice([-1, 1])

    return sorted(teams, key=cmp_to_key(compare_multi))


def calculate_standings() -> List[dict]:
    """Calculate standings from all games using full tiebreaker rules."""
    if not games_db:
        return []

    stats: Dict[str, TeamStats] = {}
    game_counter = 0

    # Sort games by game_number to process in order
    sorted_games = sorted(games_db.values(), key=lambda g: g.game_number)

    for game in sorted_games:
        game_counter += 1

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

        # Track first goal of tournament for each team
        if game.first_goal_team and game.first_goal_time_sec is not None:
            # Calculate tournament time (game_number * large number + seconds)
            tournament_time = game.game_number * 100000 + game.first_goal_time_sec

            first_goal_team_name = ""
            if game.first_goal_team.lower() == "a":
                first_goal_team_name = game.team_a
            elif game.first_goal_team.lower() == "b":
                first_goal_team_name = game.team_b
            else:
                first_goal_team_name = game.first_goal_team

            if first_goal_team_name == game.team_a:
                if team_a.first_goal_time is None or tournament_time < team_a.first_goal_time:
                    team_a.first_goal_time = tournament_time
            elif first_goal_team_name == game.team_b:
                if team_b.first_goal_time is None or tournament_time < team_b.first_goal_time:
                    team_b.first_goal_time = tournament_time

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

    # Group teams by points
    teams_list = list(stats.values())
    points_groups: Dict[int, List[TeamStats]] = {}
    for team in teams_list:
        if team.pts not in points_groups:
            points_groups[team.pts] = []
        points_groups[team.pts].append(team)

    # Sort each group using full tiebreaker rules
    sorted_teams = []
    for pts in sorted(points_groups.keys(), reverse=True):
        group = points_groups[pts]
        sorted_group = sort_tied_group(group, games_db)
        sorted_teams.extend(sorted_group)

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


def scrape_icejam() -> Dict:
    """Scrape standings data from icejam.ca/standings/"""
    try:
        logger.info(f"Fetching {STANDINGS_URL}")
        response = requests.get(STANDINGS_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        standings_data = []

        tables = soup.find_all("table")
        logger.info(f"Found {len(tables)} tables")

        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 8:
                    team_name = cells[0].get_text(strip=True)
                    if team_name.upper() == "TEAM":
                        continue
                    try:
                        standings_data.append({
                            "team": team_name,
                            "gp": int(cells[2].get_text(strip=True) or 0),
                            "w": int(cells[3].get_text(strip=True) or 0),
                            "l": int(cells[4].get_text(strip=True) or 0),
                            "otl": int(cells[5].get_text(strip=True) or 0),
                            "pts": int(cells[6].get_text(strip=True) or 0),
                            "gf": int(cells[7].get_text(strip=True) or 0),
                            "ga": int(cells[8].get_text(strip=True) or 0),
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse row: {e}")

        return {
            "ok": True,
            "url": STANDINGS_URL,
            "teams_found": len(standings_data),
            "standings": standings_data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Scrape error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "url": STANDINGS_URL
        }


def scrape_schedule(team: str = DEFAULT_TEAM) -> Dict:
    """Scrape schedule data from icejam.ca/schedule/"""
    try:
        logger.info(f"Fetching {SCHEDULE_URL}")
        response = requests.get(SCHEDULE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text
        schedule_data = []

        # Extract the json = [...] JavaScript variable
        import json as json_lib
        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)

        if json_match:
            try:
                games_json = json_lib.loads(json_match.group(1))
                logger.info(f"Found {len(games_json)} games in JSON")

                # Filter for games containing the tracked team
                team_lower = team.lower()
                search_terms = [team_lower]
                # Also search for short name like "Hitmen"
                if "hitmen" in team_lower:
                    search_terms.append("hitmen")

                for game in games_json:
                    home = (game.get("h_n") or "").lower()
                    visitor = (game.get("v_n") or "").lower()

                    # Check if team is playing
                    is_team_game = any(term in home or term in visitor for term in search_terms)

                    if is_team_game:
                        # Determine opponent and home/away
                        if any(term in home for term in search_terms):
                            opponent = game.get("v_n", "TBD")
                            location = "vs"  # Home game
                        else:
                            opponent = game.get("h_n", "TBD")
                            location = "@"  # Away game

                        schedule_data.append({
                            "game_num": str(game.get("gn", "")),
                            "opponent": opponent,
                            "location": location,
                            "time": game.get("gt3", ""),
                            "date": game.get("gdl", ""),
                            "rink": game.get("rn", ""),
                            "league": game.get("lg_n", "")
                        })

            except json_lib.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")

        # Sort by game number
        schedule_data.sort(key=lambda x: int(x["game_num"]) if x["game_num"].isdigit() else 0)

        return {
            "ok": True,
            "url": SCHEDULE_URL,
            "team": team,
            "games_found": len(schedule_data),
            "schedule": schedule_data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Schedule scrape error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "url": SCHEDULE_URL
        }


# ============ PAGE ROUTES ============

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_team": DEFAULT_TEAM, "teams": TEAMS}
    )


@app.get("/rules", response_class=HTMLResponse)
def rules(request: Request):
    return templates.TemplateResponse(
        "rules.html",
        {"request": request}
    )


# ============ API ROUTES ============

@app.get("/api/teams")
def get_teams():
    """Get list of all teams."""
    return {"ok": True, "teams": TEAMS}


@app.get("/api/standings")
def standings(team: str = Query(DEFAULT_TEAM)):
    all_standings = calculate_standings()

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
    game_number = len(games_db) + 1

    # Determine first goal team
    first_goal_team = ""
    if game.first_goal_team:
        if game.first_goal_team.lower() in ["a", "home"]:
            first_goal_team = "a"
        elif game.first_goal_team.lower() in ["b", "away"]:
            first_goal_team = "b"
        else:
            first_goal_team = game.first_goal_team

    new_game = Game(
        game_id=game_id,
        team_a=norm(game.team_a),
        team_b=norm(game.team_b),
        goals_a=game.goals_a,
        goals_b=game.goals_b,
        ot=game.ot,
        pim_a=game.pim_a,
        pim_b=game.pim_b,
        first_goal_team=first_goal_team,
        first_goal_time_sec=game.first_goal_time_sec,
        game_number=game_number
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


@app.get("/api/scrape")
def scrape():
    """Scrape standings from icejam.ca/standings/"""
    return scrape_icejam()


@app.get("/api/schedule")
def schedule(team: str = Query(DEFAULT_TEAM)):
    """Scrape schedule from icejam.ca/schedule/"""
    return scrape_schedule(team)


@app.get("/api/debug-schedule")
def debug_schedule():
    """Debug: show raw HTML from icejam.ca/schedule/"""
    try:
        response = requests.get(SCHEDULE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text
        # Find sections containing "Hitmen"
        hitmen_sections = []
        lines = html.split('\n')
        for i, line in enumerate(lines):
            if 'hitmen' in line.lower():
                # Get surrounding context (5 lines before and after)
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                hitmen_sections.append({
                    "line": i,
                    "context": '\n'.join(lines[start:end])
                })

        return {
            "ok": True,
            "total_length": len(html),
            "hitmen_mentions": len(hitmen_sections),
            "sections": hitmen_sections[:5]  # First 5 matches
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
