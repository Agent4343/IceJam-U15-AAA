"""
IceJam U15 AAA Standings Tracker

This app automatically tracks round robin standings for the IceJam U15 AAA tournament.
All round robin logic is automated - standings are scraped from icejam.ca and tiebreaker
rules are applied automatically to determine rankings.

TIEBREAKER LIMITATIONS:
    The following tiebreakers ARE implemented automatically:
        TB1: Head-to-head record
        TB2: Most wins
        TB3: Goal average (GF / (GF + GA))
        TB4: Fewest goals against
        TB5: Most goals for
        TB6: Fewest penalty minutes
        TB9: Alphabetical order (deterministic fallback for coin toss)

    The following tiebreakers CANNOT be determined (data not available from icejam.ca):
        TB7: First goal in head-to-head game (requires play-by-play data)
        TB8: Fastest first goal of tournament (requires play-by-play data)
"""
from __future__ import annotations
import json as json_lib
import re
import uuid
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
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = "https://icejam.ca"
STANDINGS_URL = f"{BASE}/standings/"
SCHEDULE_URL = f"{BASE}/schedule/"
DEFAULT_TEAM = "Eastern Hitman"
DEFAULT_LEAGUE = "500226"  # IceJam U15 league ID (Eastern Hitman's league)

# Multiplier for tournament time calculation (ensures game order takes precedence over time within game)
TOURNAMENT_TIME_MULTIPLIER = 100000

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
    team_a: str = Field(..., min_length=1, max_length=100, description="Home team name")
    team_b: str = Field(..., min_length=1, max_length=100, description="Away team name")
    goals_a: int = Field(..., ge=0, le=100, description="Goals scored by team A")
    goals_b: int = Field(..., ge=0, le=100, description="Goals scored by team B")
    ot: bool = False
    pim_a: int = Field(default=0, ge=0, le=500, description="Penalty minutes for team A")
    pim_b: int = Field(default=0, ge=0, le=500, description="Penalty minutes for team B")
    first_goal_team: str = Field(default="", max_length=100, description="'a' or 'b' or team name")
    first_goal_time_sec: Optional[int] = Field(default=None, ge=0, le=3600, description="Time in seconds when first goal was scored")

    @field_validator('team_b')
    @classmethod
    def teams_must_be_different(cls, v, info):
        if 'team_a' in info.data and v.strip().lower() == info.data['team_a'].strip().lower():
            raise ValueError('team_a and team_b must be different teams')
        return v


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


def get_head_to_head(team1: str, team2: str, games: Dict[str, Game]) -> Tuple[int, int, int, str]:
    """
    Get head-to-head record between two teams.
    Returns (team1_pts, team2_pts, games_played, first_goal_winner)
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

    return (team1_pts, team2_pts, games_played, first_goal_winner)


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
    t1_h2h_pts, t2_h2h_pts, games_played, first_goal_winner = h2h
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

    # Tiebreaker 9: Alphabetical order (deterministic fallback for coin toss)
    # Note: In actual tournament, this would be a coin toss. Using alphabetical for consistent results.
    return -1 if t1.name.lower() < t2.name.lower() else 1


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

        # Step 8: Alphabetical order (deterministic fallback for coin toss)
        return -1 if t1.name.lower() < t2.name.lower() else 1

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
            # Calculate tournament time (game_number * multiplier + seconds)
            tournament_time = game.game_number * TOURNAMENT_TIME_MULTIPLIER + game.first_goal_time_sec

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


def scrape_icejam(league_id: str = None, season: str = "2025") -> Dict:
    """Scrape standings data from icejam.ca using their API"""
    try:
        # Use provided league_id or default to IceJam U15
        lg = league_id or DEFAULT_LEAGUE

        # Use the getData.php API endpoint with season parameter
        api_url = f"{BASE}/teams/getData.php?todo=STAND&vol=2&w=211&a=125&s={season}&l={lg}"
        logger.info(f"Fetching {api_url}")
        response = requests.get(api_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        standings_data = []

        try:
            jsonst_data = response.json()
            logger.info(f"Got JSON response with {len(jsonst_data)} entries")

            # Extract standings from jsonStandings array inside first element
            if jsonst_data and isinstance(jsonst_data, list) and len(jsonst_data) > 0:
                standings_array = jsonst_data[0].get("jsonStandings", [])
                logger.info(f"Found {len(standings_array)} teams in jsonStandings")

                for team in standings_array:
                    # RYNA Hockey fields:
                    # ln = long name, mn = medium name, sn = short name
                    # gp, w, l, t, otl, otw, gf, ga, pts, pim
                    # pp_per, pk_per, ppg_f, shg_f, shots_f, shots_a (Offence stats)
                    team_name = team.get("ln") or team.get("mn") or team.get("sn") or ""

                    if team_name:
                        standings_data.append({
                            "team": team_name,
                            "gp": int(team.get("gp", 0) or 0),
                            "w": int(team.get("w", 0) or 0),
                            "l": int(team.get("l", 0) or 0),
                            "t": int(team.get("t", 0) or 0),
                            "otl": int(team.get("otl", 0) or 0),
                            "pts": int(team.get("pts", 0) or 0),
                            "gf": int(team.get("gf", 0) or 0),
                            "ga": int(team.get("ga", 0) or 0),
                            "pim": int(team.get("pim", 0) or 0),
                            # Offence stats (changeRep(2))
                            "pp_per": round(float(team.get("pp_per", 0) or 0), 3),
                            "pk_per": round(float(team.get("pk_per", 0) or 0), 3),
                            "ppg": int(team.get("ppg_f", 0) or 0),
                            "shg": int(team.get("shg_f", 0) or 0),
                            "shots_for": int(team.get("shots_f", 0) or 0),
                            "shots_against": int(team.get("shots_a", 0) or 0),
                        })
        except json_lib.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Fallback to HTML scraping if API fails
            return scrape_icejam_html(lg)

        # Sort by points (descending) then by goal differential
        standings_data.sort(key=lambda x: (x["pts"], x["gf"] - x["ga"]), reverse=True)

        # Add rank
        for i, team in enumerate(standings_data, 1):
            team["rank"] = i

        return {
            "ok": True,
            "url": api_url,
            "league_id": lg,
            "season": season,
            "teams_found": len(standings_data),
            "standings": standings_data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Scrape error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "url": api_url if 'api_url' in locals() else BASE
        }


def fetch_game_scores(league_id: str = None, season: str = "2025") -> Dict:
    """Fetch game scores from icejam.ca for tiebreaker calculations"""
    try:
        lg = league_id or DEFAULT_LEAGUE

        # Fetch scores from the scores page
        scores_url = f"{BASE}/scores/"
        logger.info(f"Fetching scores from {scores_url}")
        response = requests.get(scores_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text
        games = []

        # Try to extract json variable with game data
        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)
        if json_match:
            try:
                games_json = json_lib.loads(json_match.group(1))
                logger.info(f"Found {len(games_json)} games in scores JSON")

                for game in games_json:
                    # Filter by league
                    game_league = str(game.get("lg", ""))
                    if game_league != lg:
                        continue

                    home_team = game.get("h_n", "")
                    away_team = game.get("v_n", "")
                    home_score = int(game.get("hf", 0) or 0)
                    away_score = int(game.get("vf", 0) or 0)
                    game_status = game.get("gs", "")  # Game status

                    # Only include completed games
                    if home_team and away_team and game_status in ["F", "Final", "final", ""]:
                        games.append({
                            "home": home_team,
                            "away": away_team,
                            "home_score": home_score,
                            "away_score": away_score,
                            "ot": "OT" in str(game.get("gp", "")),
                            "game_num": game.get("gn", ""),
                        })
            except json_lib.JSONDecodeError as e:
                logger.error(f"JSON parse error in scores: {e}")

        return {
            "ok": True,
            "games_found": len(games),
            "games": games
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Scores fetch error: {e}")
        return {"ok": False, "error": str(e), "games": []}


def apply_tiebreakers_to_live(standings: List[Dict], games: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply tournament tiebreaker rules to live standings data.
    Returns (sorted_standings, tiebreaker_log) with details of calculations.

    LIMITATION: Not all tiebreakers can be applied with scraped data.
    The following tiebreakers ARE implemented:
        TB1: Head-to-head record (when game scores available)
        TB2: Most wins
        TB3: Goal average (GF / (GF + GA))
        TB4: Fewest goals against
        TB5: Most goals for
        TB6: Fewest penalty minutes
        TB9: Alphabetical order (deterministic fallback for coin toss)

    The following tiebreakers CANNOT be implemented (data not available):
        TB7: First goal in head-to-head game (requires play-by-play data)
        TB8: Fastest first goal of tournament (requires play-by-play data)
    """
    if not standings:
        return standings, []

    tiebreaker_log = []  # Log of tiebreaker decisions

    # Build head-to-head record (empty if no games provided)
    h2h = {}  # {(team1, team2): {team1: pts, team2: pts}}

    for game in (games or []):
        home = game["home"]
        away = game["away"]
        home_score = game["home_score"]
        away_score = game["away_score"]
        ot = game.get("ot", False)

        # Calculate points
        if home_score > away_score:
            home_pts = 2
            away_pts = 1 if ot else 0
        elif away_score > home_score:
            away_pts = 2
            home_pts = 1 if ot else 0
        else:
            home_pts = away_pts = 1

        # Store h2h record (use sorted tuple as key)
        key = tuple(sorted([home, away]))
        if key not in h2h:
            h2h[key] = {home: 0, away: 0}
        h2h[key][home] = h2h[key].get(home, 0) + home_pts
        h2h[key][away] = h2h[key].get(away, 0) + away_pts

    def compare_teams_with_log(t1: Dict, t2: Dict) -> Tuple[int, str]:
        """Compare two teams and return (result, reason)."""

        # Primary: Points
        if t1["pts"] != t2["pts"]:
            reason = f"Points: {t1['team']} ({t1['pts']}) vs {t2['team']} ({t2['pts']})"
            return (-1 if t1["pts"] > t2["pts"] else 1, reason)

        # Tiebreaker 1: Head-to-head record
        key = tuple(sorted([t1["team"], t2["team"]]))
        if key in h2h:
            t1_h2h = h2h[key].get(t1["team"], 0)
            t2_h2h = h2h[key].get(t2["team"], 0)
            if t1_h2h != t2_h2h:
                reason = f"TB1 H2H: {t1['team']} ({t1_h2h} pts) vs {t2['team']} ({t2_h2h} pts)"
                return (-1 if t1_h2h > t2_h2h else 1, reason)

        # Tiebreaker 2: Most wins
        if t1["w"] != t2["w"]:
            reason = f"TB2 Wins: {t1['team']} ({t1['w']}W) vs {t2['team']} ({t2['w']}W)"
            return (-1 if t1["w"] > t2["w"] else 1, reason)

        # Tiebreaker 3: Goal average (GF / (GF + GA))
        t1_ga = t1["gf"] / (t1["gf"] + t1["ga"]) if (t1["gf"] + t1["ga"]) > 0 else 0
        t2_ga = t2["gf"] / (t2["gf"] + t2["ga"]) if (t2["gf"] + t2["ga"]) > 0 else 0
        if abs(t1_ga - t2_ga) > 0.0001:
            reason = f"TB3 Goal Avg: {t1['team']} ({t1['gf']}/({t1['gf']}+{t1['ga']})={t1_ga:.3f}) vs {t2['team']} ({t2['gf']}/({t2['gf']}+{t2['ga']})={t2_ga:.3f})"
            return (-1 if t1_ga > t2_ga else 1, reason)

        # Tiebreaker 4: Fewest goals against
        if t1["ga"] != t2["ga"]:
            reason = f"TB4 Fewest GA: {t1['team']} ({t1['ga']} GA) vs {t2['team']} ({t2['ga']} GA)"
            return (-1 if t1["ga"] < t2["ga"] else 1, reason)

        # Tiebreaker 5: Most goals for
        if t1["gf"] != t2["gf"]:
            reason = f"TB5 Most GF: {t1['team']} ({t1['gf']} GF) vs {t2['team']} ({t2['gf']} GF)"
            return (-1 if t1["gf"] > t2["gf"] else 1, reason)

        # Tiebreaker 6: Fewest penalty minutes
        t1_pim = t1.get("pim", 0)
        t2_pim = t2.get("pim", 0)
        if t1_pim != t2_pim:
            reason = f"TB6 Fewest PIM: {t1['team']} ({t1_pim} PIM) vs {t2['team']} ({t2_pim} PIM)"
            return (-1 if t1_pim < t2_pim else 1, reason)

        # TB7-8: First goal data not available from scraped standings
        # TB9: Alphabetical order (deterministic fallback for coin toss)
        reason = f"TB9 Alphabetical: {t1['team']} vs {t2['team']} (TB7-8 require first goal data not available)"
        return (-1 if t1["team"].lower() < t2["team"].lower() else 1, reason)

    def compare_teams(t1: Dict, t2: Dict) -> int:
        result, reason = compare_teams_with_log(t1, t2)
        # Log tiebreaker decisions (only when not decided by points)
        if "TB" in reason or "TIED" in reason:
            tiebreaker_log.append(reason)
        return result

    # Sort standings using tiebreaker rules
    from functools import cmp_to_key
    sorted_standings = sorted(standings, key=cmp_to_key(compare_teams))

    # Update ranks and add tiebreaker info to each team
    for i, team in enumerate(sorted_standings, 1):
        team["rank"] = i
        # Calculate goal average for display
        total_goals = team["gf"] + team["ga"]
        team["goal_avg"] = round(team["gf"] / total_goals, 3) if total_goals > 0 else 0

    return sorted_standings, tiebreaker_log


def scrape_icejam_with_tiebreakers(league_id: str = None, season: str = "2025") -> Dict:
    """Scrape standings and apply tournament tiebreaker rules"""
    # Get standings
    standings_result = scrape_icejam(league_id, season)
    if not standings_result.get("ok"):
        return standings_result

    # Get game scores for head-to-head tiebreakers
    scores_result = fetch_game_scores(league_id, season)

    if scores_result.get("ok") and scores_result.get("games"):
        # Apply tiebreaker rules
        sorted_standings, tiebreaker_log = apply_tiebreakers_to_live(
            standings_result["standings"],
            scores_result["games"]
        )
        standings_result["standings"] = sorted_standings
        standings_result["tiebreakers_applied"] = True
        standings_result["games_used"] = scores_result["games_found"]
        standings_result["tiebreaker_log"] = tiebreaker_log
    else:
        standings_result["tiebreakers_applied"] = False
        standings_result["tiebreaker_note"] = "Could not fetch game scores for h2h tiebreakers"
        standings_result["tiebreaker_log"] = []

    return standings_result


def scrape_icejam_html(league_id: str = None) -> Dict:
    """Fallback: Scrape standings from HTML page"""
    try:
        lg = league_id or DEFAULT_LEAGUE
        url = f"{STANDINGS_URL}?lg={lg}"
        logger.info(f"Fetching {url}")
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text
        standings_data = []

        # Try to extract jsonSt variable (RYNA Hockey format)
        # Structure: jsonSt = [{"jsonStandings": [{team1}, {team2}, ...]}]
        jsonst_start = html.find('let jsonSt = [')
        if jsonst_start > 0:
            # Find the end of the JSON array by counting brackets
            start_idx = jsonst_start + len('let jsonSt = ')
            bracket_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False

            for i, char in enumerate(html[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > start_idx:
                json_str = html[start_idx:end_idx]
                try:
                    jsonst_data = json_lib.loads(json_str)
                    logger.info(f"Found jsonSt with {len(jsonst_data)} entries")

                    # Extract standings from jsonStandings array inside first element
                    if jsonst_data and isinstance(jsonst_data, list) and len(jsonst_data) > 0:
                        standings_array = jsonst_data[0].get("jsonStandings", [])
                        logger.info(f"Found {len(standings_array)} teams in jsonStandings")

                        for team in standings_array:
                            # RYNA Hockey fields:
                            # ln = long name, mn = medium name, sn = short name
                            # gp, w, l, t, otl, otw, gf, ga, pts, pim
                            team_name = team.get("ln") or team.get("mn") or team.get("sn") or ""

                            if team_name:
                                standings_data.append({
                                    "team": team_name,
                                    "gp": int(team.get("gp", 0) or 0),
                                    "w": int(team.get("w", 0) or 0),
                                    "l": int(team.get("l", 0) or 0),
                                    "t": int(team.get("t", 0) or 0),
                                    "otl": int(team.get("otl", 0) or 0),
                                    "pts": int(team.get("pts", 0) or 0),
                                    "gf": int(team.get("gf", 0) or 0),
                                    "ga": int(team.get("ga", 0) or 0),
                                    "pim": int(team.get("pim", 0) or 0),
                                })

                except json_lib.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")

        # If no JSON found, try other variable names
        if not standings_data:
            for var_name in ['standings', 'teams', 'data', 'tbl']:
                pattern = rf'{var_name}\s*=\s*(\[.*?\]);'
                match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        teams_json = json_lib.loads(match.group(1))
                        logger.info(f"Found data in '{var_name}' variable")
                        for team in teams_json:
                            team_name = team.get("t_n") or team.get("team") or team.get("name") or ""
                            if team_name:
                                standings_data.append({
                                    "team": team_name,
                                    "gp": int(team.get("gp", 0) or 0),
                                    "w": int(team.get("w", 0) or 0),
                                    "l": int(team.get("l", 0) or 0),
                                    "t": int(team.get("t", 0) or 0),
                                    "otl": int(team.get("otl", 0) or 0),
                                    "pts": int(team.get("pts", 0) or 0),
                                    "gf": int(team.get("gf", 0) or 0),
                                    "ga": int(team.get("ga", 0) or 0),
                                })
                        if standings_data:
                            break
                    except json_lib.JSONDecodeError:
                        continue

        # Fallback: Try HTML table parsing if no JSON found
        if not standings_data:
            logger.info("No JSON data found, trying HTML table parsing")
            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            logger.info(f"Found {len(tables)} HTML tables")

            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 5:
                        # Try to extract team name from first cell
                        team_cell = cells[0].get_text(strip=True)
                        if not team_cell or team_cell.upper() in ["TEAM", "RANK", "#", ""]:
                            continue

                        # Try to parse numeric values from remaining cells
                        try:
                            # Common formats: Team, GP, W, L, T/OTL, PTS, GF, GA
                            # Or: Rank, Team, GP, W, L, OTL, PTS, GF, GA
                            numeric_cells = []
                            for cell in cells[1:]:
                                try:
                                    numeric_cells.append(int(cell.get_text(strip=True) or 0))
                                except ValueError:
                                    numeric_cells.append(0)

                            if len(numeric_cells) >= 4:
                                standings_data.append({
                                    "team": team_cell,
                                    "gp": numeric_cells[0] if len(numeric_cells) > 0 else 0,
                                    "w": numeric_cells[1] if len(numeric_cells) > 1 else 0,
                                    "l": numeric_cells[2] if len(numeric_cells) > 2 else 0,
                                    "t": 0,
                                    "otl": numeric_cells[3] if len(numeric_cells) > 3 else 0,
                                    "pts": numeric_cells[4] if len(numeric_cells) > 4 else 0,
                                    "gf": numeric_cells[5] if len(numeric_cells) > 5 else 0,
                                    "ga": numeric_cells[6] if len(numeric_cells) > 6 else 0,
                                })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse row: {e}")
                            continue

                if standings_data:
                    break  # Found data in this table

        # Sort by points (descending) then by goal differential
        standings_data.sort(key=lambda x: (x["pts"], x["gf"] - x["ga"]), reverse=True)

        # Add rank
        for i, team in enumerate(standings_data, 1):
            team["rank"] = i

        return {
            "ok": True,
            "url": url,
            "league_id": lg,
            "teams_found": len(standings_data),
            "standings": standings_data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Scrape error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "url": url if 'url' in locals() else STANDINGS_URL
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
        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)

        if json_match:
            try:
                games_json = json_lib.loads(json_match.group(1))
                logger.info(f"Found {len(games_json)} games in JSON")

                # Filter for games containing the tracked team
                team_lower = team.lower()
                search_terms = [team_lower]
                # Also search for short name variations (Hitman/Hitmen)
                if "hitman" in team_lower or "hitmen" in team_lower:
                    search_terms.extend(["hitman", "hitmen"])

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
        {"request": request, "default_team": DEFAULT_TEAM}
    )


@app.get("/rules", response_class=HTMLResponse)
def rules(request: Request):
    return templates.TemplateResponse(
        "rules.html",
        {"request": request}
    )


# ============ API ROUTES ============

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
def scrape(
    league: str = Query(None, description="League ID (default: IceJam U15)"),
    season: str = Query("2025", description="Season year (default: 2025 for 2025/26)"),
    apply_rules: bool = Query(True, description="Apply tournament tiebreaker rules")
):
    """Scrape standings from icejam.ca API with optional tiebreaker rules"""
    if apply_rules:
        return scrape_icejam_with_tiebreakers(league, season)
    return scrape_icejam(league, season)


@app.get("/api/scores")
def get_scores(
    league: str = Query(None, description="League ID"),
    season: str = Query("2025", description="Season year")
):
    """Get game scores for tiebreaker calculations"""
    return fetch_game_scores(league, season)


@app.get("/api/playoff-bracket")
def playoff_bracket():
    """
    Generate playoff bracket based on scraped standings.

    This tournament uses REPLACEMENT SEEDING:
    - When a lower-ranked team beats a higher-ranked team, the lower-ranked team
      takes over the defeated team's seeding position for the rest of the tournament.
    - Example: If Rank 16 beats Rank 1, the Rank 16 team becomes the new Rank 1 seed.

    Bracket Structure:
    - Round of 16 (PO1-PO8): Top 16 teams from round robin (26 teams total)
    - Quarter-Finals (QF1-QF4): Winners re-seeded
    - Semi-Finals (SF1-SF2): Winners re-seeded
    - Final: Championship game
    """
    standings_result = scrape_icejam()

    if not standings_result.get("ok") or not standings_result.get("standings"):
        return {
            "ok": False,
            "error": "Could not fetch standings",
            "bracket": []
        }

    standings = standings_result["standings"]
    teams_count = len(standings)

    def get_team(rank):
        """Get team by rank, return None if rank exceeds available teams."""
        if rank <= teams_count:
            team = standings[rank - 1]
            return {
                "rank": rank,
                "team": team["team"],
                "record": f"{team['w']}-{team['l']}-{team['t']}",
                "pts": team["pts"]
            }
        return None

    # Round of 16 matchups (structured for replacement seeding bracket)
    # Pairings designed so QF matchups are: PO2vPO1, PO4vPO3, PO6vPO5, PO8vPO7
    round_of_16 = [
        {"game": "PO1", "high": 1, "low": 16},
        {"game": "PO2", "high": 8, "low": 9},
        {"game": "PO3", "high": 2, "low": 15},
        {"game": "PO4", "high": 7, "low": 10},
        {"game": "PO5", "high": 3, "low": 14},
        {"game": "PO6", "high": 6, "low": 11},
        {"game": "PO7", "high": 4, "low": 13},
        {"game": "PO8", "high": 5, "low": 12},
    ]

    bracket = []
    for matchup in round_of_16:
        high_team = get_team(matchup["high"])
        low_team = get_team(matchup["low"])

        if high_team and low_team:
            bracket.append({
                "game": matchup["game"],
                "round": "Round of 16",
                "high_seed": high_team,
                "low_seed": low_team,
                "matchup": f"#{matchup['high']} {high_team['team']} vs #{matchup['low']} {low_team['team']}",
                "home_team": high_team["team"],
                "note": "Higher seed is HOME (white jerseys)"
            })
        elif high_team:
            bracket.append({
                "game": matchup["game"],
                "round": "Round of 16",
                "high_seed": high_team,
                "low_seed": None,
                "matchup": f"#{matchup['high']} {high_team['team']} - BYE",
                "home_team": high_team["team"],
                "note": "Advances automatically"
            })

    # Quarter-final structure (winners with replacement seeding)
    quarter_finals = [
        {"game": "QF1", "from": ["PO2", "PO1"], "note": "Winner PO2 vs Winner PO1"},
        {"game": "QF2", "from": ["PO4", "PO3"], "note": "Winner PO4 vs Winner PO3"},
        {"game": "QF3", "from": ["PO6", "PO5"], "note": "Winner PO6 vs Winner PO5"},
        {"game": "QF4", "from": ["PO8", "PO7"], "note": "Winner PO8 vs Winner PO7"},
    ]

    # Semi-final structure
    semi_finals = [
        {"game": "SF1", "from": ["QF2", "QF1"], "note": "Winner QF2 vs Winner QF1"},
        {"game": "SF2", "from": ["QF4", "QF3"], "note": "Winner QF4 vs Winner QF3"},
    ]

    # Final
    final = {"game": "Final", "from": ["SF2", "SF1"], "note": "Winner SF2 vs Winner SF1"}

    return {
        "ok": True,
        "teams_count": teams_count,
        "teams_in_playoffs": min(teams_count, 16),
        "teams_eliminated": max(0, teams_count - 16),
        "bracket": bracket,
        "quarter_finals": quarter_finals,
        "semi_finals": semi_finals,
        "final": final,
        "replacement_seeding": {
            "enabled": True,
            "description": "When a lower seed beats a higher seed, the winner takes over the higher seed's ranking for the remainder of the tournament.",
            "example": "If Rank 16 beats Rank 1, the Rank 16 team becomes the new #1 seed."
        },
        "standings_url": STANDINGS_URL
    }


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


@app.get("/api/debug-standings")
def debug_standings():
    """Debug: show structure from icejam.ca/standings/"""
    try:
        response = requests.get(STANDINGS_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text

        # Try multiple regex patterns to find JSON data
        patterns_tried = []

        # Pattern 1: json = [...]
        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)
        patterns_tried.append({"pattern": "json = [...];", "found": bool(json_match)})

        # Pattern 2: json = [...] with greedy match up to ]];
        json_match2 = re.search(r'json\s*=\s*(\[[\s\S]*?\]\s*);', html)
        patterns_tried.append({"pattern": "json = [...] greedy", "found": bool(json_match2)})

        # Pattern 3: Look for any large array assignment
        array_matches = re.findall(r'(\w+)\s*=\s*\[', html)
        patterns_tried.append({"pattern": "var = [", "variables_found": array_matches[:20]})

        main_json_info = None
        if json_match:
            try:
                data = json_lib.loads(json_match.group(1))
                main_json_info = {
                    "found": True,
                    "count": len(data),
                    "keys": list(data[0].keys()) if data else [],
                    "sample": data[0] if data else None,
                }
            except Exception as e:
                main_json_info = {"found": True, "error": str(e), "raw_preview": json_match.group(1)[:500]}
        else:
            main_json_info = {"found": False}

        # Find all JavaScript variable assignments that look like arrays of objects
        all_json_vars = re.findall(r'(\w+)\s*=\s*(\[\{.*?\}\])\s*;', html, re.DOTALL)

        json_patterns = []
        for var_name, json_str in all_json_vars[:10]:
            try:
                data = json_lib.loads(json_str)
                json_patterns.append({
                    "name": var_name,
                    "count": len(data),
                    "keys": list(data[0].keys()) if data else [],
                    "sample": data[0] if data else None
                })
            except:
                json_patterns.append({"name": var_name, "error": "parse failed", "preview": json_str[:300]})

        # Look for Hitmen in the HTML
        hitmen_idx = html.lower().find('hitmen')
        hitmen_context = html[max(0,hitmen_idx-200):hitmen_idx+300] if hitmen_idx > 0 else "Not found"

        # Show a sample of the script tags
        script_content = []
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script")[:5]:
            text = script.get_text()[:500] if script.get_text() else ""
            if text and len(text) > 50:
                script_content.append(text)

        # Show HTML tables found
        tables_info = []
        for i, table in enumerate(soup.find_all("table")[:5]):
            rows = table.find_all("tr")
            table_data = {
                "index": i,
                "row_count": len(rows),
                "first_row": None,
                "sample_rows": []
            }
            if rows:
                # Get header or first row
                first_cells = rows[0].find_all(["th", "td"])
                table_data["first_row"] = [c.get_text(strip=True)[:30] for c in first_cells[:10]]
                # Get sample data rows
                for row in rows[1:4]:
                    cells = row.find_all(["td", "th"])
                    table_data["sample_rows"].append([c.get_text(strip=True)[:30] for c in cells[:10]])
            tables_info.append(table_data)

        # Check for iframes (data might be in iframe)
        iframes = soup.find_all("iframe")
        iframe_srcs = [iframe.get("src", "")[:100] for iframe in iframes[:5]]

        return {
            "ok": True,
            "total_length": len(html),
            "patterns_tried": patterns_tried,
            "main_json": main_json_info,
            "other_json_vars": len(all_json_vars),
            "json_patterns": json_patterns,
            "hitmen_context": hitmen_context,
            "script_samples": script_content[:3],
            "tables_found": len(soup.find_all("table")),
            "tables_info": tables_info,
            "iframes": iframe_srcs
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/debug-leagues")
def debug_leagues():
    """Debug: find available leagues from icejam.ca/standings/"""
    try:
        response = requests.get(STANDINGS_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text

        # Look for jsonSLeague variable - try multiple formats
        leagues_info = []
        sleague_raw = None

        # Try: let jsonSLeague = "..." (escaped JSON string)
        sleague_match = re.search(r'let\s+jsonSLeague\s*=\s*"([^"]*)"', html)
        if sleague_match:
            sleague_raw = sleague_match.group(1)
            try:
                # Unescape the JSON string
                unescaped = sleague_raw.replace('\\"', '"').replace('\\\\', '\\')
                leagues_info = json_lib.loads(unescaped)
            except:
                pass

        # Try: jsonSLeague = [...] (direct array)
        if not leagues_info:
            sleague_match2 = re.search(r'jsonSLeague\s*=\s*(\[.*?\]);', html, re.DOTALL)
            if sleague_match2:
                try:
                    leagues_info = json_lib.loads(sleague_match2.group(1))
                except:
                    sleague_raw = sleague_match2.group(1)[:500]

        # Look for leagueDef and leagueInNew
        league_def_match = re.search(r'leagueDef\s*=\s*(\d+)', html)
        league_def = league_def_match.group(1) if league_def_match else None

        league_in_new_match = re.search(r'leagueInNew\s*=\s*(\d+)', html)
        league_in_new = league_in_new_match.group(1) if league_in_new_match else None

        # Find all 6-digit numbers that might be league IDs
        all_league_ids = re.findall(r'\b(50\d{4})\b', html)
        unique_leagues = list(set(all_league_ids))

        # Look for "IceJam" or "U15" mentions with nearby numbers
        icejam_context = []
        for match in re.finditer(r'.{0,50}(icejam|u15).{0,50}', html, re.IGNORECASE):
            icejam_context.append(match.group(0))

        return {
            "ok": True,
            "current_league_def": league_def,
            "league_in_new": league_in_new,
            "unique_league_ids": unique_leagues[:30],
            "leagues_info": leagues_info[:20] if leagues_info else [],
            "sleague_raw_preview": sleague_raw[:300] if sleague_raw else None,
            "icejam_context": icejam_context[:10]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
