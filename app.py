"""
Grand Montreal International U15 Hockey Tournament - Standings Tracker
Tournoi International Laval 2026

This app automatically tracks round robin standings for the Greater Montreal
International U15 Hockey Tournament. Sanctioned by Hockey Quebec.

All round robin logic is automated - standings are scraped from icejam.ca and
tiebreaker rules are applied automatically to determine rankings.

TIEBREAKER RULES (Article 9.7 - Hockey Quebec):
    a) Highest number of points
    b) Highest number of wins
    c) Least goals against
    d) Most goals for
    e) Quickest goal scored in all games played
    f) Most Franc Jeu (Fair Play) points
    g) By a draw

TIEBREAKER LIMITATIONS:
    The following tiebreakers ARE implemented automatically:
        TB-a: Highest number of points
        TB-b: Highest number of wins
        TB-c: Least goals against
        TB-d: Most goals for
        TB-g: Alphabetical order (deterministic fallback for draw)

    The following tiebreakers CANNOT be determined (data not available):
        TB-e: Quickest goal scored (requires play-by-play data)
        TB-f: Most Franc Jeu points (not tracked by scraper)

POINTS SYSTEM:
    Win = 2 points
    Tie = 1 point
    Loss = 0 points
    Franc Jeu (Fair Play) = 1 point (not tracked automatically)

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
import json as json_lib
import re
import uuid
import logging
import os
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
DEFAULT_TEAM = "Eastern Hitmen"
DEFAULT_LEAGUE = "500226"  # IceJam U15 league ID (Eastern Hitmen's league)

TOURNAMENT_NAME = "Grand Montreal International U15 Hockey Tournament"
TOURNAMENT_SHORT = "Tournoi International Laval 2026"

# Multiplier for tournament time calculation (ensures game order takes precedence over time within game)
TOURNAMENT_TIME_MULTIPLIER = 100000

# Mercy rule threshold (7+ goal lead after 2nd period = running time)
MERCY_RULE_THRESHOLD = 7

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
    game_number: int = 0  # Order of game in tournament (for quickest goal tiebreaker)
    franc_jeu_a: int = 0  # Franc Jeu (Fair Play) points for team A (0 or 1)
    franc_jeu_b: int = 0  # Franc Jeu (Fair Play) points for team B (0 or 1)


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
    franc_jeu_a: int = Field(default=0, ge=0, le=1, description="Franc Jeu point for team A (0 or 1)")
    franc_jeu_b: int = Field(default=0, ge=0, le=1, description="Franc Jeu point for team B (0 or 1)")

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


def points_for_game(gf: int, ga: int, ot: bool) -> Tuple[int, int, int, int]:
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


def compare_two_teams(t1: TeamStats, t2: TeamStats, games: Dict[str, Game]) -> int:
    """
    Compare two teams using Hockey Quebec Article 9.7 tiebreaker rules.
    Returns: -1 if t1 ranks higher, 1 if t2 ranks higher, 0 if still tied

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


def sort_tied_group(teams: List[TeamStats], games: Dict[str, Game]) -> List[TeamStats]:
    """
    Sort a group of teams that are tied in points using Hockey Quebec Article 9.7 rules.
    """
    if len(teams) <= 1:
        return teams

    def compare_multi(t1: TeamStats, t2: TeamStats) -> int:
        return compare_two_teams(t1, t2, games)

    return sorted(teams, key=cmp_to_key(compare_multi))


def calculate_standings() -> List[dict]:
    """Calculate standings from all games using Hockey Quebec tiebreaker rules."""
    if not games_db:
        return []

    stats: Dict[str, TeamStats] = {}

    # Sort games by game_number to process in order
    sorted_games = sorted(games_db.values(), key=lambda g: g.game_number)

    for game in sorted_games:
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

        # Goals (no differential cap in Laval 2026 rules)
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

        # Calculate points (Win=2, Tie=1, Loss=0)
        pts_a, w_a, l_a, t_a = points_for_game(game.goals_a, game.goals_b, game.ot)
        pts_b, w_b, l_b, t_b = points_for_game(game.goals_b, game.goals_a, game.ot)

        team_a.pts += pts_a
        team_a.w += w_a
        team_a.l += l_a
        team_a.t += t_a

        team_b.pts += pts_b
        team_b.w += w_b
        team_b.l += l_b
        team_b.t += t_b

    # Group teams by points
    teams_list = list(stats.values())
    points_groups: Dict[int, List[TeamStats]] = {}
    for team in teams_list:
        if team.pts not in points_groups:
            points_groups[team.pts] = []
        points_groups[team.pts].append(team)

    # Sort each group using Hockey Quebec tiebreaker rules
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
            "pts": team.pts,
            "gf": team.gf,
            "ga": team.ga,
            "pim": team.pim,
            "franc_jeu": team.franc_jeu,
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

        # Sort by points (descending) then by fewest goals against (Hockey Quebec TB-c)
        standings_data.sort(key=lambda x: (x["pts"], -x["ga"], x["gf"]), reverse=True)

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
        games = []

        # First try the schedule endpoint which reliably contains game data with scores
        schedule_url = f"{SCHEDULE_URL}?lg={lg}"
        logger.info(f"Fetching scores from schedule: {schedule_url}")
        response = requests.get(schedule_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text

        # Extract json variable from schedule page
        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)

        if json_match:
            try:
                games_json = json_lib.loads(json_match.group(1))
                logger.info(f"Found {len(games_json)} total games in schedule JSON for scores")

                for game in games_json:
                    home_team = game.get("h_n", "")
                    away_team = game.get("v_n", "")
                    home_score = int(game.get("hf", 0) or 0)
                    away_score = int(game.get("vf", 0) or 0)
                    game_status = str(game.get("gs", "")).upper()

                    is_completed = game_status in ["F", "FINAL"] or (home_score > 0 or away_score > 0)

                    if home_team and away_team and is_completed:
                        games.append({
                            "home": home_team,
                            "away": away_team,
                            "home_score": home_score,
                            "away_score": away_score,
                            "ot": "OT" in str(game.get("gp", "")),
                            "game_num": game.get("gn", ""),
                        })

                logger.info(f"Found {len(games)} completed games with scores")
            except json_lib.JSONDecodeError as e:
                logger.error(f"JSON parse error in schedule/scores: {e}")
        else:
            logger.warning(f"No json variable found in schedule page for scores")

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
    Apply Hockey Quebec Article 9.7 tiebreaker rules to live standings data.
    Returns (sorted_standings, tiebreaker_log) with details of calculations.

    Tiebreaker order (Article 9.7):
        a) Highest number of points
        b) Highest number of wins
        c) Least goals against
        d) Most goals for
        e) Quickest goal scored in all games played (NOT AVAILABLE - requires play-by-play)
        f) Most Franc Jeu points (NOT AVAILABLE - not tracked by scraper)
        g) By a draw (alphabetical as deterministic fallback)
    """
    if not standings:
        return standings, []

    tiebreaker_log = []

    def compare_teams_with_log(t1: Dict, t2: Dict) -> Tuple[int, str]:
        """Compare two teams and return (result, reason)."""

        # a) Highest number of points
        if t1["pts"] != t2["pts"]:
            reason = f"Points: {t1['team']} ({t1['pts']}) vs {t2['team']} ({t2['pts']})"
            return (-1 if t1["pts"] > t2["pts"] else 1, reason)

        # b) Highest number of wins
        if t1["w"] != t2["w"]:
            reason = f"TB-b Wins: {t1['team']} ({t1['w']}W) vs {t2['team']} ({t2['w']}W)"
            return (-1 if t1["w"] > t2["w"] else 1, reason)

        # c) Least goals against
        if t1["ga"] != t2["ga"]:
            reason = f"TB-c Least GA: {t1['team']} ({t1['ga']} GA) vs {t2['team']} ({t2['ga']} GA)"
            return (-1 if t1["ga"] < t2["ga"] else 1, reason)

        # d) Most goals for
        if t1["gf"] != t2["gf"]:
            reason = f"TB-d Most GF: {t1['team']} ({t1['gf']} GF) vs {t2['team']} ({t2['gf']} GF)"
            return (-1 if t1["gf"] > t2["gf"] else 1, reason)

        # e) Quickest goal - NOT AVAILABLE (requires play-by-play data)
        # f) Most Franc Jeu points - NOT AVAILABLE (not tracked by scraper)

        # g) By a draw (alphabetical as deterministic fallback)
        reason = f"TB-g Draw (alphabetical): {t1['team']} vs {t2['team']} (TB-e quickest goal and TB-f Franc Jeu data not available)"
        return (-1 if t1["team"].lower() < t2["team"].lower() else 1, reason)

    def compare_teams(t1: Dict, t2: Dict) -> int:
        result, reason = compare_teams_with_log(t1, t2)
        if "TB" in reason:
            tiebreaker_log.append(reason)
        return result

    # Sort standings using Hockey Quebec tiebreaker rules
    from functools import cmp_to_key
    sorted_standings = sorted(standings, key=cmp_to_key(compare_teams))

    # Update ranks
    for i, team in enumerate(sorted_standings, 1):
        team["rank"] = i

    return sorted_standings, tiebreaker_log


def scrape_icejam_with_tiebreakers(league_id: str = None, season: str = "2025") -> Dict:
    """Scrape standings and apply Hockey Quebec tiebreaker rules"""
    # Get standings
    standings_result = scrape_icejam(league_id, season)
    if not standings_result.get("ok"):
        return standings_result

    # Get game scores for tiebreaker calculations
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
        # Apply tiebreakers without game data
        sorted_standings, tiebreaker_log = apply_tiebreakers_to_live(
            standings_result["standings"],
            []
        )
        standings_result["standings"] = sorted_standings
        standings_result["tiebreakers_applied"] = True
        standings_result["tiebreaker_note"] = "Could not fetch game scores - using basic tiebreakers"
        standings_result["tiebreaker_log"] = tiebreaker_log

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
        jsonst_start = html.find('let jsonSt = [')
        if jsonst_start > 0:
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

                    if jsonst_data and isinstance(jsonst_data, list) and len(jsonst_data) > 0:
                        standings_array = jsonst_data[0].get("jsonStandings", [])
                        logger.info(f"Found {len(standings_array)} teams in jsonStandings")

                        for team in standings_array:
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
                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 5:
                        team_cell = cells[0].get_text(strip=True)
                        if not team_cell or team_cell.upper() in ["TEAM", "RANK", "#", ""]:
                            continue

                        try:
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
                    break

        # Sort by points (descending) then by fewest goals against
        standings_data.sort(key=lambda x: (x["pts"], -x["ga"], x["gf"]), reverse=True)

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


def scrape_schedule(team: str = DEFAULT_TEAM, league_id: str = None) -> Dict:
    """Scrape schedule data from icejam.ca/schedule/"""
    try:
        lg = league_id or DEFAULT_LEAGUE
        url = f"{SCHEDULE_URL}?lg={lg}"
        logger.info(f"Fetching {url}")
        response = requests.get(url, headers=HEADERS, timeout=10)
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
                if "hitman" in team_lower or "hitmen" in team_lower:
                    search_terms.extend(["hitman", "hitmen"])

                for game in games_json:
                    home = (game.get("h_n") or "").lower()
                    visitor = (game.get("v_n") or "").lower()

                    is_team_game = any(term in home or term in visitor for term in search_terms)

                    if is_team_game:
                        if any(term in home for term in search_terms):
                            opponent = game.get("v_n", "TBD")
                            location = "vs"
                        else:
                            opponent = game.get("h_n", "TBD")
                            location = "@"

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
            "url": url,
            "league_id": lg,
            "team": team,
            "games_found": len(schedule_data),
            "schedule": schedule_data
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Schedule scrape error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "url": url if 'url' in locals() else SCHEDULE_URL
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
        game_number=game_number,
        franc_jeu_a=game.franc_jeu_a,
        franc_jeu_b=game.franc_jeu_b,
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
    season: str = Query("2025", description="Season year (default: 2025 for 2025-2026 season)"),
    apply_rules: bool = Query(True, description="Apply Hockey Quebec tiebreaker rules")
):
    """Scrape standings from icejam.ca API with Hockey Quebec tiebreaker rules"""
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
        hitmen_sections = []
        lines = html.split('\n')
        for i, line in enumerate(lines):
            if 'hitmen' in line.lower():
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
            "sections": hitmen_sections[:5]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/debug-scores")
def debug_scores():
    """Debug: show raw data from icejam.ca/scores/"""
    try:
        lg = DEFAULT_LEAGUE
        scores_url = f"{BASE}/scores/?lg={lg}"
        response = requests.get(scores_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text

        patterns = [
            ("json=", r'(?:var|let|const)?\s*json\s*=\s*(\[[\s\S]*?\])(?:;|\s*$)'),
            ("json greedy", r'json\s*=\s*(\[.*\]);'),
            ("named vars", r'(?:var|let|const)\s*(?:json|jsonScores|scoresData|data|games)\s*=\s*(\[[\s\S]*?\]);'),
            ("h_n field", r'=\s*(\[\s*\{[^}]*"h_n"[^}]*\}[\s\S]*?\]);'),
        ]

        json_match = None
        matched_pattern = None
        for name, pattern in patterns:
            json_match = re.search(pattern, html, re.DOTALL)
            if json_match:
                matched_pattern = name
                break

        result = {
            "ok": True,
            "url": scores_url,
            "html_length": len(html),
            "json_found": bool(json_match),
            "matched_pattern": matched_pattern,
        }

        script_matches = re.findall(r'<script[^>]*>([\s\S]*?)</script>', html, re.IGNORECASE)
        result["script_count"] = len(script_matches)
        result["script_previews"] = [s[:150] for s in script_matches if len(s.strip()) > 20][:5]

        if json_match:
            try:
                json_str = json_match.group(1)
                bracket_count = 0
                end_pos = 0
                for i, char in enumerate(json_str):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    json_str = json_str[:end_pos]

                games_json = json_lib.loads(json_str)
                result["total_games"] = len(games_json)
                result["sample_games"] = games_json[:3] if games_json else []
                league_counts = {}
                for g in games_json:
                    gl = str(g.get("lg", "unknown"))
                    league_counts[gl] = league_counts.get(gl, 0) + 1
                result["games_by_league"] = league_counts
            except Exception as e:
                result["json_parse_error"] = str(e)
        else:
            result["html_snippet"] = html[:2000]

        return result
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/debug-standings")
def debug_standings():
    """Debug: show structure from icejam.ca/standings/"""
    try:
        response = requests.get(STANDINGS_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()

        html = response.text

        patterns_tried = []

        json_match = re.search(r'json\s*=\s*(\[.*?\]);', html, re.DOTALL)
        patterns_tried.append({"pattern": "json = [...];", "found": bool(json_match)})

        json_match2 = re.search(r'json\s*=\s*(\[[\s\S]*?\]\s*);', html)
        patterns_tried.append({"pattern": "json = [...] greedy", "found": bool(json_match2)})

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

        hitmen_idx = html.lower().find('hitmen')
        hitmen_context = html[max(0,hitmen_idx-200):hitmen_idx+300] if hitmen_idx > 0 else "Not found"

        script_content = []
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script")[:5]:
            text = script.get_text()[:500] if script.get_text() else ""
            if text and len(text) > 50:
                script_content.append(text)

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
                first_cells = rows[0].find_all(["th", "td"])
                table_data["first_row"] = [c.get_text(strip=True)[:30] for c in first_cells[:10]]
                for row in rows[1:4]:
                    cells = row.find_all(["td", "th"])
                    table_data["sample_rows"].append([c.get_text(strip=True)[:30] for c in cells[:10]])
            tables_info.append(table_data)

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

        leagues_info = []
        sleague_raw = None

        sleague_match = re.search(r'let\s+jsonSLeague\s*=\s*"([^"]*)"', html)
        if sleague_match:
            sleague_raw = sleague_match.group(1)
            try:
                unescaped = sleague_raw.replace('\\"', '"').replace('\\\\', '\\')
                leagues_info = json_lib.loads(unescaped)
            except:
                pass

        if not leagues_info:
            sleague_match2 = re.search(r'jsonSLeague\s*=\s*(\[.*?\]);', html, re.DOTALL)
            if sleague_match2:
                try:
                    leagues_info = json_lib.loads(sleague_match2.group(1))
                except:
                    sleague_raw = sleague_match2.group(1)[:500]

        league_def_match = re.search(r'leagueDef\s*=\s*(\d+)', html)
        league_def = league_def_match.group(1) if league_def_match else None

        league_in_new_match = re.search(r'leagueInNew\s*=\s*(\d+)', html)
        league_in_new = league_in_new_match.group(1) if league_in_new_match else None

        all_league_ids = re.findall(r'\b(50\d{4})\b', html)
        unique_leagues = list(set(all_league_ids))

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


# ============ AI ANALYSIS ============

@app.get("/api/ai-analysis")
def ai_analysis(
    team: str = Query(DEFAULT_TEAM, description="Team to analyze"),
    league: str = Query(None, description="League ID"),
    season: str = Query("2025", description="Season year")
):
    """
    Get AI-powered analysis of a team's standings and tournament situation.
    Requires ANTHROPIC_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "ANTHROPIC_API_KEY not configured. Set this environment variable in Railway."
        }

    try:
        import anthropic
    except ImportError:
        return {
            "ok": False,
            "error": "AI feature not available. Install anthropic package: pip install anthropic"
        }

    try:
        # Get current standings with tiebreakers
        standings_result = scrape_icejam_with_tiebreakers(league, season)
        if not standings_result.get("ok"):
            return {"ok": False, "error": "Could not fetch standings"}

        standings = standings_result.get("standings", [])
        tiebreakers = standings_result.get("tiebreaker_log", [])

        # Find the tracked team
        team_data = None
        team_rank = None
        for i, t in enumerate(standings):
            if team.lower() in t["team"].lower():
                team_data = t
                team_rank = i + 1
                break

        if not team_data:
            return {"ok": False, "error": f"Team '{team}' not found in standings"}

        # Get schedule for the team
        schedule_result = scrape_schedule(team, league)
        upcoming_games = schedule_result.get("schedule", [])[:5] if schedule_result.get("ok") else []

        # Get recent scores
        scores_result = fetch_game_scores(league, season)
        team_scores = []
        if scores_result.get("ok"):
            for game in scores_result.get("games", []):
                if team.lower() in game["home"].lower() or team.lower() in game["away"].lower():
                    team_scores.append(game)

        # Build context for Claude
        total_teams = len(standings)

        # Get nearby teams in standings
        nearby_teams = []
        for i in range(max(0, team_rank - 3), min(total_teams, team_rank + 2)):
            t = standings[i]
            nearby_teams.append(f"#{i+1} {t['team']}: {t['w']}-{t['l']}-{t['t']} ({t['pts']} pts)")

        # Build the prompt
        prompt = f"""You are a hockey analyst providing a brief update for fans of {team} at the Grand Montreal International U15 Hockey Tournament (Tournoi International Laval 2026).

This tournament is sanctioned by Hockey Quebec. Points system: Win=2, Tie=1, Loss=0, Franc Jeu (Fair Play)=1.
Tiebreakers (Article 9.7): points > wins > least GA > most GF > quickest goal > Franc Jeu > draw.

Current Standings:
- {team} is ranked #{team_rank} of {total_teams} teams
- Record: {team_data['w']} wins, {team_data['l']} losses, {team_data['t']} ties ({team_data['pts']} points)
- Goals: {team_data['gf']} for, {team_data['ga']} against

Nearby Teams:
{chr(10).join(nearby_teams)}

Recent Games:
{chr(10).join([f"vs {g['away'] if team.lower() in g['home'].lower() else g['home']}: {g['home_score']}-{g['away_score']}" for g in team_scores[:3]]) if team_scores else "No completed games yet"}

Upcoming Games:
{chr(10).join([f"{g['location']} {g['opponent']} - {g['date']} {g['time']}" for g in upcoming_games]) if upcoming_games else "No upcoming games found"}

Tiebreaker Notes:
{chr(10).join(tiebreakers[:5]) if tiebreakers else "No tiebreakers applied yet"}

Provide a brief (3-4 sentences) fan-friendly analysis covering:
1. Current standing and position
2. Recent performance
3. What to watch for in upcoming games

Keep it conversational and encouraging. Use hockey terminology appropriately."""

        # Call Claude API
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        analysis = message.content[0].text

        return {
            "ok": True,
            "team": team_data["team"],
            "rank": team_rank,
            "total_teams": total_teams,
            "record": f"{team_data['w']}-{team_data['l']}-{team_data['t']}",
            "points": team_data["pts"],
            "analysis": analysis
        }

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return {"ok": False, "error": f"AI service error: {str(e)}"}
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return {"ok": False, "error": str(e)}
