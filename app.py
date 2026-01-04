from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE = "https://icejam.ca"
SCHEDULE_URL = f"{BASE}/schedule/"
DEFAULT_TEAM = "Eastern Hitmen"

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

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def points_for_game(gf, ga, ot):
    if gf == ga:
        return (1,0,0,1,0,0)
    if gf > ga:
        return (2,1,0,0,1 if ot else 0,0)
    return (1,0,1,0,0,1) if ot else (0,0,1,0,0,0)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "default_team": DEFAULT_TEAM}
    )

@app.get("/api/standings")
def standings(team: str = Query(DEFAULT_TEAM)):
    return {
        "ok": True,
        "tracked": {"team": team, "rank": None},
        "standings": [],
        "message": "Backend live. Connect scraper when tournament data is active."
    }