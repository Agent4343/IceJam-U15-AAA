# IceJam U15 AAA

Mobile-first standings tracker for IceJam U15 AAA.

- FastAPI backend
- Phone-optimized UI
- Designed for Railway deployment

---

## 2026 Tournament Rules - Quick Guide

### Game Format

| Phase | Periods | Overtime |
|-------|---------|----------|
| Round Robin | 15-15-20 min | 5 min 3v3, then tie (1 pt each) |
| Playoffs | 15-15-20 min | 10 min 3v3, then shootout |
| Championship | 15-15-20 min | 10 min 3v3, then 20 min periods until winner |

### Points System

| Result | Winner | Loser |
|--------|--------|-------|
| Regulation Win | 2 pts | 0 pts |
| Overtime Win | 2 pts | 1 pt |
| Tie (round robin only) | 1 pt | 1 pt |

### Key Rules

- **Mercy Rule**: If a team leads by 6+ goals after the 2nd period, the clock runs continuously
- **Goal Differential Cap**: Max +7 per game counts for tiebreakers
- **Timeouts**: One 30-second timeout per game
- **Jerseys**: Home team wears white

---

## Tiebreaker Rules - How It Works

When teams are tied in points, we go through these steps **in order** until the tie is broken:

### Two Teams Tied

| Step | Tiebreaker | What It Means |
|------|------------|---------------|
| 1 | Head-to-head | Who won when they played each other? |
| 2 | Most wins | Team with more wins ranks higher |
| 3 | Goal average | Goals For / (Goals For + Goals Against) |
| 4 | Fewest goals against | Better defense wins |
| 5 | Most goals for | Better offense wins |
| 6 | Fewest penalty minutes | Cleaner team wins |
| 7 | First goal in head-to-head | Who scored first when they played? |
| 8 | Fastest first goal | Who scored their first tournament goal earliest? |
| 9 | Coin toss | Random |

### Three or More Teams Tied

| Step | Tiebreaker | What It Means |
|------|------------|---------------|
| 1 | Record among tied teams | Points from games between tied teams only |
| 2 | Most wins (among tied) | Wins in games between tied teams |
| 3 | Goal average (all games) | Goals For / (Goals For + Goals Against) |
| 4 | Fewest goals against | Total from all round robin games |
| 5 | Most goals for | Total from all round robin games |
| 6 | Fewest penalty minutes | Total from all round robin games |
| 7 | Fastest first goal | Ranked by time of first tournament goal |
| 8 | Coin toss | Odd team out wins in 3-team toss |

### Goal Average Formula

```
Goal Average = Goals For / (Goals For + Goals Against)

Example: 10 goals for, 4 against
         10 / (10 + 4) = 10/14 = 0.714

Higher percentage = Higher rank
```

---

## Important Notes

- **Replacement Seeding**: In playoffs, if a lower seed beats a higher seed, they take that higher seed position
- **AP Players**: Players registered with higher teams can't play for their main team until the lower team is eliminated
- **No Protests**: Tournament committee decisions are final