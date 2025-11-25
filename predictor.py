import requests
import numpy as np


# -------------------------------------------------------------
# SAFE FOTMOB FETCH FUNCTION
# -------------------------------------------------------------
def get_team_recent_xg(team_id: str, matches: int = 5):
    """
    Safely fetch recent xG for a team from FotMob.
    If FotMob blocks or returns non-JSON, fallback xG is used.
    """
    url = f"https://www.fotmob.com/api/teams?id={team_id}&ccode3=ENG"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json,text/plain,*/*",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        text = r.text.strip()

        # Try alternate URL if first fails
        if not text or not text.startswith("{"):
            alt_url = f"https://www.fotmob.com/api/teams?id={team_id}"
            r = requests.get(alt_url, headers=headers, timeout=10)
            text = r.text.strip()

        # Still not JSON → fallback
        if not text or not text.startswith("{"):
            print(f"[WARN] FotMob returned non-JSON for team {team_id}, using fallback xG.")
            return 1.4, 1.3

        data = r.json()

        matches_section = data.get("matches", {})

        # Try all possible match lists
        events = None
        for key in ["all", "recentMatches", "previousMatches", "played"]:
            v = matches_section.get(key)
            if isinstance(v, list) and len(v) > 0:
                events = v
                break

        if not events:
            print(f"[WARN] No matches list for team {team_id}, using fallback xG.")
            return 1.4, 1.3

        xg_for = []
        xg_against = []

        for m in events[:matches]:
            home = m.get("home", {})
            away = m.get("away", {})

            if str(home.get("id")) == str(team_id):
                xf = home.get("xG")
                xa = away.get("xG")
            elif str(away.get("id")) == str(team_id):
                xf = away.get("xG")
                xa = home.get("xG")
            else:
                continue

            if xf is None or xa is None:
                continue

            xg_for.append(float(xf))
            xg_against.append(float(xa))

        if not xg_for:
            print(f"[WARN] No xG values for team {team_id}, using fallback xG.")
            return 1.4, 1.3

        return float(np.mean(xg_for)), float(np.mean(xg_against))

    except Exception:
        print(f"[WARN] Error fetching FotMob for team {team_id}, using fallback xG.")
        return 1.4, 1.3


# -------------------------------------------------------------
# TEAM STRENGTH
# -------------------------------------------------------------
def compute_team_strength(avg_for: float, avg_against: float, league_avg: float = 1.35):
    attack = avg_for / league_avg
    defence = avg_against / league_avg
    return attack, defence


# -------------------------------------------------------------
# MATCH PREDICTOR
# -------------------------------------------------------------
def predict_match(
    home_name: str,
    away_name: str,
    hxGF: float,
    hxGA: float,
    axGF: float,
    axGA: float,
    sims: int = 20000,
):
    """
    Monte-Carlo Poisson match predictor.
    """
    hA, hD = compute_team_strength(hxGF, hxGA)
    aA, aD = compute_team_strength(axGF, axGA)

    # Expected goals
    home_xg = max(0.05, hA * aD * 1.05)
    away_xg = max(0.05, aA * hD * 0.95)

    home_goals = np.random.poisson(home_xg, sims)
    away_goals = np.random.poisson(away_xg, sims)

    home_win = float(np.mean(home_goals > away_goals))
    draw = float(np.mean(home_goals == away_goals))
    away_win = float(np.mean(home_goals < away_goals))

    # Most likely scoreline
    scores, counts = np.unique(
        list(zip(home_goals, away_goals)), axis=0, return_counts=True
    )
    idx = int(np.argmax(counts))
    ml_home, ml_away = scores[idx]

    return {
        "home_team": home_name,
        "away_team": away_name,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "most_likely_score": f"{ml_home}-{ml_away}",
    }


if __name__ == "__main__":
    print(predict_match("A", "B", 1.5, 1.2, 1.3, 1.1))
