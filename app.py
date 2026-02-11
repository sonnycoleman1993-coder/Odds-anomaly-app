import os
from datetime import datetime, timezone, timedelta
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ----------------------------
# Target league set
# (If a sport_key doesn't exist on your Odds API plan/region, we skip it.)
# ----------------------------
WORLD_TOP_10 = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_belgium_first_div",
    "soccer_scotland_premiership",
    "soccer_uefa_champs_league",
]

ENGLAND_TOP_4 = [
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_england_league1",
    "soccer_england_league2",
]

SCOTLAND_TOP_3 = [
    "soccer_scotland_premiership",
    "soccer_scotland_championship",
    "soccer_scotland_league_one",
]

TARGET_LEAGUES = list(dict.fromkeys(WORLD_TOP_10 + ENGLAND_TOP_4 + SCOTLAND_TOP_3))


# ----------------------------
# Odds helpers
# ----------------------------
def dec_to_prob(odds: float) -> float:
    return np.nan if odds <= 1 else 1.0 / odds

def devig_normalize(probs: dict) -> dict:
    s = sum(v for v in probs.values() if np.isfinite(v))
    return {k: (v / s if (np.isfinite(v) and s > 0) else np.nan) for k, v in probs.items()}

def consensus_fair(book_probs: list[dict]) -> dict:
    if not book_probs:
        return {}
    keys = sorted(set().union(*[p.keys() for p in book_probs]))
    avg = {}
    for k in keys:
        vals = [p.get(k, np.nan) for p in book_probs]
        vals = [v for v in vals if np.isfinite(v)]
        avg[k] = float(np.mean(vals)) if vals else np.nan
    return devig_normalize(avg)

def ev_per_unit(odds: float, prob: float) -> float:
    if not (np.isfinite(odds) and np.isfinite(prob)):
        return np.nan
    return prob * odds - 1.0

def zscore(value: float, sample: list[float]) -> float:
    vals = [v for v in sample if np.isfinite(v)]
    if len(vals) < 3 or not np.isfinite(value):
        return np.nan
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    if sd <= 1e-9:
        return np.nan
    return (value - mu) / sd

def rate_pick(ev: float, anomaly_z: float) -> int:
    # 1 avoid ... 5 rare anomaly
    if not np.isfinite(ev) or ev < 0:
        return 1
    rare = np.isfinite(anomaly_z) and anomaly_z >= 2.5
    if ev >= 0.08:
        return 5 if rare else 4
    if ev >= 0.05:
        return 4
    if ev >= 0.02:
        return 3
    if ev >= 0.005:
        return 2
    return 2


# ----------------------------
# Simple model layer (transparent)
# - Uses a league-average goals baseline (slider)
# - Uses H2H fair probs to infer strength gap
# - Outputs model probs for Over/Under 2.5
# - Blends model with market consensus to avoid wild outputs
# ----------------------------
def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))

def prob_over_25(total_lambda: float) -> float:
    # P(G>=3) = 1 - P(G<=2)
    return 1.0 - poisson_cdf(2, total_lambda)

def infer_total_lambda(league_avg_goals: float, p_home: float, p_away: float) -> float:
    # Mild adjustment: more mismatch -> slightly higher expected goals
    if not (np.isfinite(p_home) and np.isfinite(p_away)):
        return league_avg_goals
    gap = abs(float(p_home - p_away))  # 0..1
    lam = league_avg_goals + 0.25 * min(1.0, gap)  # small bump
    return float(max(1.8, min(3.8, lam)))

def blend_probs(model_p: float, market_p: float, w_model: float) -> float:
    if not np.isfinite(model_p) and np.isfinite(market_p):
        return market_p
    if np.isfinite(model_p) and not np.isfinite(market_p):
        return model_p
    if not (np.isfinite(model_p) and np.isfinite(market_p)):
        return np.nan
    w = max(0.0, min(1.0, float(w_model)))
    return w * model_p + (1.0 - w) * market_p


# ----------------------------
# Odds API
# ----------------------------
def get_odds(api_key: str, sport_key: str, markets: str, regions: str = "uk"):
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=25,
    )
    r.raise_for_status()
    return r.json()

def list_sports(api_key: str):
    r = requests.get(
        f"{ODDS_API_BASE}/sports",
        params={"apiKey": api_key, "all": "true"},
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Odds Anomaly Finder", layout="wide")
st.title("⚽ Odds Anomaly Finder (Top Leagues + Totals + 4/5★ Alerts + Model Blend)")

api_key = st.secrets.get("ODDS_API_KEY") if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("ODDS_API_KEY", "")
if not api_key:
    st.error("Missing ODDS_API_KEY. Add it in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("Scope")
    regions = st.selectbox("Region", ["uk", "eu", "us"], index=0)
    hours_ahead = st.slider("Look ahead (hours)", 6, 168, 72, step=6)

    st.header("Market")
    market_mode = st.selectbox("Market", ["h2h (1X2)", "totals (O/U)"], index=1)

    st.header("Model")
    league_avg_goals = st.slider("League avg goals baseline", 2.0, 3.2, 2.7, 0.05)
    w_model = st.slider("Model weight (vs market)", 0.0, 0.8, 0.35, 0.05)

    st.header("Output")
    min_rating = st.slider("Minimum rating", 1, 5, 3)
    show_only_target_leagues = st.checkbox("Only show target leagues", value=True)

now = datetime.now(timezone.utc)
cutoff = now + timedelta(hours=hours_ahead)

# Fetch sports list to validate which target keys exist on your plan
sports = list_sports(api_key)
sports_df = pd.DataFrame(sports)
active_keys = set(sports_df[sports_df.get("active", False) == True]["key"].tolist()) if not sports_df.empty else set()

# Determine chosen competitions
target_present = [k for k in TARGET_LEAGUES if k in active_keys]
if show_only_target_leagues:
    chosen = target_present
else:
    # fall back: all active soccer competitions
    chosen = sports_df[(sports_df.get("active") == True) & (sports_df.get("key").astype(str).str.startswith("soccer_"))]["key"].tolist()

if not chosen:
    st.warning("No target leagues found in your Odds API availability list for this region/plan.")
    st.stop()

st.caption(f"Competitions loaded: {len(chosen)} (missing keys are skipped automatically).")

markets = "h2h" if market_mode.startswith("h2h") else "totals"

rows = []
progress = st.progress(0)

for i, sport_key in enumerate(chosen, start=1):
    try:
        events = get_odds(api_key, sport_key, markets=markets, regions=regions)
    except Exception:
        progress.progress(i / len(chosen))
        continue

    for e in events:
        commence = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if not (now <= commence <= cutoff):
            continue

        home = e.get("home_team", "")
        away = e.get("away_team", "")
        match = f"{home} vs {away}"

        # Collect each bookmaker quote -> probabilities
        book_probs = []
        book_data = []  # (book, selection, odds, book_prob)

        for b in e.get("bookmakers", []):
            book = b.get("title", "Unknown")
            for m in b.get("markets", []):
                if m.get("key") != markets:
                    continue

                outcomes = m.get("outcomes", [])
                # Standardize names for H2H
                if markets == "h2h":
                    mapped = {}
                    for o in outcomes:
                        name = o["name"]
                        if name == home:
                            name = "Home"
                        elif name == away:
                            name = "Away"
                        elif name.lower() == "draw":
                            name = "Draw"
                        mapped[name] = dec_to_prob(float(o["price"]))
                    probs = devig_normalize(mapped)
                    book_probs.append(probs)
                    for o in outcomes:
                        name = o["name"]
                        if name == home:
                            name = "Home"
                        elif name == away:
                            name = "Away"
                        elif name.lower() == "draw":
                            name = "Draw"
                        odds = float(o["price"])
                        book_data.append((book, name, odds, probs.get(name, np.nan)))

                # Totals market (Over/Under) — we focus on 2.5 line
                else:
                    # outcomes typically like {"name":"Over","price":..., "point":2.5}
                    mapped = {}
                    for o in outcomes:
                        point = o.get("point", None)
                        if point is None or float(point) != 2.5:
                            continue
                        name = o["name"]  # "Over" / "Under"
                        mapped[name] = dec_to_prob(float(o["price"]))
                    if len(mapped) < 2:
                        continue
                    probs = devig_normalize(mapped)
                    book_probs.append(probs)
                    for o in outcomes:
                        point = o.get("point", None)
                        if point is None or float(point) != 2.5:
                            continue
                        name = o["name"]
                        odds = float(o["price"])
                        book_data.append((book, name, odds, probs.get(name, np.nan)))

        fair = consensus_fair(book_probs)
        if not fair:
            continue

        # Model probabilities
        model_over = np.nan
        if markets == "totals":
            # infer strength from H2H if we can (optional)
            # We don't have h2h fetched in this call; we approximate using fair over/under only.
            # Instead, we keep it simple: total lambda is league baseline.
            total_lam = league_avg_goals
            model_over = prob_over_25(total_lam)
            model_under = 1.0 - model_over

            # Blend model with market consensus
            blended = {
                "Over": blend_probs(model_over, fair.get("Over", np.nan), w_model),
                "Under": blend_probs(model_under, fair.get("Under", np.nan), w_model),
            }
        else:
            blended = fair  # For H2H, we keep market consensus for now (still useful)

        # Score each bookmaker selection
        for (book, sel, odds, book_prob) in book_data:
            sample = [bp.get(sel, np.nan) for bp in book_probs]
            z = zscore(book_prob, sample)

            p = float(blended.get(sel, np.nan))
            ev = ev_per_unit(float(odds), p)
            r = rate_pick(ev, z)

            rows.append({
                "Competition": sport_key,
                "Kickoff (UTC)": commence.strftime("%Y-%m-%d %H:%M"),
                "Match": match,
                "Market": "H2H" if markets == "h2h" else "Totals 2.5",
                "Selection": sel if markets == "h2h" else f"{sel} 2.5",
                "Bookmaker": book,
                "Odds": float(odds),
                "Model/Blend Prob": round(p, 4) if np.isfinite(p) else np.nan,
                "EV": round(ev, 4) if np.isfinite(ev) else np.nan,
                "Anomaly z": round(z, 2) if np.isfinite(z) else np.nan,
                "Rating": int(r),
            })

    progress.progress(i / len(chosen))

progress.empty()

df = pd.DataFrame(rows)
if df.empty:
    st.info("No odds found in the selected window. Increase look-ahead or change region.")
    st.stop()

df = df.sort_values(["Rating", "EV", "Anomaly z"], ascending=[False, False, False]).reset_index(drop=True)

# Alerts: 4/5 only
alerts = df[df["Rating"] >= 4].copy()
alerts = alerts.head(50)

st.subheader("🚨 4★ / 5★ Alerts (top 50)")
if alerts.empty:
    st.write("No 4★/5★ alerts right now.")
else:
    st.dataframe(alerts, use_container_width=True, height=320)

st.subheader("All rated picks")
df_show = df[df["Rating"] >= min_rating].copy()
st.dataframe(df_show, use_container_width=True, height=520)

st.download_button(
    "Download CSV (rated picks)",
    data=df_show.to_csv(index=False).encode("utf-8"),
    file_name="rated_picks.csv",
    mime="text/csv",
)