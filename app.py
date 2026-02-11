import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

def dec_to_prob(odds: float) -> float:
    return np.nan if odds <= 1 else 1.0 / odds

def devig_normalize(probs: dict) -> dict:
    s = sum(v for v in probs.values() if np.isfinite(v))
    return {k: (v / s if (np.isfinite(v) and s > 0) else np.nan) for k, v in probs.items()}

def consensus_fair(book_probs: list[dict]) -> dict:
    keys = sorted(set().union(*[p.keys() for p in book_probs])) if book_probs else []
    avg = {}
    for k in keys:
        vals = [p.get(k, np.nan) for p in book_probs]
        vals = [v for v in vals if np.isfinite(v)]
        avg[k] = float(np.mean(vals)) if vals else np.nan
    return devig_normalize(avg)

def ev_per_unit(odds: float, fair_prob: float) -> float:
    if not (np.isfinite(odds) and np.isfinite(fair_prob)):
        return np.nan
    return fair_prob * odds - 1.0

def rating(ev: float, anomaly_z: float) -> int:
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
    r = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key, "all": "true"}, timeout=25)
    r.raise_for_status()
    return r.json()

st.set_page_config(page_title="Odds Anomaly Finder", layout="wide")
st.title("⚽ Odds Anomaly Finder")

api_key = st.secrets.get("ODDS_API_KEY") if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("ODDS_API_KEY", "")

if not api_key:
    st.error("Missing ODDS_API_KEY. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    regions = st.selectbox("Region", ["uk", "eu", "us"], index=0)
    markets = st.selectbox("Market", ["h2h"], index=0)
    hours_ahead = st.slider("Look ahead (hours)", 6, 168, 72, step=6)
    min_rating = st.slider("Minimum rating", 1, 5, 3)

sports = list_sports(api_key)
soccer = [s for s in sports if "soccer" in (s.get("group") or "").lower() or "soccer" in (s.get("key") or "")]
soccer = [s for s in soccer if s.get("active")]

df_sports = pd.DataFrame(soccer)

if df_sports.empty:
    st.warning("No active soccer competitions returned.")
    st.stop()

default_keys = df_sports["key"].head(10).tolist()

chosen = st.multiselect(
    "Choose competitions",
    options=df_sports["key"].tolist(),
    default=default_keys,
    format_func=lambda k: df_sports.loc[df_sports["key"] == k, "title"].iloc[0],
)

if not chosen:
    st.stop()

now = datetime.now(timezone.utc)
cutoff = now + timedelta(hours=hours_ahead)

rows = []
progress = st.progress(0)

for i, sport_key in enumerate(chosen, start=1):
    try:
        events = get_odds(api_key, sport_key, markets=markets, regions=regions)
    except:
        continue

    for e in events:
        commence = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if not (now <= commence <= cutoff):
            continue

        home = e.get("home_team", "")
        away = e.get("away_team", "")
        match = f"{home} vs {away}"

        book_probs = []
        book_data = []

        for b in e.get("bookmakers", []):
            book = b.get("title", "Unknown")
            for m in b.get("markets", []):
                if m.get("key") != markets:
                    continue

                outcomes = m.get("outcomes", [])
                probs = {o["name"]: dec_to_prob(float(o["price"])) for o in outcomes}
                probs = devig_normalize(probs)
                book_probs.append(probs)

                for o in outcomes:
                    name = o["name"]
                    odds = float(o["price"])
                    book_prob = probs.get(name, np.nan)
                    book_data.append((book, name, odds, book_prob))

        fair = consensus_fair(book_probs)
        if not fair:
            continue

        for (book, name, odds, book_prob) in book_data:
            sample = [bp.get(name, np.nan) for bp in book_probs]
            sample = [v for v in sample if np.isfinite(v)]
            z = np.nan
            if len(sample) >= 3 and np.isfinite(book_prob):
                sd = float(np.std(sample, ddof=1))
                if sd > 1e-9:
                    z = (book_prob - float(np.mean(sample))) / sd

            fair_p = float(fair.get(name, np.nan))
            ev = ev_per_unit(odds, fair_p)
            r = rating(ev, z)

            rows.append({
                "Competition": sport_key,
                "Kickoff (UTC)": commence.strftime("%Y-%m-%d %H:%M"),
                "Match": match,
                "Selection": name,
                "Bookmaker": book,
                "Odds": odds,
                "EV": round(ev, 4) if np.isfinite(ev) else np.nan,
                "Rating": r,
            })

    progress.progress(i / len(chosen))

progress.empty()

df = pd.DataFrame(rows)

if df.empty:
    st.info("No matches found.")
    st.stop()

df = df.sort_values(["Rating", "EV"], ascending=[False, False])
df = df[df["Rating"] >= min_rating]

st.dataframe(df, use_container_width=True)
