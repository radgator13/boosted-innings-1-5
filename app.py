import streamlit as st
st.set_page_config(layout="wide")  # ✅ MUST BE FIRST

import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from scipy.stats import norm

# === Load data ===
df = pd.read_csv("data/mlb_predictions_merged.csv")
df.columns = df.columns.str.strip()

if "Runs_1_5" not in df.columns:
    st.error("❌ 'Runs_1_5' column not found. Run merge_predictions.py first.")
    st.stop()

df = df.dropna(subset=["Home_Team", "Away_Team"])
df["Game_Date"] = pd.to_datetime(df["Game_Date"])
df["Runs_1_5"] = pd.to_numeric(df["Runs_1_5"], errors="coerce")
df["Actual Runs"] = df["Runs_1_5"].round(1)

# === Fireball display tier ===
def fireballs(p):
    if p >= 0.90: return "🔥🔥🔥🔥🔥"
    elif p >= 0.80: return "🔥🔥🔥🔥"
    elif p >= 0.70: return "🔥🔥🔥"
    elif p >= 0.60: return "🔥🔥"
    else: return "🔥"

# === Confidence logic ===
STD_DEV = 1.25
def get_dynamic_confidence(model_total, target, bet_type):
    if bet_type.startswith("OVER"):
        return 1 - norm.cdf(target, loc=model_total, scale=STD_DEV)
    else:
        return norm.cdf(target, loc=model_total, scale=STD_DEV)

# === Correctness checkers ===
def mark_correct_symbol(row):
    try:
        actual = float(row["Actual Runs"])
        if row["Bet"].startswith("OVER"):
            line = float(row["Bet"].split(" ")[-1])
            return "✅" if actual > line else "❌"
        elif row["Bet"].startswith("UNDER"):
            line = float(row["Bet"].split(" ")[-1])
            return "✅" if actual < line else "❌"
        else:
            return "—"
    except:
        return "—"


def mark_correct_numeric(row):
    if row["Bet"] == "OVER 4.5" and row["Actual Runs"] != "—":
        return float(row["Actual Runs"]) > 4.5
    elif row["Bet"] == "UNDER 4.5" and row["Actual Runs"] != "—":
        return float(row["Actual Runs"]) < 4.5
    else:
        return None

# === Assign core model columns
target_total = 4.5
df["Bet"] = df["Model_Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
df["Confidence"] = df.apply(lambda row: get_dynamic_confidence(row["Model_Total"], target_total, row["Bet"]), axis=1)
df["Correct"] = df.apply(mark_correct_numeric, axis=1)
df["Correct Symbol"] = df.apply(mark_correct_symbol, axis=1)

# === View selector
view = st.sidebar.radio("📊 Select View", [
    "Daily Predictions",
    "Summary & Performance",
    "Bet Sizing Analysis",
    "Confidence Accuracy Breakdown",
    "Fireball Volume Over Time",
    "Fireball Profit Curve",
    "Top Daily Picks Leaderboard",
    "Calendar Heatmap",
    "Confidence Distribution Histogram"
])

# === Daily Predictions Tab
if view == "Daily Predictions":
    st.title("🔥 MLB Over 4.5 Prediction Dashboard")

    available_dates = set(df["Game_Date"].dt.date.unique())
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    available_dates.update([today, tomorrow])
    available_dates = sorted(available_dates)

    selected_date = st.date_input("📅 Select Game Date", tomorrow, min_value=min(available_dates), max_value=max(available_dates))
    target_total = st.slider("🎯 Target Bet Line", 3.0, 6.0, 4.5, 0.5)
    min_conf = st.slider("🔥 Minimum Confidence", 0.50, 1.0, 0.55, 0.01)

    daily = df[df["Game_Date"].dt.date == selected_date].copy()

    daily["Matchup_ID"] = daily["Game_Date"].astype(str) + "_" + daily["Home_Team"].str.strip() + "_" + daily["Away_Team"].str.strip()
    daily = daily.sort_values("Confidence", ascending=False)
    daily = daily.drop_duplicates("Matchup_ID")
    season_to_date = df[df["Game_Date"].dt.date <= selected_date].copy()

    if daily.empty:
        st.warning("⚠️ No predictions found for this date.")
    else:
        daily["Matchup"] = daily["Away_Team"] + " @ " + daily["Home_Team"]
        daily["Model Total"] = daily["Model_Total"].round(2)
        daily["Bet"] = daily["Model Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
        daily["Confidence"] = daily.apply(lambda row: get_dynamic_confidence(row["Model Total"], target_total, row["Bet"]), axis=1)
        daily = daily[daily["Confidence"] >= min_conf].copy()
        daily["Confidence 🔥"] = daily["Confidence"].apply(fireballs)
        daily["Actual Runs"] = daily["Actual Runs"].fillna("—")
        daily["Correct Symbol"] = daily.apply(mark_correct_symbol, axis=1)
        daily["Correct"] = daily.apply(mark_correct_numeric, axis=1).astype("boolean")

        st.dataframe(
            daily[["Matchup", "Bet", "Confidence 🔥", "Model Total", "Actual Runs", "Correct Symbol"]],
            use_container_width=True
        )

        correct = daily["Correct"].sum()
        total = daily["Correct"].notna().sum()
        if total > 0:
            st.metric("Daily Accuracy", f"{(correct / total * 100):.1f}% ({int(correct)}/{total})")
        else:
            st.metric("Daily Accuracy", "—")

        # Rolling accuracy
        historical = season_to_date.copy()
        historical["Bet"] = historical["Model_Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
        historical["Confidence"] = historical.apply(lambda row: get_dynamic_confidence(row["Model_Total"], target_total, row["Bet"]), axis=1)
        historical = historical[historical["Confidence"] >= min_conf].copy()
        historical["Correct"] = historical.apply(mark_correct_numeric, axis=1).astype("boolean")

        total_wins = historical["Correct"].sum()
        total_games = historical["Correct"].notna().sum()
        if total_games > 0:
            st.metric("Rolling Accuracy", f"{(total_wins / total_games * 100):.1f}% ({int(total_wins)}/{int(total_games)})")
        else:
            st.metric("Rolling Accuracy", "—")


# ✅ Continue with the rest of your tabs as-is.
