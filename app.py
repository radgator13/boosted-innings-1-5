import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

st.set_page_config(layout="wide")

# === Load data ===
df = pd.read_csv("data/mlb_predictions_merged.csv")
df.columns = df.columns.str.strip()

if "Runs_1_5" not in df.columns:
    st.error("❌ 'Runs_1_5' column not found in mlb_predictions_merged.csv. Run merge_predictions.py first.")
    st.stop()

df = df.dropna(subset=["Home_Team", "Away_Team"])
df["Game_Date"] = pd.to_datetime(df["Game_Date"])
df["Runs_1_5"] = pd.to_numeric(df["Runs_1_5"], errors="coerce")
df["Actual Runs"] = df["Runs_1_5"].round(1)

# === Fireball tier
def fireballs(p):
    if p >= 0.90: return "🔥🔥🔥🔥🔥"
    elif p >= 0.80: return "🔥🔥🔥🔥"
    elif p >= 0.70: return "🔥🔥🔥"
    elif p >= 0.60: return "🔥🔥"
    else: return "🔥"

# === Confidence estimator
STD_DEV = 1.25
def get_dynamic_confidence(model_total, target, bet_type):
    return 1 - norm.cdf(target, loc=model_total, scale=STD_DEV) if bet_type.startswith("OVER") else norm.cdf(target, loc=model_total, scale=STD_DEV)

# === View Selector
view = st.radio("📊 Select View", ["Daily Predictions", "Summary & Performance"])

def check_correct(row):
    if isinstance(row["Actual Runs"], float):
        return row["Actual Runs"] > target_total if row["Bet"].startswith("OVER") else row["Actual Runs"] <= target_total
    return None

# === DAILY VIEW
if view == "Daily Predictions":
    st.title("🔥 MLB Over 4.5 Prediction Dashboard")

    # Include today + tomorrow even if predictions not available yet
    available_dates = set(df["Game_Date"].dt.date.unique())
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    available_dates.update([today, tomorrow])
    available_dates = sorted(available_dates)

    selected_date = st.date_input(
        "📅 Select Game Date",
        value=tomorrow if tomorrow in available_dates else available_dates[-1],
        min_value=min(available_dates),
        max_value=max(available_dates)
    )

    target_total = st.slider("🎯 Target Bet Line", 3.0, 6.0, 4.5, 0.5)
    min_conf = st.slider("🔥 Minimum Confidence", 0.50, 1.0, 0.55, 0.01)

    daily = df[df["Game_Date"].dt.date == selected_date].copy()
    season_to_date = df[df["Game_Date"].dt.date <= selected_date].copy()

    if daily.empty:
        st.warning("⚠️ No predictions found for this date. Make sure you've merged data using `merge_predictions.py`.")

    else:
        daily["Matchup"] = daily["Away_Team"] + " @ " + daily["Home_Team"]
        daily["Model Total"] = daily["Model_Total"].round(2)
        daily["Bet"] = daily["Model Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
        daily["Confidence"] = daily.apply(lambda row: get_dynamic_confidence(row["Model Total"], target_total, row["Bet"]), axis=1)
        daily = daily[daily["Confidence"] >= min_conf].copy()
        daily["Confidence 🔥"] = daily["Confidence"].apply(fireballs)
        daily["Actual Runs"] = daily["Actual Runs"].fillna("—")
        daily["Correct"] = daily.apply(check_correct, axis=1)

        st.subheader(f"🎯 Predictions for {selected_date.strftime('%B %d, %Y')} (Confidence ≥ {min_conf})")
        st.dataframe(daily[["Matchup", "Bet", "Confidence 🔥", "Model Total", "Actual Runs", "Correct"]], use_container_width=True)

        if not daily.empty:
            total = daily["Correct"].notna().sum()
            correct = daily["Correct"].sum()
            st.metric("Daily Accuracy", f"{(correct / total * 100):.1f}% ({correct}/{total})")

    if not season_to_date.empty:
        season = season_to_date.copy()
        season["Matchup"] = season["Away_Team"] + " @ " + season["Home_Team"]
        season["Model Total"] = season["Model_Total"].round(2)
        season["Bet"] = season["Model Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
        season["Confidence"] = season.apply(lambda row: get_dynamic_confidence(row["Model Total"], target_total, row["Bet"]), axis=1)
        season = season[season["Confidence"] >= min_conf].copy()
        season["Confidence 🔥"] = season["Confidence"].apply(fireballs)
        season["Actual Runs"] = season["Runs_1_5"].round(1)
        season["Correct"] = season.apply(check_correct, axis=1)

        st.divider()
        st.subheader("📊 Season-to-Date Accuracy")

        total_szn = season["Correct"].notna().sum()
        correct_szn = season["Correct"].sum()
        st.metric("Overall Accuracy", f"{(correct_szn / total_szn * 100):.1f}% ({correct_szn}/{total_szn})")

        season["Fireball_Level"] = (season["Confidence"] * 100).apply(
            lambda p: 5 if p >= 90 else 4 if p >= 80 else 3 if p >= 70 else 2 if p >= 60 else 1
        )
        fb_summary = (
            season[season["Correct"].notna()]
            .groupby("Fireball_Level")["Correct"]
            .agg([("Total", "count"), ("Correct", "sum")])
            .reset_index()
            .sort_values("Fireball_Level", ascending=False)
        )
        fb_summary["Accuracy"] = (fb_summary["Correct"] / fb_summary["Total"] * 100).round(1)
        fb_summary["Fireball 🔥 Tier"] = fb_summary["Fireball_Level"].map({
            5: "🔥🔥🔥🔥🔥",
            4: "🔥🔥🔥🔥",
            3: "🔥🔥🔥",
            2: "🔥🔥",
            1: "🔥"
        })

        st.dataframe(
            fb_summary[["Fireball 🔥 Tier", "Total", "Correct", "Accuracy"]],
            hide_index=True
        )

# === SUMMARY VIEW
elif view == "Summary & Performance":
    st.title("📊 Model Performance Summary")

    target_total = 4.5
    df["Bet"] = df["Model_Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
    df["Confidence"] = df.apply(lambda row: get_dynamic_confidence(row["Model_Total"], target_total, row["Bet"]), axis=1)
    df["Correct"] = df.apply(lambda row: check_correct(row) if pd.notna(row["Actual Runs"]) else None, axis=1)

    st.subheader("📈 Rolling 7-Day Accuracy")
    daily_acc = df.groupby("Game_Date")["Correct"].mean().rolling(7).mean()
    st.line_chart(daily_acc)

    st.subheader("🔥 Missed High-Confidence Predictions")
    misses = df[(df["Confidence"] >= 0.85) & (df["Correct"] == False)].copy()

    if not misses.empty:
        misses["Matchup"] = misses["Away_Team"] + " @ " + misses["Home_Team"]
        misses["Confidence 🔥"] = misses["Confidence"].apply(fireballs)
        misses["Total Runs"] = misses["Runs_1_5"].round(1)
        st.dataframe(misses[["Game_Date", "Matchup", "Confidence 🔥", "Model_Total", "Total Runs"]])
    else:
        st.info("✅ No high-confidence misses found!")

    st.subheader("💰 Simulated Profit (Flat $100 Bets)")
    df["Profit"] = df.apply(
        lambda row: 100 if row["Correct"] else -110 if row["Predicted_Over_4_5"] == 1 else -100,
        axis=1
    )
    profit_curve = df.groupby("Game_Date")["Profit"].sum().cumsum()
    st.line_chart(profit_curve)
