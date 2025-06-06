﻿import streamlit as st
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

# === Tab 2: Summary & Performance
elif view == "Summary & Performance":
    st.title("📊 Model Performance Summary")
    target_total = 4.5
    df["Bet"] = df["Model_Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
    df["Confidence"] = df.apply(lambda row: get_dynamic_confidence(row["Model_Total"], target_total, row["Bet"]), axis=1)
    df["Correct"] = df.apply(lambda row: mark_correct_numeric(row) if pd.notna(row["Actual Runs"]) else None, axis=1)


    st.subheader("📈 Rolling 7-Day Accuracy")
    st.line_chart(df.groupby("Game_Date")["Correct"].mean().rolling(7).mean())

    st.subheader("🔥 Missed High-Confidence Predictions")
    misses = df[(df["Confidence"] >= 0.85) & (df["Correct"] == False)].copy()
    if not misses.empty:
        misses["Matchup"] = misses["Away_Team"] + " @ " + misses["Home_Team"]
        misses["Confidence 🔥"] = misses["Confidence"].apply(fireballs)
        misses["Total Runs"] = misses["Runs_1_5"].round(1)
        st.dataframe(misses[["Game_Date", "Matchup", "Confidence 🔥", "Model_Total", "Total Runs"]])
    else:
        st.success("✅ No high-confidence misses found!")

# === Tab 3: Bet Sizing Analysis
elif view == "Bet Sizing Analysis":
    st.markdown("## 💸 Bet Sizing Simulation by Fireball Tier")
    min_date = df["Game_Date"].min().date()
    max_date = df["Game_Date"].max().date()
    start_date, end_date = st.date_input("📆 Select date range:", [min_date, max_date])

    df_filtered = df[(df["Game_Date"] >= pd.to_datetime(start_date)) & (df["Game_Date"] <= pd.to_datetime(end_date))].copy()
    target_total = 4.5
    df_filtered["Bet"] = df_filtered["Model_Total"].apply(lambda x: f"OVER {target_total}" if x > target_total else f"UNDER {target_total}")
    df_filtered["Confidence"] = df_filtered.apply(lambda row: get_dynamic_confidence(row["Model_Total"], target_total, row["Bet"]), axis=1)
    df_filtered["Correct"] = df_filtered.apply(lambda row: mark_correct_numeric(row) if pd.notna(row["Actual Runs"]) else None, axis=1)
    df_filtered["Fireball_Level"] = (df_filtered["Confidence"] * 100).apply(lambda p: 5 if p >= 90 else 4 if p >= 80 else 3 if p >= 70 else 2 if p >= 60 else 1)
    df_filtered["Fireball 🔥 Tier"] = df_filtered["Fireball_Level"].map({5: "🔥🔥🔥🔥🔥", 4: "🔥🔥🔥🔥", 3: "🔥🔥🔥", 2: "🔥🔥", 1: "🔥"})

    strategy = st.radio("🧮 Bet Sizing Strategy:", ["Fireball-Based", "Flat $100 Bets"])
    if strategy == "Flat $100 Bets":
        df_filtered["Bet_Size"] = 100
    else:
        fireball_bets = {5: 15.0, 4: 10.0, 3: 5.0, 2: 2.5, 1: 1.0}
        df_filtered["Bet_Size"] = df_filtered["Fireball_Level"].map(fireball_bets)

    valid = df_filtered[df_filtered["Correct"].notna()].copy()
    valid["Profit"] = valid.apply(lambda row: row["Bet_Size"] if row["Correct"] else -row["Bet_Size"] * 1.1, axis=1)
    summary = valid.groupby("Fireball 🔥 Tier").agg(
        Bets=("Correct", "count"),
        Amount_Staked=("Bet_Size", "sum"),
        Amount_Won=("Bet_Size", lambda s: s[valid.loc[s.index, "Correct"]].sum() * 2),
        Net_Profit=("Profit", "sum")
    ).reset_index()
    summary["ROI %"] = (summary["Net_Profit"] / summary["Amount_Staked"] * 100).round(1)

    st.dataframe(summary, use_container_width=True)
    st.altair_chart(
        alt.Chart(summary).mark_bar().encode(
            x="Fireball 🔥 Tier",
            y="ROI %",
            color=alt.Color("ROI %", scale=alt.Scale(scheme="redyellowgreen"))
        ).properties(title="📊 ROI by Fireball Tier"),
        use_container_width=True
    )

    st.divider()
    st.metric("Total Bets", summary["Bets"].sum())
    st.metric("Total Staked", f"${summary['Amount_Staked'].sum():,.2f}")
    st.metric("Net Profit", f"${summary['Net_Profit'].sum():,.2f}")
    st.metric("Overall ROI", f"{(summary['Net_Profit'].sum() / summary['Amount_Staked'].sum()) * 100:.1f}%")

# === Tab 4: Confidence Accuracy Breakdown
elif view == "Confidence Accuracy Breakdown":
    st.title("📊 Accuracy by Confidence Bucket")
    df["Confidence_Bucket"] = (df["Confidence"] * 10).astype(int) / 10
    buckets = df[df["Correct"].notna()].groupby("Confidence_Bucket")["Correct"].agg(["count", "sum"])
    buckets["Accuracy %"] = (buckets["sum"] / buckets["count"] * 100).round(1)
    st.bar_chart(buckets["Accuracy %"])
    st.dataframe(buckets.reset_index(), use_container_width=True)

# === Tab 5: Fireball Volume Over Time
elif view == "Fireball Volume Over Time":
    st.title("🔥 Fireball Volume by Date")
    fire_df = df[df["Confidence"].notna()].copy()
    fire_df["Fireball_Level"] = (fire_df["Confidence"] * 100).apply(lambda p: 5 if p >= 90 else 4 if p >= 80 else 3 if p >= 70 else 2 if p >= 60 else 1)
    counts = fire_df.groupby(["Game_Date", "Fireball_Level"]).size().unstack(fill_value=0)
    st.bar_chart(counts)

# === Tab 6: Fireball Profit Curve
elif view == "Fireball Profit Curve":
    st.title("📈 Profit Curve by Fireball Tier")
    p_df = df[df["Correct"].notna()].copy()
    p_df["Fireball_Level"] = (p_df["Confidence"] * 100).apply(lambda p: 5 if p >= 90 else 4 if p >= 80 else 3 if p >= 70 else 2 if p >= 60 else 1)
    p_df["Bet_Size"] = p_df["Fireball_Level"].map({5: 15, 4: 10, 3: 5, 2: 2.5, 1: 1})
    p_df["Profit"] = p_df.apply(lambda r: r["Bet_Size"] if r["Correct"] else -r["Bet_Size"] * 1.1, axis=1)
    cum = p_df.groupby(["Game_Date", "Fireball_Level"])["Profit"].sum().groupby(level=1).cumsum().unstack().fillna(method="ffill")
    st.line_chart(cum)

# === Tab 7: Top Daily Picks Leaderboard
elif view == "Top Daily Picks Leaderboard":
    st.title("🏅 Top Fireball Picks Per Day")

    df["Fireball_Level"] = (df["Confidence"] * 100).apply(
        lambda p: 5 if p >= 90 else 4 if p >= 80 else 3 if p >= 70 else 2 if p >= 60 else 1
    )
    df["Matchup"] = df["Away_Team"] + " @ " + df["Home_Team"]
    df["Date"] = df["Game_Date"].dt.date
    day = st.date_input(
    "📅 Pick a date",
    value=max(df["Date"]),
    min_value=min(df["Date"]),
    max_value=max(df["Date"])
)


    # Select top 5 fireball picks for this day
    top5 = df[(df["Date"] == day) & (df["Fireball_Level"] == 5)].sort_values(
        "Confidence", ascending=False
    ).copy()

    if top5.empty:
        st.warning("⚠️ No 5-fireball picks found for this day.")
    else:
        top5["Actual Runs"] = top5["Runs_1_5"].round(1).fillna("—")
        top5["Result"] = top5.apply(
            lambda row: (
                "✅" if (row["Bet"].startswith("OVER") and row["Actual Runs"] > 4.5) or
                         (row["Bet"].startswith("UNDER") and row["Actual Runs"] < 4.5)
                else "❌"
            ) if row["Actual Runs"] != "—" else "—",
            axis=1
        )

        st.dataframe(
            top5[["Matchup", "Model_Total", "Confidence", "Fireball_Level", "Actual Runs", "Result"]],
            use_container_width=True
        )

        # Daily summary
        valid = top5[top5["Result"].isin(["✅", "❌"])]
        hits = (valid["Result"] == "✅").sum()
        total = len(valid)
        acc = f"{(hits / total * 100):.1f}% ({hits}/{total})" if total > 0 else "—"
        st.metric("🔥 Daily Accuracy for Top Picks", acc)

        # Rolling accuracy up to selected day
        history = df[(df["Date"] <= day) & (df["Fireball_Level"] == 5)].copy()
        history = history[history["Runs_1_5"].notna()]
        history["Actual Runs"] = history["Runs_1_5"].round(1)
        history["Result"] = history.apply(
            lambda row: (
                "✅" if (row["Bet"].startswith("OVER") and row["Actual Runs"] > 4.5) or
                         (row["Bet"].startswith("UNDER") and row["Actual Runs"] < 4.5)
                else "❌"
            ) if row["Actual Runs"] != "—" else "—",
            axis=1
        )
        roll_valid = history[history["Result"].isin(["✅", "❌"])]
        roll_hits = (roll_valid["Result"] == "✅").sum()
        roll_total = len(roll_valid)
        roll_acc = f"{(roll_hits / roll_total * 100):.1f}% ({roll_hits}/{roll_total})" if roll_total > 0 else "—"
        st.metric("📈 Rolling Accuracy (All Time to This Day)", roll_acc)


# === Tab 8: Calendar Heatmap
elif view == "Calendar Heatmap":
    st.title("📅 Calendar Summary: Accuracy & Profit")
    cal = df[df["Correct"].notna()].copy()
    cal["Profit"] = cal.apply(lambda r: 100 if r["Correct"] else -110, axis=1)
    daily = cal.groupby("Game_Date").agg(
        Accuracy=("Correct", "mean"),
        Profit=("Profit", "sum"),
        Volume=("Correct", "count")
    ).reset_index()
    st.altair_chart(alt.Chart(daily).mark_bar().encode(
        x="Game_Date:T", y="Accuracy", color="Accuracy"
    ).properties(title="🎯 Accuracy % by Day"), use_container_width=True)
    st.altair_chart(alt.Chart(daily).mark_bar().encode(
        x="Game_Date:T", y="Profit", color=alt.condition("datum.Profit > 0", alt.value("green"), alt.value("red"))
    ).properties(title="💰 Profit by Day"), use_container_width=True)

# === Tab 9: Confidence Histogram
elif view == "Confidence Distribution Histogram":
    st.title("🧮 Model Confidence Histogram")
    hist = df[df["Confidence"].notna()]
    st.bar_chart(hist["Confidence"].round(2).value_counts().sort_index())
