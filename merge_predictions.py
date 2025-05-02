import pandas as pd
import os
from datetime import datetime

TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
}

# === Load data
box = pd.read_csv("data/mlb_boxscores_full.csv")
preds = pd.read_csv("data/mlb_predictions.csv")

box.columns = box.columns.str.strip().str.replace(" ", "_")
preds.columns = preds.columns.str.strip().str.replace(" ", "_")

# 🧼 Drop stale prediction columns
if "Runs_1_5" in preds.columns:
    print("🧼 Dropping stale Runs_1_5 column from predictions...")
    preds = preds.drop(columns=["Runs_1_5"])

if "Game_Date" not in preds.columns:
    if "Date" in preds.columns:
        preds.rename(columns={"Date": "Game_Date"}, inplace=True)
    else:
        raise ValueError("Missing Game_Date or Date column in predictions")

preds.rename(columns={"Home": "Home_Team", "Away": "Away_Team"}, inplace=True)

# === Normalize fields
preds["Game_Date"] = pd.to_datetime(preds["Game_Date"])
box["Game_Date"] = pd.to_datetime(box["Game_Date"])
preds["Home_Team"] = preds["Home_Team"].str.strip()
preds["Away_Team"] = preds["Away_Team"].str.strip()
box["Home_Team"] = box["Home_Team"].str.strip().replace(TEAM_NAME_MAP).fillna(box["Home_Team"])
box["Away_Team"] = box["Away_Team"].str.strip().replace(TEAM_NAME_MAP).fillna(box["Away_Team"])

# === Compute Runs_1_5
expected_innings = ["1th", "2th", "3th", "4th", "5th"]
inning_cols = [f"{side}_{inn}" for inn in expected_innings for side in ["Away", "Home"] if f"{side}_{inn}" in box.columns]
box[inning_cols] = box[inning_cols].apply(pd.to_numeric, errors="coerce")
box["Runs_1_5"] = box[inning_cols].sum(axis=1)

print(f"📊 Detected inning columns: {inning_cols}")
print("✅ Sample Runs_1_5 values:")
print(box[["Game_Date", "Away_Team", "Home_Team", "Runs_1_5"]].head())

# 🛑 Filter to only past games
today = pd.to_datetime(datetime.now().date())
pre_filter_count = len(box)
box = box[box["Game_Date"] < today]
print(f"📉 Filtered boxscores to past games: {len(box)} rows (removed {pre_filter_count - len(box)})")

# === Merge
merged = pd.merge(
    preds,
    box[["Game_Date", "Away_Team", "Home_Team", "Runs_1_5"]],
    on=["Game_Date", "Away_Team", "Home_Team"],
    how="left"
)

# === Diagnostics
if "Runs_1_5" in merged.columns:
    missing_count = merged["Runs_1_5"].isna().sum()
    print(f"⚠️ {missing_count} predictions missing actual 1-5 run totals")
    if missing_count > 0:
        merged[merged["Runs_1_5"].isna()].to_csv("unmatched_rows.csv", index=False)
        print("📄 Exported unmatched rows to unmatched_rows.csv")
else:
    print("❌ Merge failed: Runs_1_5 column not present in merged DataFrame.")
    print(f"🔍 Merged columns: {merged.columns.tolist()}")

# === Save
merged.to_csv("data/mlb_predictions_merged.csv", index=False)
print("✅ Merged predictions saved to mlb_predictions_merged.csv")
