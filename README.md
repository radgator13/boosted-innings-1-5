# 🔥 MLB Over 4.5 Innings Prediction Dashboard

This project provides an automated pipeline and Streamlit app to predict and evaluate MLB games on whether the total runs scored in the first 5 innings exceed 4.5.

---

## 📦 Project Structure

```plaintext
.
├── app.py                            # Streamlit dashboard for predictions & analytics
├── run_pipeline.py                  # Main orchestration script
├── Scrape_Fan_Graph.py              # Pulls pitcher/batter data from FanGraphs
├── get_scores_full.py               # Pulls final scores for actual result mapping
├── predict_over_4_5.py              # Generates model-based predictions
├── backfill_predict_over_4_5.py     # (Optional) Backfills missed predictions
├── train_model.py                   # (Optional) Retrains the predictive model
├── merge_predictions.py             # Merges predictions + actuals into final CSV
├── requirements.txt                 # Python dependencies for the full app
└── data/
    └── mlb_predictions_merged.csv   # ✅ Final dataset consumed by app.py
