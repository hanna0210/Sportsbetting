import joblib, xgboost as xgb
import pandas as pd
from core.features import compute_features
from core.db import load_matches
from core.config import ARTIFACT_DIR
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # load DB_URL from .env

DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.joblib")
    le_country = joblib.load(f"{ARTIFACT_DIR}/le_country.joblib")
    le_league = joblib.load(f"{ARTIFACT_DIR}/le_league.joblib")
    used_features = joblib.load(f"{ARTIFACT_DIR}/features.joblib")
    return model, used_features, le_country, le_league

def prepare_single_match_features(home, away, league, country, match_date, n_last=5):
    """
    Compute features for a single upcoming match by querying historical DB rows and building same features as training.
    """

    # 1️⃣ Convert match_date to datetime
    match_date_ts = pd.to_datetime(match_date)

    # 2️⃣ Load historical matches BEFORE match_date
    sql = text("""
        SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
        FROM bettingschema.odds
        WHERE date < :dt
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        hist = pd.read_sql(sql, conn, params={'dt': match_date_ts})

    # 3️⃣ Ensure 'date' column is datetime
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')

    # 4️⃣ Append dummy row for the match we're predicting
    new_row = {
        'season': None,
        'date': match_date_ts,
        'time': None,
        'home_team': home,
        'away_team': away,
        'result': None,
        'odd_1': None,
        'odd_X': None,
        'odd_2': None,
        'bets': None,
        'country': country,
        'league': league
    }

    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    hist = hist.sort_values('date').reset_index(drop=True)

    # 5️⃣ Compute features (elo, form, h2h, etc.)
    df_feats, feat_cols = compute_features(hist, n_last=n_last)

    # 6️⃣ Take last row (our upcoming match)
    match_row = df_feats.iloc[-1].copy()

    # 7️⃣ Load model artifacts
    model, used_features, le_country, le_league = load_artifacts()

    # 8️⃣ Encode categorical features safely
    if str(match_row['country']) in le_country.classes_:
        match_row['country_enc'] = le_country.transform([str(match_row['country'])])[0]
    else:
        match_row['country_enc'] = -1

    if str(match_row['league']) in le_league.classes_:
        match_row['league_enc'] = le_league.transform([str(match_row['league'])])[0]
    else:
        match_row['league_enc'] = -1

    # 9️⃣ Prepare feature matrix
    X = match_row[used_features].fillna(0).values.reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=used_features)

    # 10️⃣ Predict probabilities
    proba = model.predict(dmat)[0]  # [p_home, p_draw, p_away]

    return {
        'p_home': float(proba[0]),
        'p_draw': float(proba[1]),
        'p_away': float(proba[2])
    }
