# train_and_api.py
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()  # expects .env file for DB credentials

# DB connection string (postgres)
DB_URL = os.getenv("DB_URL") or "postgresql+psycopg2://postgres:q1w2e3@localhost:5432/sportsbetting"
engine = create_engine(DB_URL, pool_pre_ping=True)

# Where to save model/artifacts
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------- Utilities ----------------
def load_matches():
    sql = """
    SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
    FROM bettingschema.odds
    WHERE date IS NOT NULL
    ORDER BY date ASC;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    # normalize date
    df['date'] = pd.to_datetime(df['date'])
    return df

def parse_result_to_label(res):
    """
    Accepts many formats:
     - "H" / "D" / "A"
     - "1-0", "2:1" -> parse goals
    returns 0=Home, 1=Draw, 2=Away or np.nan if unknown
    """
    if pd.isna(res):
        return np.nan
    s = str(res).strip()
    # single letter
    if s in ('H','h','Home','home'):
        return 0
    if s in ('D','d','Draw','draw'):
        return 1
    if s in ('A','a','Away','away'):
        return 2
    # goals style
    if '-' in s or ':' in s:
        sep = '-' if '-' in s else ':'
        try:
            a,b = s.split(sep)
            a = int(a); b = int(b)
            if a>b: return 0
            if a==b: return 1
            return 2
        except:
            return np.nan
    return np.nan

# -------------- Elo rating functions ----------------
def init_elo(df, start_rating=1500):
    teams = pd.unique(df[['home_team','away_team']].values.ravel())
    elo = {t: start_rating for t in teams}
    return elo

def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def update_elo(elo, home, away, score, k_home=20, k_away=20, home_field_advantage=100):
    # score: 1=home win, 0.5 draw, 0 away win
    Ra = elo.get(home,1500) + home_field_advantage
    Rb = elo.get(away,1500)
    Ea = expected_score(Ra, Rb)
    Eb = 1 - Ea
    Sa = score
    Sb = 1 - score
    elo[home] = elo.get(home,1500) + k_home * (Sa - Ea)
    elo[away] = elo.get(away,1500) + k_away * (Sb - Eb)
    return elo

# -------------- Feature engineering ----------------
def compute_features(df, n_last=5):
    """
    df: historical matches sorted by date ascending
    returns new DataFrame with features and target
    """
    df = df.copy().reset_index(drop=True)
    # parse target
    df['target'] = df['result'].apply(parse_result_to_label)

    # implied probabilities from odds (bookmaker)
    def implied_prob(row):
        try:
            p1 = 1.0 / float(row['odd_1']) if pd.notna(row['odd_1']) and row['odd_1']>0 else np.nan
            pX = 1.0 / float(row['odd_X']) if pd.notna(row['odd_X']) and row['odd_X']>0 else np.nan
            p2 = 1.0 / float(row['odd_2']) if pd.notna(row['odd_2']) and row['odd_2']>0 else np.nan
            s = sum([x for x in (p1,pX,p2) if pd.notna(x)])
            if s>0:
                return pd.Series([p1/s if pd.notna(p1) else 0,
                                  pX/s if pd.notna(pX) else 0,
                                  p2/s if pd.notna(p2) else 0])
            else:
                return pd.Series([np.nan,np.nan,np.nan])
        except:
            return pd.Series([np.nan,np.nan,np.nan])
    df[['imp_prob_H','imp_prob_D','imp_prob_A']] = df.apply(implied_prob, axis=1)

    # initialize elo & rolling stats
    elo = init_elo(df)
    home_elos=[]; away_elos=[]
    home_goal_diff_last = []  # placeholder if you have goals column (not in current assumption)
    # We'll compute rolling stats using a helper dictionary of match history per team
    history = {}  # team -> list of last results (1=win,0.5 draw,0 loss), goals_for, goals_against if known

    for idx, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        # append current elos
        home_elos.append(elo.get(h,1500))
        away_elos.append(elo.get(a,1500))

        # recent form features (last n matches)
        def last_stats(team, n=n_last):
            hist = history.get(team, [])
            last = hist[-n:] if len(hist)>0 else []
            if len(last)==0:
                return {
                    'form_wins': 0.0,
                    'form_points': 0.0,
                    'form_matches': 0
                }
            pts = 0.0
            wins = 0
            for rec in last:
                if rec==1: wins+=1; pts+=3
                elif rec==0.5: pts+=1
            return {'form_wins': wins/len(last), 'form_points': pts/(3*len(last)), 'form_matches': len(last)}

        # compute features for match from both teams
        hstats = last_stats(h)
        astats = last_stats(a)
        # put features into df row later
        for k,v in hstats.items():
            df.at[idx, f'h_{k}'] = v
        for k,v in astats.items():
            df.at[idx, f'a_{k}'] = v

        # head-to-head: naive count of home-team recent wins vs away-team
        pair_key = tuple(sorted([h,a]))
        # we'll count last 5 head-to-head results where h was home and a away (simple)
        # gather recent h2h from df earlier rows
        h2h_df = df.loc[:idx-1]
        mask = ((h2h_df['home_team']==h)&(h2h_df['away_team']==a))|((h2h_df['home_team']==a)&(h2h_df['away_team']==h))
        last_h2h = h2h_df.loc[mask].tail(n_last)
        # compute h2h advantage as fraction of matches that home (current home) won
        if len(last_h2h)>0:
            wins_by_hometeam = 0
            for _,r in last_h2h.iterrows():
                lab = parse_result_to_label(r['result'])
                # determine whether the match winner was the same team as current home
                if lab==0 and r['home_team']==h: wins_by_hometeam+=1
                if lab==2 and r['away_team']==h: wins_by_hometeam+=1
                if lab==0 and r['home_team']==a: wins_by_hometeam-=1
                if lab==2 and r['away_team']==a: wins_by_hometeam-=1
            df.at[idx, 'h2h_adv'] = wins_by_hometeam / len(last_h2h)
        else:
            df.at[idx, 'h2h_adv'] = 0.0

        # update history & elo only if result available
        t = df.at[idx,'target']
        if not pd.isna(t):
            # score for elo update uses home perspective: 1=home win, 0.5 draw, 0 away win
            if t==0: score = 1.0
            elif t==1: score = 0.5
            else: score = 0.0
            # update elo
            update_elo(elo, h, a, score)
            # update history lists
            history.setdefault(h,[]).append(score)
            history.setdefault(a,[]).append(1.0-score)
        else:
            # future match: just append nothing
            pass

    df['elo_home'] = home_elos
    df['elo_away'] = away_elos

    # feature: elo_diff
    df['elo_diff'] = df['elo_home'] - df['elo_away']

    # common categorical features (league encoded later)
    # fill na for implied probs with zeros (or better, median)
    df['imp_prob_H'] = df['imp_prob_H'].fillna(0.33)
    df['imp_prob_D'] = df['imp_prob_D'].fillna(0.34)
    df['imp_prob_A'] = df['imp_prob_A'].fillna(0.33)

    # fill other feature nans
    df['h_form_wins'] = df['h_form_wins'].fillna(0.0)
    df['a_form_wins'] = df['a_form_wins'].fillna(0.0)
    df['h_form_points'] = df['h_form_points'].fillna(0.0)
    df['a_form_points'] = df['a_form_points'].fillna(0.0)
    df['h2h_adv'] = df['h2h_adv'].fillna(0.0)

    # select final features
    feat_cols = [
        'elo_home','elo_away','elo_diff',
        'h_form_wins','h_form_points',
        'a_form_wins','a_form_points',
        'h2h_adv',
        'imp_prob_H','imp_prob_D','imp_prob_A',
        'country','league'
    ]
    # ensure columns exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0

    return df, feat_cols

# -------------- Train model ----------------
def train_save_model(df, feat_cols, target_col='target'):
    # drop rows with no target (future matches)
    df_train = df[df[target_col].notna()].copy()
    df_train = df_train.sort_values('date')
    # encode categorical features: country & league -> label encode
    le_country = LabelEncoder()
    le_league = LabelEncoder()
    df_train['country_enc'] = le_country.fit_transform(df_train['country'].astype(str))
    df_train['league_enc'] = le_league.fit_transform(df_train['league'].astype(str))

    # replace categorical in feat_cols with enc columns
    used_features = [c for c in feat_cols if c not in ('country','league')]
    used_features += ['country_enc','league_enc']

    X = df_train[used_features].fillna(0)
    y = df_train[target_col].astype(int)

    # Use time-aware split for validation
    # We'll train on all but last 10% by time
    split_idx = int(len(df_train) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective':'multi:softprob',
        'num_class':3,
        'eval_metric':'mlogloss',
        'eta':0.05,
        'max_depth':6,
        'subsample':0.8,
        'colsample_bytree':0.7,
        'seed':42,
        'verbosity':1
    }
    evallist = [(dtrain,'train'), (dval, 'eval')]
    num_boost_round = 1000
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evallist,
                      early_stopping_rounds=50, verbose_eval=50)

    # metrics
    preds_val = model.predict(dval)
    val_loss = log_loss(y_val, preds_val)
    val_pred_labels = np.argmax(preds_val, axis=1)
    val_acc = accuracy_score(y_val, val_pred_labels)
    print(f"Validation LogLoss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save artifacts
    joblib.dump(model, f"{ARTIFACT_DIR}/xgb_model.joblib")
    joblib.dump(le_country, f"{ARTIFACT_DIR}/le_country.joblib")
    joblib.dump(le_league, f"{ARTIFACT_DIR}/le_league.joblib")
    # also save used feature list
    joblib.dump(used_features, f"{ARTIFACT_DIR}/features.joblib")
    print("Saved model and encoders to", ARTIFACT_DIR)
    return model, used_features, le_country, le_league

# -------------- Prediction functions ----------------
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
    # convert to timestamp first
    match_date_ts = pd.to_datetime(match_date)

    # load matches up to match_date
    sql = text("""
        SELECT season, date, time, home_team, away_team, result, odd_1, "odd_X", odd_2, bets, country, league
        FROM bettingschema.odds
        WHERE date < :dt
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        hist = pd.read_sql(sql, conn, params={'dt': match_date_ts})

    # ensure all dates are Timestamp
    hist['date'] = pd.to_datetime(hist['date'])


    # append a dummy row for the match we're predicting (with NaN result & odds if not present)
    match_date_ts = pd.to_datetime(match_date)
    new_row = {
        'season': None, 'date': match_date_ts,
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

    # re-use compute_features to produce features including elo / recent form up to the match
    df_feats, feat_cols = compute_features(hist, n_last=n_last)
    # last row is the match we want
    match_row = df_feats.iloc[-1].copy()

    # load encoders
    model, used_features, le_country, le_league = load_artifacts()
    # prepare features same order
    match_row['country_enc'] = le_country.transform([str(match_row['country'])])[0] if str(match_row['country']) in le_country.classes_ else -1
    match_row['league_enc'] = le_league.transform([str(match_row['league'])])[0] if str(match_row['league']) in le_league.classes_ else -1

    X = match_row[used_features].fillna(0).values.reshape(1,-1)
    dmat = xgb.DMatrix(X, feature_names=used_features)
    proba = model.predict(dmat)[0]  # [p_home, p_draw, p_away]
    return {'p_home': float(proba[0]), 'p_draw': float(proba[1]), 'p_away': float(proba[2])}

# -------------- p;Main training flow --------------
def main_train():
    df = load_matches()
    print("Loaded", len(df), "rows")
    df_sorted = df.sort_values('date')
    df_feats, feat_cols = compute_features(df_sorted, n_last=5)
    model, used_features, le_country, le_league = train_save_model(df_feats, feat_cols)

# if __name__ == "__main__":
    # run training if called directly
    # main_train()

