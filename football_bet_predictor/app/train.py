# football_bet_predictor/app/train.py
import joblib, xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.db import load_matches
from core.features import compute_features
from core.calibration import TemperatureScaler
from core.config import ARTIFACT_DIR

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------
# Utility functions
# -------------------------------
def cross_entropy(true_probs, pred_probs, eps=1e-15):
    """Cross-entropy between two probability distributions."""
    true_probs = np.clip(true_probs, eps, 1 - eps)
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    return -np.mean(np.sum(true_probs * np.log(pred_probs), axis=1))

def bookmaker_implied_probs_row(row):
    """Compute normalized probabilities from bookmaker odds."""
    def inv(x):
        try:
            return 1.0 / float(x) if x and float(x) > 0 else 0.0
        except:
            return 0.0
    p1 = inv(row.get('odd_1'))
    pX = inv(row.get('odd_X') or row.get('odd_x'))  # defensive
    p2 = inv(row.get('odd_2'))
    s = p1 + pX + p2
    if s <= 0:
        return np.array([1/3, 1/3, 1/3])
    return np.array([p1/s, pX/s, p2/s])

# -------------------------------
# Training function
# -------------------------------
def train_save_model():
    # 1️⃣ Load matches and compute features
    df = load_matches()
    df_feats, feat_cols = compute_features(df)
    
    # 2️⃣ Encode categorical features
    le_country = LabelEncoder()
    le_league = LabelEncoder()
    df_feats['country_enc'] = le_country.fit_transform(df_feats['country'].astype(str))
    df_feats['league_enc'] = le_league.fit_transform(df_feats['league'].astype(str))

    used_features = [c for c in feat_cols if c not in ('country','league')] + ['country_enc','league_enc']

    # Only rows with labels for training
    df_train = df_feats[df_feats['target'].notna()].copy()
    X = df_train[used_features].fillna(0)
    y = df_train['target'].astype(int)

    # 3️⃣ Split into train/validation (chronological)
    split_idx = int(len(df_train) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train,label=y_train)
    dval = xgb.DMatrix(X_val,label=y_val)

    # 4️⃣ Train XGBoost
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
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain,'train'),(dval,'eval')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val", X_val.shape, "y_val", y_val.shape)

    # 5️⃣ Raw metrics vs actual match outcomes
    preds_val = model.predict(dval)
    print("LogLoss (raw):", log_loss(y_val,preds_val), 
          "Acc:", accuracy_score(y_val, preds_val.argmax(axis=1)))

    # -------------------------
    # 6️⃣ Calibration vs bookmaker odds
    # -------------------------
    # Raw logits
    logits_val = model.predict(dval, output_margin=True)

    # Bookmaker implied probs for validation
    odds_cols = df_train.iloc[split_idx:][['odd_1','odd_X','odd_2']].fillna(0).to_dict(orient='records')
    market_probs = np.vstack([bookmaker_implied_probs_row(r) for r in odds_cols])

    # Fit TemperatureScaler
    ts = TemperatureScaler()
    ts.fit(logits_val, market_probs, initial_T=1.0)
    print("Fitted temperature T (to market):", ts.T)

    # Calibrated probabilities
    calibrated_probs = ts.transform_proba(logits_val)

    # Cross-entropy vs market
    print("Cross-Entropy (calibrated->market):", cross_entropy(market_probs, calibrated_probs))

    # -------------------------
    # 7️⃣ Save artifacts
    # -------------------------
    joblib.dump(model, f"{ARTIFACT_DIR}/xgb_model.joblib")
    joblib.dump(le_country, f"{ARTIFACT_DIR}/le_country.joblib")
    joblib.dump(le_league, f"{ARTIFACT_DIR}/le_league.joblib")
    joblib.dump(used_features, f"{ARTIFACT_DIR}/features.joblib")
    ts.save(f"{ARTIFACT_DIR}/temp_scaler.joblib")

    print("Saved artifacts to", ARTIFACT_DIR)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    train_save_model()


# [0]     train-mlogloss:1.08821  eval-mlogloss:1.08909
# [50]    train-mlogloss:0.93248  eval-mlogloss:0.97894
# [100]   train-mlogloss:0.89321  eval-mlogloss:0.98002
# [105]   train-mlogloss:0.89040  eval-mlogloss:0.98008
# X_train (17356, 13) y_train (17356,)
# X_val (1929, 13) y_val (1929,)
# LogLoss (raw): 0.9800774678118033 Acc: 0.5308449974079834
# Fitted temperature T (to market): 1.0
# Cross-Entropy (calibrated->market): 1.4198342218755562
# Saved artifacts to ./models/artifacts