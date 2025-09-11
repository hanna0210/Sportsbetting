import joblib, xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from core.db import load_matches
from core.features import compute_features
from core.config import ARTIFACT_DIR
import os

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def train_save_model():
    df = load_matches()
    df_feats, feat_cols = compute_features(df)
    
    # Encode categories
    le_country = LabelEncoder()
    le_league = LabelEncoder()
    df_feats['country_enc'] = le_country.fit_transform(df_feats['country'].astype(str))
    df_feats['league_enc'] = le_league.fit_transform(df_feats['league'].astype(str))

    used_features = [c for c in feat_cols if c not in ('country','league')] + ['country_enc','league_enc']
    df_train = df_feats[df_feats['target'].notna()].copy()  # only rows with label
    X = df_train[used_features].fillna(0)
    y = df_train['target'].astype(int)


    # Time split
    split_idx = int(len(df_train) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train,label=y_train)
    dval = xgb.DMatrix(X_val,label=y_val)

    params = {'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','eta':0.05,'max_depth':6,'subsample':0.8,'colsample_bytree':0.7,'seed':42,'verbosity':1}
    model = xgb.train(params,dtrain,num_boost_round=1000,evals=[(dtrain,'train'),(dval,'eval')],early_stopping_rounds=50,verbose_eval=50)

    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val", X_val.shape, "y_val", y_val.shape)

    # Metrics
    preds_val = model.predict(dval)
    print("LogLoss:", log_loss(y_val,preds_val), "Acc:", accuracy_score(y_val, preds_val.argmax(axis=1)))

    # Save
    joblib.dump(model,f"{ARTIFACT_DIR}/xgb_model.joblib")
    joblib.dump(le_country,f"{ARTIFACT_DIR}/le_country.joblib")
    joblib.dump(le_league,f"{ARTIFACT_DIR}/le_league.joblib")
    joblib.dump(used_features,f"{ARTIFACT_DIR}/features.joblib")

if __name__ == "__main__":
    train_save_model()



# [0]     train-mlogloss:1.08821  eval-mlogloss:1.08909
# [50]    train-mlogloss:0.93248  eval-mlogloss:0.97894
# [100]   train-mlogloss:0.89321  eval-mlogloss:0.98002
# [104]   train-mlogloss:0.89107  eval-mlogloss:0.98023
# X_train (17356, 13) y_train (17356,)
# X_val (1929, 13) y_val (1929,)
# LogLoss: 0.9800774678118033 Acc: 0.5308449974079834

# xgb_model.joblib → trained XGBoost model
# le_country.joblib → LabelEncoder for country
# le_league.joblib → LabelEncoder for league
# features.joblib → list of feature column names