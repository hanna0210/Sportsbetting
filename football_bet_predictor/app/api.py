from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from .predict import prepare_single_match_features

app = FastAPI(title="Football Bet Predictor")

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    country: str
    date: str

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = prepare_single_match_features(
            req.home_team, req.away_team, req.league, req.country, req.date
        )
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e),
             "p_home": None, "p_draw": None, "p_away": None,
             "odd_1": None, "odd_X": None, "odd_2": None},
            status_code=200
        )

from .predict import prepare_single_match_features, prepare_upcoming_matches

@app.get("/predict_upcoming")
def predict_upcoming(limit: int = 50):
    try:
        result = prepare_upcoming_matches(limit=limit)
        return {"success": True, "matches": result}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=200)
