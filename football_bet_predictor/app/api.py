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
        result = prepare_single_match_features(req.home_team, req.away_team, req.league, req.country, req.date)
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e), "p_home": None, "p_draw": None, "p_away": None}, status_code=200)
