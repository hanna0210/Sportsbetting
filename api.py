from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from train_and_save import prepare_single_match_features
app = FastAPI(title="Football Bet Predictor")

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    country: str
    date: str   # format YYYY-MM-DD

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = prepare_single_match_features(
            home=req.home_team,
            away=req.away_team,
            league=req.league,
            country=req.country,
            match_date=req.date
        )

        # ✅ Always return success + probabilities
        return {
            "success": True,
            "p_home": result.get("p_home", None),
            "p_draw": result.get("p_draw", None),
            "p_away": result.get("p_away", None)
        }

    except Exception as e:
        # ✅ Always return error inside same structure
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "p_home": None,
            "p_draw": None,
            "p_away": None
        }, status_code=200)
