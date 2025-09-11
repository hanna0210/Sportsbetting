# scripts/sample_call.py
import requests

# Make sure your API is running (uvicorn app.api:app --reload)
url = "http://127.0.0.1:8000/predict"

payload = {
    "home_team": "Manchester United",
    "away_team": "Liverpool",
    "league": "Premier League",
    "country": "England",
    "date": "2025-09-20"   # format YYYY-MM-DD
}

response = requests.post(url, json=payload)
data = response.json()

# Print raw response
print("Raw response:", data)

# Handle success/failure
if not data.get("success", False):
    print("API error:", data.get("error"))
else:
    print("Home win probability:", data.get("p_home"))
    print("Draw probability:", data.get("p_draw"))
    print("Away win probability:", data.get("p_away"))
