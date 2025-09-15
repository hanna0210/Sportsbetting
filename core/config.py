import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:q1w2e3@localhost:5432/sportsbetting")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models/artifacts")
