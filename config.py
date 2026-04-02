from pathlib import Path

from dotenv import load_dotenv

DOTENV_PATH_LOCAL = Path("/Users/aidensamuel/Desktop/.env")
DOTENV_PATH_REPO = Path(".env")

# Prefer local machine path, but fall back to repo .env (used in GitHub Actions).
if DOTENV_PATH_LOCAL.exists():
    load_dotenv(dotenv_path=DOTENV_PATH_LOCAL, override=True)
else:
    load_dotenv(dotenv_path=DOTENV_PATH_REPO, override=True)
