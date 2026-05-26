from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = one folder above /src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# Explicit dotenv path avoids python-dotenv edge cases in stdin / notebooks
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()

CHROMA_DIR = str(PROJECT_ROOT / ".chroma_great")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file in the project root "
        "and add OPENAI_API_KEY=sk-..."
    )
