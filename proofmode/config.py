from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "ProofMode"
    model_name: str = os.getenv("PROOFMODE_MODEL", "gemma-4-e4b-it-q4")
    gemma_base_url: str = os.getenv("PROOFMODE_GEMMA_URL", "http://127.0.0.1:8080/v1")
    gemma_api_key: str = os.getenv("PROOFMODE_GEMMA_KEY", "local-gemma")
    database_path: Path = Path(
        os.getenv("PROOFMODE_DB", str(PROJECT_ROOT / "data" / "proofmode.db"))
    )
    request_timeout_seconds: float = float(os.getenv("PROOFMODE_TIMEOUT", "120"))
    max_evidence_chars: int = int(os.getenv("PROOFMODE_MAX_EVIDENCE", "18000"))
    demo_mode: bool = os.getenv("PROOFMODE_DEMO", "1").lower() not in {"0", "false", "off"}


settings = Settings()
