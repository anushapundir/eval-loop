"""Central configuration: model provider switch, paths, thresholds, sample sizes.

Nothing in the codebase hardcodes a path, model name, or threshold — it reads
from here. Flipping ``MODEL_PROVIDER`` or a threshold touches only this file.
Values are read from environment / ``.env`` via pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (config/ -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed application settings, loaded once and cached.

    Cost-safe defaults (CLAUDE.md §2): the high-volume agent runs locally on
    Ollama; the paid Haiku model is reserved for the sampled LLM judge.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model providers -------------------------------------------------
    # Default agent runtime. "ollama" = free/local; "haiku" = all-Haiku mode.
    model_provider: Literal["ollama", "haiku"] = Field(default="ollama")
    anthropic_api_key: str | None = Field(default=None)

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")
    haiku_model: str = Field(default="claude-haiku-4-5-20251001")

    # Generation controls.
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1024)
    request_timeout_s: float = Field(default=120.0)
    max_retries: int = Field(default=3)

    # --- Evaluation knobs ------------------------------------------------
    # Score threshold a response must reach to "pass".
    pass_threshold: float = Field(default=0.7)
    # Fraction of the dataset the paid judge scores (deterministic checks run
    # on the full set; only the judge is sampled to control cost).
    judge_sample_rate: float = Field(default=0.2)
    judge_sample_seed: int = Field(default=42)

    # --- Feedback loop ---------------------------------------------------
    max_iterations: int = Field(default=2)

    # --- Paths -----------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")
    db_path: Path = Field(default=REPO_ROOT / "data" / "traces.db")
    demo_db_path: Path = Field(default=REPO_ROOT / "data" / "demo_results.db")
    kb_dir: Path = Field(default=REPO_ROOT / "datasets" / "kb")
    golden_dir: Path = Field(default=REPO_ROOT / "datasets" / "golden")
    synthetic_dir: Path = Field(default=REPO_ROOT / "datasets" / "synthetic")
    reports_output_dir: Path = Field(default=REPO_ROOT / "reports" / "output")

    def ensure_dirs(self) -> None:
        """Create the runtime directories that are produced fresh each run."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings singleton."""
    return Settings()
