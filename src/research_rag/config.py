"""Load settings from config.yaml and the Ollama address from the environment."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """config.yaml by default; RAG_CONFIG=config_vqa.yaml selects another collection."""
    path = path or ROOT / os.environ.get("RAG_CONFIG", "config.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["ollama_host"] = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return config


def questions_path(config: dict) -> Path:
    return ROOT / config["paths"].get("questions", "eval/questions.jsonl")
