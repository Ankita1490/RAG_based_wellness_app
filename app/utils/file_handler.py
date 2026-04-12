import json
from typing import Any
from pathlib import Path


def ensure_directory(path:Path) -> None:
    """Creates the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, file_path: Path) -> None:
    """Saves data to a JSON file."""
    ensure_directory(file_path.parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_text(text: str, file_path: Path) -> None:
    """Saves text to a file."""
    ensure_directory(file_path.parent)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_json(file_path: Path) -> Any:
    """Loads data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)