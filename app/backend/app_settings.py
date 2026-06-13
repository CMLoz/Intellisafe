"""Persistent application settings stored in SQLite."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "default_scan_mode": "standard",
    "default_redaction_strategy": "blackout",
    "default_redaction_mode": "auto",
    "redaction_output_dir": "",
    "reload_session_on_startup": True,
    "rescan_after_redaction": True,
}


class AppSettings:
    def __init__(self, db_manager):
        self.db = db_manager
        self._cache: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        stored = self.db.get_all_settings()
        self._cache = {**DEFAULTS, **stored}
        if not self._cache.get("redaction_output_dir"):
            self._cache["redaction_output_dir"] = str(
                Path(__file__).resolve().parents[2] / "redacted_output"
            )
        return dict(self._cache)

    def get(self, key: str, default: Any = None) -> Any:
        if not self._cache:
            self.load()
        return self._cache.get(key, default if default is not None else DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        if not self._cache:
            self.load()
        self._cache[key] = value
        self.db.set_setting(key, json.dumps(value) if isinstance(value, (dict, list)) else str(value))

    def save_many(self, values: Dict[str, Any]) -> None:
        if not self._cache:
            self.load()
        for key, value in values.items():
            self.set(key, value)
