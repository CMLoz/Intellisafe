"""spaCy-based entity detection for contextual validation."""

from __future__ import annotations

import logging
from typing import Dict, List

import spacy
from spacy.util import is_package

from .regex_engine import RegexEngine

logger = logging.getLogger(__name__)


class SpacyEngine:
    """Detect contextual entities using spaCy NER."""

    DEFAULT_MODEL = "en_core_web_md"
    ENTITY_MAP = {
        "PERSON": ("Person Name", "medium", 0.7),
        "ORG": ("Organization", "medium", 0.65),
        "GPE": ("Location", "low", 0.6),
        "LOC": ("Location", "low", 0.6),
        "FAC": ("Location", "low", 0.55),
        "NORP": ("Group", "low", 0.55),
    }

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._nlp = self._load_model(self.model_name)

    @staticmethod
    def _load_model(model_name: str):
        try:
            if not is_package(model_name):
                logger.warning("spaCy model package not found: %s", model_name)
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

    def detect(self, text: str) -> List[Dict]:
        """Return spaCy entity findings for extracted text."""
        if not text:
            return []

        doc = self._nlp(text)
        findings: List[Dict] = []
        seen = set()

        for ent in doc.ents:
            mapping = self.ENTITY_MAP.get(ent.label_)
            if not mapping:
                continue

            type_label, severity, confidence = mapping
            value = ent.text.strip()
            if not value:
                continue

            start, end = ent.start_char, ent.end_char
            key = (type_label, value, start, end)
            if key in seen:
                continue
            seen.add(key)

            findings.append(
                {
                    "type": type_label,
                    "label": ent.label_,
                    "value": value,
                    "masked_value": RegexEngine.mask_value(value),
                    "severity": severity,
                    "start": start,
                    "end": end,
                    "line": RegexEngine._line_number(text, start),
                    "context": RegexEngine._context(text, start, end),
                    "engine": "spacy",
                    "context_type": ent.label_,
                    "confidence": confidence,
                }
            )

        logger.info("spaCy detection found %s possible sensitive entities", len(findings))
        return findings
