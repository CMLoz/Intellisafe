"""GLiNER-based entity detection wrapper.

This module provides a thin adapter around a GLiNER-style API. It is
defensive about imports and maps results into the project's common
`finding` dictionary format so the `ValidationPipeline` can merge
results from multiple detectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class GLiNEREngine:
    """Adapter for GLiNER. Attempts to import a GLiNER implementation and
    normalize its output.

    Expected behaviour (best-effort): a GLiNER object exposes a callable
    that accepts text and returns a list of entity dicts containing at
    minimum `start`, `end`, `label`, `score`, and `text` keys. If the
    real runtime differs, adjust the adapter accordingly.
    """

    DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
    FALLBACK_MODELS = (
        "urchade/gliner_medium-v2.1",
        "urchade/gliner_base",
        "urchade/gliner_small-v2.1",
    )
    # Minimum confidence threshold — raised to 0.50 to cut low-confidence
    # label-text false positives (e.g. "Phone" detected as a phone number).
    CONFIDENCE_THRESHOLD = 0.50
    # Minimum character length for a detected value to be kept.
    MIN_VALUE_LENGTH = 3

    LABEL_CANDIDATES = [
        "person",
        "organization",
        "location",
        "address",
        "email address",
        "phone number",
        "passport number",
        "credit card",
        "bank account",
        "date of birth",
        "id number",
        "social security number",
        "tax id number",
        "medical record number",
        "driver license number",
        "ip address",
        "api key",
        "password",
    ]

    LABEL_MAP = {
        "person": "Person Name",
        "organization": "Organization",
        "location": "Location",
        "address": "Location",
        "email address": "Email",
        "phone number": "Phone Number",
        "passport number": "ID Number",
        "credit card": "Credit Card",
        "bank account": "Bank Account",
        "date of birth": "Date of Birth",
        "id number": "ID Number",
        "social security number": "ID Number",
        "tax id number": "ID Number",
        "medical record number": "ID Number",
        "driver license number": "ID Number",
        "ip address": "IP Address",
        "api key": "API Key",
        "password": "Password",
    }

    HIGH_RISK_LABELS = {"Credit Card", "Bank Account", "ID Number", "API Key", "Password"}
    MEDIUM_RISK_LABELS = {"Email", "Phone Number", "Date of Birth", "IP Address"}

    # Minimum digit count required per entity type (prevents label-text false positives).
    DIGIT_REQUIREMENTS: dict = {
        "Phone Number": 4,
        "Credit Card": 8,
        "ID Number": 3,
        "Date of Birth": 1,
    }

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = self._load_model(self.model_name)
        self._disabled = self._model is None

    @staticmethod
    def _load_model(model_name: str):
        candidates = list(dict.fromkeys([model_name, *GLiNEREngine.FALLBACK_MODELS]))

        gliner_cls = None
        try:
            # Try common GLiNER import patterns; be explicit about failure.
            try:
                from gliner import GLiNER  # type: ignore
                gliner_cls = GLiNER
            except Exception:
                import gliner as _gliner  # type: ignore
                if hasattr(_gliner, "GLiNER"):
                    gliner_cls = _gliner.GLiNER
        except ImportError as exc:
            raise RuntimeError(
                "GLiNER is not installed. Install with: pip install gliner"
            ) from exc

        if gliner_cls is None:
            logger.error("GLiNER package is present but no usable GLiNER class was found.")
            return None

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                if hasattr(gliner_cls, "from_pretrained"):
                    return gliner_cls.from_pretrained(candidate)
                return gliner_cls(candidate)
            except Exception as exc:
                last_error = exc

        if candidates:
            logger.warning(
                "GLiNER model '%s' unavailable; contextual NER disabled. %s",
                candidates[0],
                str(last_error).split("\n")[0] if last_error else "",
            )
        return None

    @classmethod
    def _is_valid_content(cls, normalized_label: str, value: str) -> bool:
        """Return False when the value is clearly a label/placeholder rather than real PII."""
        digit_count = sum(c.isdigit() for c in value)
        min_digits = cls.DIGIT_REQUIREMENTS.get(normalized_label)
        if min_digits is not None and digit_count < min_digits:
            return False
        if normalized_label == "Email" and "@" not in value:
            return False
        if normalized_label == "API Key" and len(value) < 8:
            return False
        if normalized_label == "IP Address" and digit_count < 4:
            return False
        return True

    def detect(self, text: str) -> List[Dict]:
        """Run GLiNER detection and return standardized findings."""
        if not text or self._disabled or self._model is None:
            return []

        try:
            if hasattr(self._model, "predict_entities"):
                raw = self._model.predict_entities(
                    text, self.LABEL_CANDIDATES, threshold=self.CONFIDENCE_THRESHOLD
                )
            elif hasattr(self._model, "predict"):
                raw = self._model.predict(text, self.LABEL_CANDIDATES)
            else:
                raw = self._model(text)
        except Exception as exc:
            logger.error("GLiNER detection failed: %s", exc)
            return []

        findings: List[Dict] = []
        seen = set()

        for ent in raw:
            # GLiNER outputs vary; accept both attribute and dict-like objects.
            if isinstance(ent, dict):
                start = int(ent.get("start", 0))
                end = int(ent.get("end", 0))
                label = str(ent.get("label", ent.get("entity", "UNKNOWN")))
                score = float(ent.get("score", 0.0))
                value = ent.get("text") or ent.get("entity_text") or text[start:end]
            else:
                start = int(getattr(ent, "start", 0))
                end = int(getattr(ent, "end", 0))
                label = str(getattr(ent, "label", getattr(ent, "entity", "UNKNOWN")))
                score = float(getattr(ent, "score", 0.0))
                value = getattr(ent, "text", getattr(ent, "entity_text", text[start:end]))

            value = str(value).strip()
            if not value or len(value) < self.MIN_VALUE_LENGTH:
                continue

            normalized_label = self.LABEL_MAP.get(label.lower(), label)
            if not self._is_valid_content(normalized_label, value):
                continue
            severity = self._severity_for(normalized_label)
            key = (normalized_label, value, start, end)
            if key in seen:
                continue
            seen.add(key)

            findings.append(
                {
                    "type": normalized_label,
                    "label": label,
                    "value": value,
                    "masked_value": value[:3] + "***" + (value[-3:] if len(value) > 3 else ""),
                    "severity": severity,
                    "start": start,
                    "end": end,
                    "line": text.count("\n", 0, start) + 1,
                    "context": text[max(0, start - 80) : min(len(text), end + 80)].replace("\n", " "),
                    "engine": "gliner",
                    "confidence": score,
                }
            )

        logger.info("GLiNER detection found %s entities", len(findings))
        return findings

    @classmethod
    def _severity_for(cls, label: str) -> str:
        if label in cls.HIGH_RISK_LABELS:
            return "high"
        if label in cls.MEDIUM_RISK_LABELS:
            return "medium"
        return "low"
