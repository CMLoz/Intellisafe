"""Microsoft Presidio analyzer adapter.

This adapter wraps `presidio_analyzer.AnalyzerEngine` and maps results
to the project's finding structure. If Presidio is not installed the
adapter raises a helpful error explaining how to install it.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PresidioEngine:
    DEFAULT_LANGUAGE = "en"
    DEFAULT_LANGUAGES = ("en",)

    PATTERN_SPECS = [
        {
            "entity_type": "EMAIL_ADDRESS",
            "pattern_name": "email_address",
            "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "score": 0.85,
            "context": ["email", "mail", "correo", "courriel"],
        },
        {
            "entity_type": "PHONE_NUMBER",
            "pattern_name": "phone_number",
            "regex": r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}(?!\w)",
            "score": 0.8,
            "context": ["phone", "mobile", "tel", "telefono", "téléphone"],
        },
        {
            "entity_type": "CREDIT_CARD",
            "pattern_name": "credit_card",
            "regex": r"\b(?:\d[ -]*?){13,19}\b",
            "score": 0.95,
            "context": ["card", "credit", "tarjeta"],
        },
        {
            "entity_type": "US_SSN",
            "pattern_name": "us_ssn",
            "regex": r"\b\d{3}-\d{2}-\d{4}\b",
            "score": 0.98,
            "context": ["ssn", "social security"],
        },
        {
            "entity_type": "IBAN_CODE",
            "pattern_name": "iban",
            "regex": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
            "score": 0.92,
            "context": ["iban", "bank", "account", "cuenta", "compte"],
        },
        {
            "entity_type": "IP_ADDRESS",
            "pattern_name": "ip_address",
            "regex": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
            "score": 0.9,
            "context": ["ip", "address", "network", "adresse"],
        },
        {
            "entity_type": "API_TOKEN",
            "pattern_name": "api_token",
            "regex": r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|bearer)\b\s*[:=]\s*['\"]?([A-Za-z0-9._~+/=-]{16,})",
            "score": 0.97,
            "context": ["token", "api", "secret"],
            "group": 1,
        },
    ]

    def __init__(self, language: str | None = None, languages: List[str] | None = None):
        self.languages = self._normalize_languages(language, languages)
        self._engine = self._load_engine()

    @staticmethod
    def _normalize_languages(language: str | None, languages: List[str] | None) -> List[str]:
        if languages:
            cleaned = [item.strip() for item in languages if item and item.strip()]
            return cleaned or list(PresidioEngine.DEFAULT_LANGUAGES)

        if language:
            split_languages = [item.strip() for item in language.replace("+", ",").split(",") if item.strip()]
            return split_languages or list(PresidioEngine.DEFAULT_LANGUAGES)

        return list(PresidioEngine.DEFAULT_LANGUAGES)

    @staticmethod
    def _load_engine():
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Presidio Analyzer is not installed. Install with: pip install presidio-analyzer"
            ) from exc

        registry = RecognizerRegistry()
        for recognizer in PresidioEngine._build_recognizers(PatternRecognizer, Pattern):
            registry.add_recognizer(recognizer)

        return AnalyzerEngine(registry=registry)

    @staticmethod
    def _build_recognizers(pattern_recognizer_cls, pattern_cls):
        recognizers = []
        for spec in PresidioEngine.PATTERN_SPECS:
            pattern = pattern_cls(name=spec["pattern_name"], regex=spec["regex"], score=spec["score"])
            recognizers.append(
                pattern_recognizer_cls(
                    supported_entity=spec["entity_type"],
                    patterns=[pattern],
                    context=spec.get("context", []),
                )
            )
        return recognizers

    @staticmethod
    def _merge_results(results: List[Dict]) -> List[Dict]:
        merged = []
        seen = set()
        for res in results:
            key = (res.get("type"), res.get("value"), res.get("start"), res.get("end"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(res)
        merged.sort(key=lambda item: (item.get("start", 0), item.get("type", "")))
        return merged

    def detect(self, text: str) -> List[Dict]:
        if not text:
            return []

        findings: List[Dict] = []
        for language in self.languages:
            try:
                results = self._engine.analyze(text=text, language=language)
            except Exception as exc:
                logger.error("Presidio analysis failed for language %s: %s", language, exc)
                continue

            for res in results:
                start = int(res.start)
                end = int(res.end)
                entity_type = res.entity_type
                score = float(res.score or 0.0)
                value = text[start:end]

                findings.append(
                    {
                        "type": entity_type,
                        "label": entity_type,
                        "value": value,
                        "masked_value": value[:2] + "***" + (value[-2:] if len(value) > 2 else ""),
                        "severity": "medium",
                        "start": start,
                        "end": end,
                        "line": text.count("\n", 0, start) + 1,
                        "context": text[max(0, start - 80) : min(len(text), end + 80)].replace("\n", " "),
                        "engine": "presidio",
                        "confidence": score,
                        "language": language,
                    }
                )

        merged = self._merge_results(findings)
        logger.info("Presidio detection found %s entities", len(merged))
        return merged
