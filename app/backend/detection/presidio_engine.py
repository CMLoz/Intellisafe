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

    ENTITY_MAP = {
        "EMAIL_ADDRESS": "Email",
        "PHONE_NUMBER": "Phone Number",
        "CREDIT_CARD": "Credit Card",
        "US_SSN": "ID Number",
        "US_PASSPORT": "ID Number",
        "US_DRIVER_LICENSE": "ID Number",
        "IBAN_CODE": "Bank Account",
        "IP_ADDRESS": "IP Address",
        "API_TOKEN": "API Key",
        "PERSON": "Person Name",
        "LOCATION": "Location",
        "ORGANIZATION": "Organization",
        "NRP": "ID Number",
        "DATE_TIME": "Date of Birth",
        "MEDICAL_LICENSE": "ID Number",
        "CRYPTO": "API Key",
        "URL": "URL",
    }

    HIGH_RISK_ENTITIES = {
        "CREDIT_CARD",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "IBAN_CODE",
        "API_TOKEN",
        "MEDICAL_LICENSE",
        "CRYPTO",
    }

    MEDIUM_RISK_ENTITIES = {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "IP_ADDRESS",
        "DATE_TIME",
        "NRP",
    }

    # spaCy NER types that produce noisy detections on field labels.
    # Only keep these when Presidio's score meets the higher bar.
    SPACY_NER_TYPES = {"ORGANIZATION", "PERSON", "LOCATION", "URL", "NRP"}
    SPACY_NER_MIN_CONFIDENCE = 0.75
    # Suppress URL type entirely — email sub-parts are frequently mis-tagged as URLs.
    SUPPRESSED_TYPES = {"URL"}
    MIN_VALUE_LENGTH = 3

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
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Presidio Analyzer is not installed. Install with: pip install presidio-analyzer"
            ) from exc

        registry = RecognizerRegistry()
        for recognizer in PresidioEngine._build_recognizers(PatternRecognizer, Pattern):
            registry.add_recognizer(recognizer)

        # spaCy NER types that are not mapped to any Presidio entity type.
        # Listing them in labels_to_ignore prevents Presidio from emitting
        # "Entity X is not mapped to a Presidio entity" WARNING logs.
        # NOTE: NlpEngineProvider internally does NerModelConfiguration(**config["ner_model_configuration"]),
        # so this must be a plain dict — NOT a pre-built NerModelConfiguration object.
        _ner_model_configuration = {
            "labels_to_ignore": [
                "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
                "LANGUAGE", "WORK_OF_ART", "LAW", "EVENT", "PRODUCT", "TIME",
            ]
        }

        for model_name in ("en_core_web_sm", "en_core_web_lg"):
            try:
                nlp_engine = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": model_name}],
                        "ner_model_configuration": _ner_model_configuration,
                    }
                ).create_engine()
                return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
            except Exception as exc:
                logger.warning("Presidio spaCy model %s unavailable: %s", model_name, exc)

        try:
            return AnalyzerEngine(registry=registry)
        except Exception as exc:
            raise RuntimeError(
                "Presidio Analyzer could not start. Install a spaCy English model with: "
                "python -m spacy download en_core_web_sm"
            ) from exc

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

    def _should_keep(self, entity_type: str, score: float, value: str) -> bool:
        """Return False for findings that are likely false positives."""
        # Suppress entire entity types that are too noisy.
        if entity_type in self.SUPPRESSED_TYPES:
            return False
        # Drop very short values (field-label fragments).
        if len(value.strip()) < self.MIN_VALUE_LENGTH:
            return False
        # spaCy NER types require higher confidence to pass.
        if entity_type in self.SPACY_NER_TYPES and score < self.SPACY_NER_MIN_CONFIDENCE:
            return False
        return True

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

                if not self._should_keep(entity_type, score, value):
                    continue

                normalized_type = self.ENTITY_MAP.get(entity_type, entity_type)

                findings.append(
                    {
                        "type": normalized_type,
                        "label": entity_type,
                        "value": value,
                        "masked_value": value[:2] + "***" + (value[-2:] if len(value) > 2 else ""),
                        "severity": self._severity_for(entity_type),
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

    @classmethod
    def _severity_for(cls, entity_type: str) -> str:
        if entity_type in cls.HIGH_RISK_ENTITIES:
            return "high"
        if entity_type in cls.MEDIUM_RISK_ENTITIES:
            return "medium"
        return "low"
