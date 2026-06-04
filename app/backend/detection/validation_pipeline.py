"""Tiered validation pipeline for detection engines."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .regex_engine import RegexEngine
from .gliner_engine import GLiNEREngine
from .presidio_engine import PresidioEngine
from .transformer_engine import TransformerEngine
from .entity_aggregator import aggregate as aggregate_entities

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Run tiered detection with regex, GLiNER, Presidio, and transformer validation."""

    def __init__(
        self,
        gliner_model: str | None = None,
        presidio_language: str | None = None,
        transformer_model: str | None = None,
    ):
        self._gliner_model = gliner_model
        self._presidio_language = presidio_language
        self._transformer_model = transformer_model
        self._gliner_engine: GLiNEREngine | None = None
        self._presidio_engine: PresidioEngine | None = None
        self._transformer_engine: TransformerEngine | None = None

    def _get_gliner(self) -> GLiNEREngine:
        if self._gliner_engine is None:
            self._gliner_engine = GLiNEREngine(model_name=self._gliner_model)
        return self._gliner_engine

    def _get_presidio(self) -> PresidioEngine:
        if self._presidio_engine is None:
            self._presidio_engine = PresidioEngine(language=self._presidio_language)
        return self._presidio_engine

    def _get_transformer(self) -> TransformerEngine:
        if self._transformer_engine is None:
            self._transformer_engine = TransformerEngine(model_name=self._transformer_model)
        return self._transformer_engine

    @staticmethod
    def _merge_findings(*groups: List[Dict]) -> List[Dict]:
        merged: List[Dict] = []
        seen = set()
        for group in groups:
            for finding in group:
                key = (
                    finding.get("type"),
                    finding.get("value"),
                    finding.get("start"),
                    finding.get("end"),
                    finding.get("engine"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(finding)
        merged.sort(key=lambda finding: (finding.get("start", 0), finding.get("type", "")))
        return merged

    @staticmethod
    def _confidence_breakdown(findings: List[Dict]) -> Dict:
        summary = {
            "regex": {"count": 0},
            "gliner": {"count": 0, "avg_confidence": 0.0},
            "presidio": {"count": 0, "avg_confidence": 0.0},
        }
        gliner_scores = []
        presidio_scores = []
        transformer_scores = []
        transformer_validated = 0

        for finding in findings:
            engine = finding.get("engine")
            if engine == "regex":
                summary["regex"]["count"] += 1
            elif engine == "gliner":
                summary["gliner"]["count"] += 1
                confidence = finding.get("confidence")
                if isinstance(confidence, (int, float)):
                    gliner_scores.append(float(confidence))
            elif engine == "presidio":
                summary["presidio"]["count"] += 1
                confidence = finding.get("confidence")
                if isinstance(confidence, (int, float)):
                    presidio_scores.append(float(confidence))

            transformer_confidence = finding.get("transformer_confidence")
            if isinstance(transformer_confidence, (int, float)):
                transformer_scores.append(float(transformer_confidence))
            if finding.get("transformer_validated") is True:
                transformer_validated += 1

        if gliner_scores:
            summary["gliner"]["avg_confidence"] = sum(gliner_scores) / len(gliner_scores)
        if presidio_scores:
            summary["presidio"]["avg_confidence"] = sum(presidio_scores) / len(presidio_scores)

        if transformer_scores:
            summary["transformer"] = {
                "avg_confidence": sum(transformer_scores) / len(transformer_scores),
                "validated": transformer_validated,
            }
        else:
            summary["transformer"] = {"avg_confidence": None, "validated": 0}

        return summary

    def run(self, text: str, mode: str = "standard") -> Dict:
        """Run detection pipeline based on scan mode."""
        if mode not in {"quick", "standard", "deep"}:
            raise ValueError(f"Unsupported scan mode: {mode}")
        # 1) Regex (fast conservative pass)
        regex_findings = RegexEngine.detect(text)
        if mode == "quick":
            findings = regex_findings
        else:
            # 2) GLiNER (contextual NER)
            gliner_findings = self._get_gliner().detect(text)

            # 3) Presidio (recognizers & contextual scoring)
            presidio_findings = self._get_presidio().detect(text)

            # 4) Aggregate and compare outputs from all engines
            merged, aggregation_meta = aggregate_entities([
                regex_findings,
                gliner_findings,
                presidio_findings,
            ])

            findings = merged

            # In deep mode run transformer-based validation to attach
            # additional confidence where available and prefer validated
            # records.
            if mode == "deep":
                findings = self._get_transformer().validate(findings)
                findings = [
                    finding
                    for finding in findings
                    if finding.get("transformer_validated") is True or finding.get("engines_count", 1) > 0
                ]

        summary = RegexEngine.summarize(findings)
        confidence_breakdown = self._confidence_breakdown(findings)

        logger.info(
            "Validation pipeline completed: mode=%s findings=%s",
            mode,
            len(findings),
        )
        return {
            "findings": findings,
            "summary": summary,
            "validation_tier": mode,
            "confidence_breakdown": confidence_breakdown,
        }
