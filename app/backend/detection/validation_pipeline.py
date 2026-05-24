"""Tiered validation pipeline for detection engines."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .regex_engine import RegexEngine
from .spacy_engine import SpacyEngine
from .transformer_engine import TransformerEngine

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Run tiered detection with regex, spaCy, and transformer validation."""

    def __init__(self, spacy_model: str | None = None, transformer_model: str | None = None):
        self._spacy_model = spacy_model
        self._transformer_model = transformer_model
        self._spacy_engine: SpacyEngine | None = None
        self._transformer_engine: TransformerEngine | None = None

    def _get_spacy(self) -> SpacyEngine:
        if self._spacy_engine is None:
            self._spacy_engine = SpacyEngine(model_name=self._spacy_model)
        return self._spacy_engine

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
        summary = {"regex": {"count": 0}, "spacy": {"count": 0, "avg_confidence": 0.0}}
        spacy_scores = []
        transformer_scores = []
        transformer_validated = 0

        for finding in findings:
            engine = finding.get("engine")
            if engine == "regex":
                summary["regex"]["count"] += 1
            elif engine == "spacy":
                summary["spacy"]["count"] += 1
                confidence = finding.get("confidence")
                if isinstance(confidence, (int, float)):
                    spacy_scores.append(float(confidence))

            transformer_confidence = finding.get("transformer_confidence")
            if isinstance(transformer_confidence, (int, float)):
                transformer_scores.append(float(transformer_confidence))
            if finding.get("transformer_validated") is True:
                transformer_validated += 1

        if spacy_scores:
            summary["spacy"]["avg_confidence"] = sum(spacy_scores) / len(spacy_scores)

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

        regex_findings = RegexEngine.detect(text)
        if mode == "quick":
            findings = regex_findings
        else:
            spacy_findings = self._get_spacy().detect(text)
            findings = self._merge_findings(regex_findings, spacy_findings)

            if mode == "deep":
                findings = self._get_transformer().validate(findings)
                findings = [
                    finding
                    for finding in findings
                    if finding.get("engine") != "spacy"
                    or finding.get("transformer_validated") is True
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
