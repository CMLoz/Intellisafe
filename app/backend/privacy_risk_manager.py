"""
IntelliSafe - Privacy Risk Manager
Assesses document risk based on detected entities and provides recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

CATEGORY_WEIGHTS: Dict[str, float] = {
    "Credit Card": 1.5,
    "SSN": 1.5,
    "US_SSN": 1.5,
    "ID Number": 1.4,
    "Password": 1.4,
    "API Key": 1.4,
    "API_TOKEN": 1.4,
    "IBAN_CODE": 1.4,
    "Email": 1.0,
    "Phone Number": 1.0,
    "Person Name": 0.8,
    "Organization": 0.8,
    "Location": 0.8,
    "Address": 0.9,
    "bank account": 1.3,
    "passport number": 1.4,
    "date of birth": 1.1,
}

HIGH_RISK_TYPES = {
    "Credit Card", "SSN", "Password", "API Key", "API_TOKEN",
    "US_SSN", "IBAN_CODE", "passport number", "bank account",
}
MEDIUM_RISK_TYPES = {
    "Email", "Phone Number", "ID Number", "date of birth",
}
LOW_RISK_TYPES = {
    "Person Name", "Organization", "Location", "Address",
}

SEVERITY_SCORE = {"low": 1, "medium": 2, "high": 3}


class PrivacyRiskManager:
    """Analyse a list of findings and classify overall document risk."""

    HIGH_THRESHOLD = 60
    MEDIUM_THRESHOLD = 25

    FULL_REDACT_TYPES = {
        "Credit Card", "SSN", "US_SSN", "Password", "API Key", "API_TOKEN",
        "IBAN_CODE", "passport number", "bank account", "ID Number",
    }

    def __init__(
        self,
        high_threshold: float = HIGH_THRESHOLD,
        medium_threshold: float = MEDIUM_THRESHOLD,
        category_weights: Dict[str, float] | None = None,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.weights = {**CATEGORY_WEIGHTS, **(category_weights or {})}

    def assess(self, findings: List[Dict]) -> Dict:
        if not findings:
            return self._empty_assessment()
        by_type = self._count_by_type(findings)
        by_severity = self._count_by_severity(findings)
        composite_score = self._composite_score(findings, by_type, by_severity)
        risk_level = self._classify_level(composite_score)
        recommendations = self._recommendations(risk_level, findings, by_type)
        redaction_strategy = self._redaction_strategy(findings)

        return {
            "risk_score": round(composite_score, 2),
            "risk_level": risk_level,
            "total_findings": len(findings),
            "high_risk_count": by_severity.get("high", 0),
            "medium_risk_count": by_severity.get("medium", 0),
            "low_risk_count": by_severity.get("low", 0),
            "findings_by_type": by_type,
            "findings_by_severity": by_severity,
            "category_at_risk": self._top_categories(by_type, limit=5),
            "recommendation": recommendations["message"],
            "recommendation_detail": recommendations["detail"],
            "redaction_strategy": redaction_strategy,
            "safe_to_share": risk_level == "low",
        }

    def get_entity_redaction_mode(self, finding: Dict) -> str:
        ftype = finding.get("type", finding.get("label", ""))
        if ftype in self.FULL_REDACT_TYPES:
            return "full"
        return "partial"

    def _empty_assessment(self) -> Dict:
        return {
            "risk_score": 0.0,
            "risk_level": "low",
            "total_findings": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "findings_by_type": {},
            "findings_by_severity": {},
            "category_at_risk": [],
            "recommendation": "No sensitive information detected. Document appears safe to share.",
            "recommendation_detail": "The document contains no entities flagged by the detection pipeline.",
            "redaction_strategy": {"default": "partial"},
            "safe_to_share": True,
        }

    @staticmethod
    def _count_by_type(findings: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in findings:
            t = f.get("type", f.get("label", "Unknown"))
            counts[t] = counts.get(t, 0) + 1
        return counts

    @staticmethod
    def _count_by_severity(findings: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for f in findings:
            sev = str(f.get("severity", f.get("risk_level", "medium"))).lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    def _composite_score(self, findings, by_type, by_severity) -> float:
        if not findings:
            return 0.0
        severity_component = min(
            (by_severity.get("high", 0) * 10)
            + (by_severity.get("medium", 0) * 5)
            + (by_severity.get("low", 0) * 2),
            100,
        )
        weighted_type_component = 0.0
        for ftype, count in by_type.items():
            weight = self.weights.get(ftype, 1.0)
            sample = findings[0]
            sev_key = str(sample.get("severity", sample.get("risk_level", "medium"))).lower()
            sev_factor = SEVERITY_SCORE.get(sev_key, 2)
            weighted_type_component += count * weight * sev_factor
        weighted_type_component = min(weighted_type_component * 2, 100 - severity_component)
        return min(severity_component + weighted_type_component, 100)

    def _classify_level(self, score: float) -> str:
        if score >= self.high_threshold:
            return "high"
        if score >= self.medium_threshold:
            return "medium"
        return "low"

    def _recommendations(self, level, findings, by_type) -> Dict[str, str]:
        high_types = {t for t, c in by_type.items() if t in HIGH_RISK_TYPES and c > 0}
        if level == "high":
            msg = "Do NOT share this document without redaction."
            if high_types:
                detail = f"Document contains {len(findings)} sensitive findings including: " + ", ".join(sorted(high_types))
            else:
                detail = "high-risk entities (SSN, credit card, passwords)."
        elif level == "medium":
            msg = "Review recommended before sharing."
            detail = (
                f"{len(findings)} potentially sensitive items detected. "
                "Apply partial masking or redaction for names, emails, and phone numbers."
            )
        else:
            msg = "Document appears safe to share."
            detail = "No high-severity sensitive entities were detected."
        return {"message": msg, "detail": detail}

    def _redaction_strategy(self, findings: List[Dict]) -> Dict[str, str]:
        strategy: Dict[str, str] = {}
        for f in findings:
            ftype = f.get("type", f.get("label", ""))
            strategy[ftype] = self.get_entity_redaction_mode(f)
        return strategy

    @staticmethod
    def _top_categories(by_type: Dict[str, int], limit: int = 5) -> List[str]:
        ranked = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)
        return [cat for cat, _ in ranked[:limit]]
