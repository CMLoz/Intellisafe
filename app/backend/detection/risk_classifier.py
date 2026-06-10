"""Risk classification system for detected sensitive data.

Provides a configurable RiskClassifier that evaluates findings and assigns
risk levels (low, medium, high) based on data type, confidence scores,
and other contextual factors.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RISK_LEVELS = ("low", "medium", "high")

HIGH_RISK_TYPES = {
    "Credit Card",
    "SSN",
    "Password",
    "API Key",
    "API_TOKEN",
    "US_SSN",
    "IBAN_CODE",
    "ID Number",
    "Bank Account",
    "Date of Birth",
    "passport number",
    "bank account",
    "date of birth",
}

MEDIUM_RISK_TYPES = {
    "Email",
    "Phone Number",
    "IP Address",
}

LOW_RISK_TYPES = {
    "Person Name",
    "Organization",
    "Location",
    "Address",
}


class RiskClassifier:
    """Classify detected entities by risk level."""

    DEFAULT_THRESHOLD_HIGH = 0.8
    DEFAULT_THRESHOLD_MEDIUM = 0.5

    def __init__(
        self,
        threshold_high: float = DEFAULT_THRESHOLD_HIGH,
        threshold_medium: float = DEFAULT_THRESHOLD_MEDIUM,
        custom_rules: Optional[Dict] = None,
    ):
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        self.custom_rules = custom_rules or {}

    def classify(self, findings: List[Dict]) -> List[Dict]:
        """Assign risk level to each finding.

        Args:
            findings: List of detection findings

        Returns:
            Findings with added/modified `risk_level` field
        """
        for finding in findings:
            finding["risk_level"] = self._calculate_risk_level(finding)
        logger.info("Classified %s findings", len(findings))
        return findings

    def _calculate_risk_level(self, finding: Dict) -> str:
        detection_type = str(finding.get("type", "")).lower()
        label = str(finding.get("label", "")).lower()
        severity = str(finding.get("severity", "medium")).lower()
        confidence = float(finding.get("confidence", 0.0))
        transformer_conf = float(finding.get("transformer_confidence", 0.0))
        validated = finding.get("transformer_validated", False)

        if self.custom_rules:
            for rule_type, rule_level in self.custom_rules.items():
                if rule_type.lower() in detection_type or rule_type.lower() in label:
                    return str(rule_level).lower()

        if validated and transformer_conf >= self.threshold_high:
            return "high"

        for high_type in HIGH_RISK_TYPES:
            if high_type.lower() in detection_type or high_type.lower() in label:
                return "high"

        if severity == "high":
            if confidence >= self.threshold_medium or transformer_conf >= self.threshold_medium:
                return "high"
            return "medium"

        for medium_type in MEDIUM_RISK_TYPES:
            if medium_type.lower() in detection_type or medium_type.lower() in label:
                if confidence >= self.threshold_medium:
                    return "medium"
                return "low"

        for low_type in LOW_RISK_TYPES:
            if low_type.lower() in detection_type or low_type.lower() in label:
                if confidence >= self.threshold_high:
                    return "medium"
                return "low"

        if confidence >= self.threshold_high:
            return "high"
        elif confidence >= self.threshold_medium:
            return "medium"
        return "low"

    def get_risk_distribution(self, findings: List[Dict]) -> Dict[str, int]:
        """Get count of findings by risk level.

        Args:
            findings: List of classified findings

        Returns:
            Dict with low/medium/high counts
        """
        distribution = {"low": 0, "medium": 0, "high": 0}
        for finding in findings:
            risk = str(finding.get("risk_level", "medium")).lower()
            if risk in distribution:
                distribution[risk] += 1
        return distribution

    def get_high_risk_findings(self, findings: List[Dict]) -> List[Dict]:
        """Filter findings to only high-risk ones.

        Args:
            findings: List of classified findings

        Returns:
            Only findings with risk_level == 'high'
        """
        return [f for f in findings if f.get("risk_level") == "high"]

    def get_medium_risk_findings(self, findings: List[Dict]) -> List[Dict]:
        """Filter findings to medium and high risk.

        Args:
            findings: List of classified findings

        Returns:
            Findings with risk_level 'medium' or 'high'
        """
        return [f for f in findings if f.get("risk_level") in ("medium", "high")]