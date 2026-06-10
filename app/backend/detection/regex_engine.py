"""
Regex-based sensitive data detection.

This engine is intentionally conservative enough for a first review pass:
it flags likely findings with category, severity, position, line number, and
nearby context so users can inspect possible false positives.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class RegexEngine:
    """Detect common sensitive data patterns with regular expressions."""

    PATTERNS = [
        {
            "type": "Email",
            "severity": "medium",
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "confidence": 0.92,
        },
        {
            "type": "Phone Number",
            "severity": "medium",
            "pattern": r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}(?!\w)",
            "confidence": 0.78,
        },
        {
            "type": "Credit Card",
            "severity": "high",
            "pattern": r"\b(?:\d[ -]*?){13,19}\b",
            "validator": "luhn",
            "confidence": 0.97,
        },
        {
            "type": "Password",
            "severity": "high",
            "pattern": r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*['\"]?([^\s'\";,\]]{6,})",
            "group": 1,
            "confidence": 0.95,
        },
        {
            "type": "API Key",
            "severity": "high",
            "pattern": r"\bAKIA[0-9A-Z]{16}\b",
            "confidence": 0.98,
        },
        {
            "type": "API Key",
            "severity": "high",
            "pattern": r"\bAIza[0-9A-Za-z_-]{35}\b",
            "confidence": 0.97,
        },
        {
            "type": "API Key",
            "severity": "high",
            "pattern": r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b",
            "confidence": 0.96,
        },
        {
            "type": "API Key",
            "severity": "high",
            "pattern": r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|bearer)\b\s*[:=]\s*['\"]?([A-Za-z0-9._~+/=-]{16,})",
            "group": 1,
            "confidence": 0.92,
        },
        {
            "type": "ID Number",
            "severity": "high",
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
            "label": "SSN-like ID",
            "confidence": 0.96,
        },
        {
            "type": "ID Number",
            "severity": "medium",
            "pattern": r"(?i)\b(?:employee|student|customer|account|user|member|passport|license|id)\s*(?:id|no|number|#)?\s*[:=]\s*['\"]?([A-Z0-9][A-Z0-9-]{4,24})",
            "group": 1,
            "confidence": 0.72,
        },
    ]

    @staticmethod
    def detect(text: str) -> List[Dict]:
        """Return regex findings for extracted text."""
        if not text:
            return []

        findings = []
        seen = set()

        for spec in RegexEngine.PATTERNS:
            pattern = re.compile(spec["pattern"])
            for match in pattern.finditer(text):
                group = spec.get("group", 0)
                value = match.group(group)
                if not value:
                    continue

                if spec.get("validator") == "luhn" and not RegexEngine._passes_luhn(value):
                    continue

                start, end = match.span(group)
                key = (spec["type"], value, start, end)
                if key in seen:
                    continue
                seen.add(key)

                findings.append(
                    {
                        "type": spec["type"],
                        "label": spec.get("label", spec["type"]),
                        "value": value,
                        "masked_value": RegexEngine.mask_value(value),
                        "severity": spec["severity"],
                        "start": start,
                        "end": end,
                        "line": RegexEngine._line_number(text, start),
                        "context": RegexEngine._context(text, start, end),
                        "engine": "regex",
                        "confidence": spec.get("confidence", 0.7),
                    }
                )

        findings = RegexEngine._remove_overlapping_findings(findings)
        findings.sort(key=lambda finding: (finding["start"], finding["type"]))
        logger.info("Regex detection found %s possible sensitive items", len(findings))
        return findings

    @staticmethod
    def mask_value(value: str) -> str:
        """Mask sensitive values while leaving enough shape for review."""
        if len(value) <= 4:
            return "*" * len(value)
        if "@" in value:
            name, domain = value.split("@", 1)
            visible_name = name[:2] if len(name) > 2 else name[:1]
            return f"{visible_name}***@{domain}"
        return f"{value[:3]}***{value[-4:]}"

    @staticmethod
    def summarize(findings: List[Dict]) -> Dict:
        """Summarize findings by type and severity."""
        summary = {
            "total": len(findings),
            "by_type": {},
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for finding in findings:
            summary["by_type"][finding["type"]] = summary["by_type"].get(finding["type"], 0) + 1
            severity = finding.get("severity", "low")
            summary[severity] = summary.get(severity, 0) + 1

        return summary

    @staticmethod
    def _passes_luhn(value: str) -> bool:
        digits = [int(char) for char in re.sub(r"\D", "", value)]
        if len(digits) < 13 or len(digits) > 19:
            return False

        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0

    @staticmethod
    def _remove_overlapping_findings(findings: List[Dict]) -> List[Dict]:
        severity_rank = {"high": 3, "medium": 2, "low": 1}
        ranked = sorted(
            findings,
            key=lambda finding: (
                -severity_rank.get(finding.get("severity", "low"), 1),
                -(finding["end"] - finding["start"]),
                finding["start"],
            ),
        )

        selected = []
        for finding in ranked:
            overlaps = any(
                finding["start"] < existing["end"] and existing["start"] < finding["end"]
                for existing in selected
            )
            if not overlaps:
                selected.append(finding)

        return selected

    @staticmethod
    def _line_number(text: str, position: int) -> int:
        return text.count("\n", 0, position) + 1

    @staticmethod
    def _context(text: str, start: int, end: int, radius: int = 80) -> str:
        context_start = max(0, start - radius)
        context_end = min(len(text), end + radius)
        context = text[context_start:context_end].replace("\n", " ")
        return re.sub(r"\s+", " ", context).strip()
