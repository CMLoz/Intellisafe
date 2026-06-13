"""Rule-based compliance assessment from detected entities."""

from __future__ import annotations

from typing import Dict, List


GDPR_TRIGGERS = {
    "Email", "Phone Number", "Person Name", "Address", "Location",
    "ID Number", "Date of Birth", "Credit Card", "Bank Account",
}
HIPAA_TRIGGERS = {
    "Person Name", "Date of Birth", "ID Number", "Email", "Phone Number",
    "Medical Record", "Address",
}
DPA_TRIGGERS = {
    "Person Name", "Email", "Phone Number", "ID Number", "Address",
}


class ComplianceEngine:
    """Evaluate basic GDPR / HIPAA / DPA exposure from findings."""

    def assess(self, findings: List[Dict]) -> Dict:
        types = {f.get("type", f.get("label", "")) for f in findings}
        high = sum(1 for f in findings if str(f.get("severity", f.get("risk_level", ""))).lower() == "high")

        gdpr_hits = types & GDPR_TRIGGERS
        hipaa_hits = types & HIPAA_TRIGGERS
        dpa_hits = types & DPA_TRIGGERS

        return {
            "GDPR": self._verdict(gdpr_hits, high, "personal data processing"),
            "HIPAA": self._verdict(hipaa_hits, high, "protected health information indicators"),
            "DPA": self._verdict(dpa_hits, high, "personal data under DPA"),
            "frameworks": {
                "GDPR": sorted(gdpr_hits),
                "HIPAA": sorted(hipaa_hits),
                "DPA": sorted(dpa_hits),
            },
        }

    @staticmethod
    def _verdict(hits: set, high_count: int, label: str) -> Dict:
        if not hits:
            return {
                "status": "Pass",
                "detail": f"No {label} detected.",
                "action": "No compliance action required.",
            }
        if high_count > 0:
            return {
                "status": "Fail",
                "detail": f"High-risk {label}: {', '.join(sorted(hits))}.",
                "action": "Redact or remove high-risk fields before sharing.",
            }
        return {
            "status": "Review",
            "detail": f"Potential {label}: {', '.join(sorted(hits))}.",
            "action": "Review findings and apply redaction before external sharing.",
        }
