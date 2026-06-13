"""Redaction orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app.backend.redaction_engine import FULL_REDACT_TYPES, RedactionEngine


def filter_findings_by_mode(findings: List[Dict], mode: str) -> List[Dict]:
    if mode == "full":
        return [
            f for f in findings
            if RedactionEngine._mode_for(f) == "full"
            or f.get("type") in FULL_REDACT_TYPES
            or str(f.get("severity", f.get("risk_level", ""))).lower() == "high"
        ]
    if mode == "partial":
        return [
            f for f in findings
            if RedactionEngine._mode_for(f) == "partial"
            and f.get("type") not in FULL_REDACT_TYPES
            and str(f.get("severity", f.get("risk_level", ""))).lower() != "high"
        ]
    return list(findings)


def selected_findings(file_info: Dict) -> List[Dict]:
    findings = file_info.get("findings", [])
    indices = file_info.get("selected_finding_indices")
    if not indices:
        return findings
    return [findings[i] for i in indices if 0 <= i < len(findings)]


def run_redaction_for_file(
    engine: RedactionEngine,
    file_info: Dict,
    findings: List[Dict],
    strategy: str,
    mode: str,
) -> Dict:
    findings = filter_findings_by_mode(findings, mode)
    if not findings:
        return {"output_path": None, "findings_redacted": 0, "skipped": True}

    path = file_info.get("path", "")
    fmt = file_info.get("format", Path(path).suffix.lower())

    if fmt in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        word_boxes = file_info.get("ocr_result", {}).get("word_boxes") or []
        preprocessing_steps = file_info.get("ocr_result", {}).get("preprocessing_steps") or []
        return engine.redact_image(
            path, findings, strategy,
            word_boxes=word_boxes or None,
            preprocessing_steps=preprocessing_steps,
        )
    if fmt == ".pdf":
        return engine.redact_pdf(path, findings, strategy)
    if fmt == ".docx":
        return engine.redact_docx(path, findings)
    if fmt in {".txt", ".sql"}:
        return engine.redact_text_file(path, findings)
    raise ValueError(f"Redaction not supported for format: {fmt}")
