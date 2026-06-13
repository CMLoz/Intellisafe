"""Rebuild in-memory session state from the SQLite database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def detection_to_finding(row: Dict) -> Dict:
    entity_type = row.get("entity_type") or row.get("pattern_matched") or "Unknown"
    raw_value = row.get("entity_value") or row.get("data_found") or ""
    return {
        "type": entity_type,
        "value": raw_value,
        "masked_value": row.get("data_found") or raw_value,
        "severity": row.get("risk_level", "medium"),
        "risk_level": row.get("risk_level", "medium"),
        "engine": row.get("detection_type", "unknown"),
        "start": row.get("char_start"),
        "end": row.get("char_end"),
        "confidence": row.get("confidence"),
        "line": _line_from_location(row.get("location_info")),
    }


def _line_from_location(location: Optional[str]) -> int:
    if not location:
        return 1
    text = str(location).lower().replace("line", "").strip()
    try:
        return int(text)
    except ValueError:
        return 1


class SessionStore:
    def __init__(self, db_manager):
        self.db = db_manager

    def load_uploaded_files(self, limit: int = 200) -> List[Dict]:
        files = self.db.get_all_files(limit=limit)
        session: List[Dict] = []
        for row in files:
            file_info = self._file_row_to_info(row)
            if file_info:
                session.append(file_info)
        logger.info("Loaded %s file(s) from database", len(session))
        return session

    def _file_row_to_info(self, row: Dict) -> Optional[Dict]:
        path = row.get("file_path")
        if not path or not Path(path).exists():
            logger.debug("Skipping missing file on disk: %s", path)
            return None

        file_path = Path(path)
        detections = self.db.get_detections_for_file(row["id"])
        findings = [detection_to_finding(d) for d in detections]

        parsed_content = row.get("parsed_content_preview") or ""
        ocr_row = self.db.get_ocr_result(row["id"])
        word_boxes: List[Dict] = []
        ocr_result: Dict = {}

        if ocr_row:
            parsed_content = ocr_row.get("extracted_text") or parsed_content
            word_boxes = ocr_row.get("word_boxes") or []
            ocr_result = {
                "text": parsed_content,
                "confidence": ocr_row.get("confidence", 0),
                "word_count": ocr_row.get("word_count", 0),
                "word_boxes": word_boxes,
                "preprocessing_steps": ocr_row.get("preprocessing_steps", []),
                "language": ocr_row.get("ocr_language", "eng"),
                "validation_tier": ocr_row.get("validation_tier"),
                "confidence_breakdown": ocr_row.get("confidence_breakdown"),
            }
        elif file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            try:
                from app.backend.file_parser import FileParser
                parsed = FileParser.parse(file_path)
                if parsed.get("status") == "success":
                    parsed_content = parsed.get("content", parsed_content)
            except Exception as exc:
                logger.warning("Could not re-parse %s: %s", file_path.name, exc)

        risk_assessment = {
            "risk_score": row.get("risk_score", 0),
            "risk_level": row.get("risk_level", "low"),
            "total_findings": row.get("total_findings_count", len(findings)),
            "findings_by_type": _parse_entity_counts(row.get("entity_type_counts")),
        }
        if row.get("post_redaction_risk_score") is not None:
            risk_assessment["post_redaction"] = {
                "risk_score": row.get("post_redaction_risk_score"),
                "risk_level": row.get("post_redaction_risk_level"),
                "total_findings": row.get("post_redaction_findings_count", 0),
            }

        compliance = None
        if findings:
            from app.backend.compliance_engine import ComplianceEngine
            compliance = ComplianceEngine().assess(findings)

        info = {
            "name": row.get("filename") or file_path.name,
            "path": str(file_path),
            "size": row.get("file_size") or file_path.stat().st_size,
            "format": row.get("file_format") or file_path.suffix.lower(),
            "file_id": row["id"],
            "file_hash": row.get("file_hash"),
            "parsed_content": parsed_content,
            "parsed_metadata": {"source": "database"},
            "findings": findings,
            "regex_findings": findings,
            "findings_summary": {"total": len(findings)},
            "regex_summary": {"total": len(findings)},
            "risk_assessment": risk_assessment,
            "compliance_assessment": compliance,
            "redacted_path": row.get("redacted_version_path"),
            "selected_finding_indices": list(range(len(findings))),
        }
        if ocr_result:
            info["ocr_result"] = {**ocr_result, "file_path": str(file_path), "file_name": file_path.name}
        if row.get("status") == "error":
            info["parse_error"] = row.get("error_message") or "Previous scan failed"
        return info


def _parse_entity_counts(raw) -> Dict[str, int]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
