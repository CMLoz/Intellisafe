"""
IntelliSafe - Sensitive Data Tracker
File hashing and persistent tracking of detected entities.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Return a hex digest of file contents."""
    path = Path(file_path)
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_entity_hash(entity_type: str, value: str) -> str:
    """Return a stable hash for an entity type + raw value pair."""
    payload = f"{entity_type}:{value.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SensitiveDataTracker:
    """Persist file hashes and detection findings to the database."""

    def __init__(self, db_manager):
        self.db = db_manager

    def persist_file_scan(
        self,
        file_info: Dict,
        findings: List[Dict],
        risk_assessment: Dict,
    ) -> Optional[int]:
        """Store or update a scanned file and its sensitive-data findings."""
        file_path = file_info.get("path")
        if not file_path:
            return None

        try:
            file_hash = compute_file_hash(file_path)
        except OSError as exc:
            logger.warning("Could not hash %s: %s", file_path, exc)
            file_hash = None

        duplicate = None
        if file_hash:
            duplicate = self.db.get_file_by_hash(file_hash, exclude_path=file_path)
            if duplicate:
                file_info["duplicate_of"] = duplicate.get("filename")
                file_info["duplicate_file_id"] = duplicate.get("id")

        preview = (file_info.get("parsed_content") or "")[:500]
        status = "success" if "parse_error" not in file_info else "error"

        file_id = self.db.add_file(
            filename=file_info["name"],
            file_path=file_info["path"],
            file_size=file_info["size"],
            file_format=file_info["format"],
            file_hash=file_hash,
            status=status,
            error_message=file_info.get("parse_error"),
            parsed_preview=preview,
        )
        if not file_id:
            return None

        file_info["file_hash"] = file_hash
        file_info["file_id"] = file_id

        self.db.replace_detections_for_file(file_id, self._detection_rows(file_id, findings))
        self.db.update_file_risk_metadata(
            file_id=file_id,
            risk_score=risk_assessment.get("risk_score", 0),
            risk_level=risk_assessment.get("risk_level", "low"),
            findings_count=risk_assessment.get("total_findings", len(findings)),
            entity_counts=risk_assessment.get("findings_by_type", {}),
        )

        ocr_result = file_info.get("ocr_result")
        if ocr_result and file_info.get("parsed_metadata", {}).get("source") == "ocr":
            self.persist_ocr_result(file_id, ocr_result)

        from app.backend.compliance_engine import ComplianceEngine
        file_info["compliance_assessment"] = ComplianceEngine().assess(findings)

        logger.info(
            "Tracked %s findings for %s (hash=%s…)",
            len(findings),
            file_info["name"],
            (file_hash or "")[:12],
        )
        return file_id

    def persist_ocr_result(self, file_id: int, ocr_result: Dict) -> None:
        self.db.upsert_ocr_result(
            file_id=file_id,
            extracted_text=ocr_result.get("text", ""),
            confidence=ocr_result.get("confidence", 0),
            preprocessing_steps=ocr_result.get("preprocessing_steps"),
            word_count=ocr_result.get("word_count", 0),
            language=ocr_result.get("language", "eng"),
            validation_tier=ocr_result.get("validation_tier"),
            confidence_breakdown=ocr_result.get("confidence_breakdown"),
            word_boxes=ocr_result.get("word_boxes"),
        )

    def rescan_redacted_file(
        self,
        file_id: int,
        redacted_path: str,
        risk_manager,
        validation_pipeline,
    ) -> Dict:
        """Re-scan a redacted copy and store post-redaction risk metadata."""
        path = Path(redacted_path)
        if not path.exists():
            raise FileNotFoundError(redacted_path)

        suffix = path.suffix.lower()
        text = ""
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            from app.backend.ocr_processor import OCRPipeline
            result = OCRPipeline().process(str(path), preprocessing_steps=[])
            text = result.get("text", "")
        elif suffix == ".pdf":
            from app.backend.file_parser import PDFParser
            parsed = PDFParser.parse(path)
            text = parsed.get("content", "") if parsed.get("status") == "success" else ""
        elif suffix == ".docx":
            from app.backend.file_parser import DOCXParser
            parsed = DOCXParser.parse(path)
            text = parsed.get("content", "") if parsed.get("status") == "success" else ""
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        validation = validation_pipeline.run(text, mode="quick")
        findings = validation.get("findings", [])
        assessment = risk_manager.assess(findings)
        self.db.update_post_redaction_metadata(
            file_id=file_id,
            risk_score=assessment.get("risk_score", 0),
            risk_level=assessment.get("risk_level", "low"),
            findings_count=assessment.get("total_findings", len(findings)),
        )
        return {"findings": findings, "assessment": assessment, "text": text}

    @staticmethod
    def _detection_rows(file_id: int, findings: List[Dict]) -> List[Dict]:
        rows: List[Dict] = []
        for finding in findings:
            entity_type = finding.get("type", finding.get("label", "Unknown"))
            value = finding.get("value", "")
            masked = finding.get("masked_value") or _mask(value)
            engine = finding.get("engine")
            if not engine and isinstance(finding.get("sources"), list):
                engine = ",".join(finding["sources"])
            engine = engine or "unknown"
            rows.append(
                {
                    "file_id": file_id,
                    "detection_type": engine,
                    "pattern_matched": entity_type,
                    "data_found": masked,
                    "entity_value": value,
                    "location_info": f"line {finding.get('line', '?')}",
                    "risk_level": finding.get("severity", finding.get("risk_level", "medium")),
                    "entity_type": entity_type,
                    "entity_hash": compute_entity_hash(entity_type, value) if value else None,
                    "char_start": finding.get("start"),
                    "char_end": finding.get("end"),
                    "confidence": finding.get(
                        "consensus_score",
                        finding.get("confidence", finding.get("highest_confidence")),
                    ),
                }
            )
        return rows

    def log_redaction(
        self,
        file_id: int,
        output_path: str,
        strategy: str,
        mode: str,
        findings_redacted: int,
    ) -> None:
        self.db.add_redaction(
            file_id=file_id,
            redaction_type="document_redaction",
            mode=mode,
            strategy=strategy,
            output_path=output_path,
            findings_redacted=findings_redacted,
        )


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:3] + "***" + (value[-3:] if len(value) > 3 else "")
