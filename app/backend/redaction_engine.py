"""
IntelliSafe - Redaction Engine
Generates redacted copies of images and PDF documents.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
FULL_REDACT_TYPES = {
    "Credit Card", "SSN", "US_SSN", "Password", "API Key", "API_TOKEN",
    "IBAN_CODE", "passport number", "bank account", "ID Number",
}


class RedactionEngine:
    """Create redacted versions of documents."""

    def __init__(self, output_dir: str = "redacted_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact_image(
        self,
        image_path: str,
        findings: List[Dict],
        strategy: str = "blackout",
    ) -> Dict:
        """Return a redacted image (as bytes) and metadata for an image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        h, w = image.shape[:2]

        for finding in findings:
            coords = self._resolve_box(finding, w, h)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            mode = self._mode_for(finding)

            if mode == "full" or strategy == "blackout":
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
            elif strategy == "pixelate":
                self._pixelate_region(pil_img, draw, x1, y1, x2, y2)
            elif strategy == "blur":
                self._blur_region(pil_img, x1, y1, x2, y2)
            else:
                # mask (default partial for non-critical types)
                mask_text = "****"
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
                draw.text((x1 + 2, y1 + 2), mask_text, fill=(255, 255, 255), font=font)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        out_path = self.output_dir / f"redacted_{Path(image_path).name}"
        with open(out_path, "wb") as f:
            f.write(buf.read())

        return {
            "output_path": str(out_path),
            "format": "image/png",
            "original_format": Path(image_path).suffix.lower(),
            "strategy": strategy,
            "findings_redacted": len(findings),
        }

    def redact_pdf(
        self,
        pdf_path: str,
        findings: List[Dict],
        strategy: str = "blackout",
    ) -> Dict:
        """Create a redacted copy of a PDF file."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF is required for PDF redaction. Install with: pip install PyMuPDF")

        doc = fitz.open(pdf_path)
        output_path = self.output_dir / f"redacted_{Path(pdf_path).name}"
        redacted_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            for finding in findings:
                mode = self._mode_for(finding)
                if mode == "full" or strategy == "blackout":
                    coords = self._resolve_pdf_coords(finding, page_width, page_height)
                    if coords is None:
                        continue
                    rect = fitz.Rect(*coords)
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    redacted_count += 1
                else:
                    coords = self._resolve_pdf_coords(finding, page_width, page_height)
                    if coords is None:
                        continue
                    rect = fitz.Rect(*coords)
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    redacted_count += 1

            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        doc.save(str(output_path), deflate=True)
        doc.close()

        return {
            "output_path": str(output_path),
            "format": "application/pdf",
            "strategy": strategy,
            "findings_redacted": redacted_count,
            "pages_affected": len(doc) if isinstance(doc, fitz.Document) else 0,
        }

    def redact_text_file(
        self,
        text_path: str,
        findings: List[Dict],
    ) -> Dict:
        """Create a redacted copy of a text-based file by replacing sensitive spans."""
        content = Path(text_path).read_text(encoding="utf-8", errors="replace")
        original = content

        # Sort by descending start so replacements don't shift earlier offsets
        sorted_findings = sorted(
            [f for f in findings if "start" in f and "end" in f],
            key=lambda f: f["start"],
            reverse=True,
        )

        for finding in sorted_findings:
            start = finding.get("start", 0)
            end = finding.get("end", len(content))
            mode = self._mode_for(finding)
            if mode == "full" or finding.get("type") in FULL_REDACT_TYPES:
                replacement = "[REDACTED]"
            else:
                replacement = self._partial_mask(finding.get("value", "*****"))
            content = content[:start] + replacement + content[end:]

        out_path = self.output_dir / f"redacted_{Path(text_path).name}"
        out_path.write_text(content, encoding="utf-8")
        return {
            "output_path": str(out_path),
            "format": "text/plain",
            "strategy": "replace",
            "findings_redacted": len(sorted_findings),
            "chars_redacted": len(original) - len(content),
        }

    def should_redact_file(self, findings: List[Dict]) -> bool:
        return len(findings) > 0

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mode_for(finding: Dict) -> str:
        ftype = finding.get("type", finding.get("label", ""))
        if ftype in FULL_REDACT_TYPES:
            return "full"
        return "partial"

    @staticmethod
    def _partial_mask(value: str, visible_prefix: int = 2, visible_suffix: int = 2) -> str:
        if not value:
            return "****"
        if len(value) <= visible_prefix + visible_suffix + 1:
            return "*" * len(value)
        return value[:visible_prefix] + "*" * (len(value) - visible_prefix - visible_suffix) + value[-visible_suffix:]

    # ------------------------------------------------------------------
    # Image-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_box(finding: Dict, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
        """Return a valid (x1, y1, x2, y2) box guaranteed to satisfy x1<x2, y1<y2."""
        raw = finding.get("box") or finding.get("bbox")
        if raw is None and "start" in finding and "end" in finding:
            # start/end are absolute character offsets into the full document.
            # Estimate position within the current text line so the box stays
            # inside the image even when offsets are very large.
            line_num = max(1, finding.get("line", 1))
            span_len = max(1, finding["end"] - finding["start"])
            # Assume ~80 characters fit across the image width
            chars_per_line = max(1, img_w // max(8, 1))
            col = finding["start"] % chars_per_line
            char_w = img_w / chars_per_line
            x1 = int(col * char_w)
            x2 = min(img_w, int((col + span_len) * char_w))
            y1 = (line_num - 1) * 22
            y2 = y1 + 22
            # Clamp to image bounds and guarantee minimum box size.
            x1 = max(0, min(x1, img_w - 4))
            x2 = max(x1 + 4, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 4))
            y2 = max(y1 + 4, min(y2, img_h))
            return x1, y1, x2, y2
        if raw and len(raw) == 4:
            x1, y1, x2, y2 = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            return x1, y1, x2, y2
        return None

    @staticmethod
    def _resolve_pdf_coords(finding: Dict, page_w: float, page_h: float):
        """Return [x1, y1, x2, y2] in PDF points, guaranteed x1<x2 and y1<y2."""
        raw = finding.get("box") or finding.get("bbox")
        if raw and len(raw) == 4:
            x1, y1, x2, y2 = (float(v) for v in raw)
            x1 = max(0.0, min(x1, page_w - 1))
            y1 = max(0.0, min(y1, page_h - 1))
            x2 = max(x1 + 1.0, min(x2, page_w))
            y2 = max(y1 + 1.0, min(y2, page_h))
            return [x1, y1, x2, y2]
        if "start" in finding and "end" in finding:
            # Use line-relative positioning (same logic as _resolve_box).
            line_num = max(1, finding.get("line", 1))
            span_len = max(1, finding["end"] - finding["start"])
            # Assume ~80 characters per line across the page width.
            chars_per_line = 80
            col = finding["start"] % chars_per_line
            char_w = page_w / chars_per_line
            x1 = float(col * char_w)
            x2 = min(page_w, float((col + span_len) * char_w))
            y1 = float((line_num - 1) * 14)  # ~14pt line height for PDF
            y2 = y1 + 14.0
            # Clamp and guarantee non-degenerate rect.
            x1 = max(0.0, min(x1, page_w - 1.0))
            x2 = max(x1 + 1.0, min(x2, page_w))
            y1 = max(0.0, min(y1, page_h - 1.0))
            y2 = max(y1 + 1.0, min(y2, page_h))
            return [x1, y1, x2, y2]
        return None

    @staticmethod
    def _pixelate_region(pil_img: Image.Image, draw, x1: int, y1: int, x2: int, y2: int, size: int = 8):
        region = pil_img.crop((x1, y1, x2, y2))
        small = region.resize((max(1, (x2 - x1) // size), max(1, (y2 - y1) // size)), Image.Resampling.NEAREST)
        pixelated = small.resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
        pil_img.paste(pixelated, (x1, y1))

    @staticmethod
    def _blur_region(pil_img: Image.Image, x1: int, y1: int, x2: int, y2: int, radius: int = 10):
        region = pil_img.crop((x1, y1, x2, y2))
        blurred = region.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
        pil_img.paste(blurred, (x1, y1))