"""
Patch: replace the broken _coords_for_finding in redaction_engine.py.

The old logic collected EVERY word box in the document that contained any
part of the finding value as a substring, then merged them all into one huge
bounding rect.  That produced the giant black bar covering the whole page.

The new logic uses a sliding window over consecutive OCR word boxes:
PaddleOCR returns boxes in reading order (top-left to bottom-right), so a
multi-word entity like "Jan Pierre Balein" will appear as a contiguous run
of adjacent boxes.  We find that run and return only its tight bounding rect.
"""

import re

path = r'c:\Projects\Python\IntelliSafe\app\backend\redaction_engine.py'
with open(path, 'rb') as f:
    text = f.read().decode('utf-8')

# ------------------------------------------------------------------
# The exact block to replace (normalise to LF for matching, keep original EOL)
# ------------------------------------------------------------------
USE_CRLF = '\r\n' in text

def make(src: str) -> str:
    """Return src with the file's native EOL."""
    if USE_CRLF:
        return src.replace('\n', '\r\n')
    return src

OLD = make(
"""    @staticmethod
    def _coords_for_finding(
        finding: Dict,
        word_boxes: Optional[List[Dict]],
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        \"\"\"Return (x1, y1, x2, y2) for a finding.

        Priority:
        1. Look up the finding's value in the OCR word_boxes list and return
           the exact pixel bounding rect reported by the OCR engine.
        2. Fall back to _resolve_box which estimates position from character
           offsets or reads a pre-computed box/bbox field.
        \"\"\"
        value = (finding.get("value") or "").strip()
        if word_boxes and value:
            value_lower = value.lower()
            # Collect all boxes whose text contains the target value (case-insensitive).
            matched: List[Tuple[int, int, int, int]] = []
            for wb in word_boxes:
                wb_text = (wb.get("text") or "").strip().lower()
                if value_lower in wb_text or wb_text in value_lower:
                    bbox = wb.get("bbox")
                    if bbox and len(bbox) == 4:
                        matched.append(
                            (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        )
            if matched:
                # Merge all matched boxes into a single bounding rect.
                x1 = max(0, min(b[0] for b in matched))
                y1 = max(0, min(b[1] for b in matched))
                x2 = min(img_w, max(b[2] for b in matched))
                y2 = min(img_h, max(b[3] for b in matched))
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2

        # Fallback: estimate from character offsets or explicit box field.
        return RedactionEngine._resolve_box(finding, img_w, img_h)
""")

NEW = make(
"""    @staticmethod
    def _coords_for_finding(
        finding: Dict,
        word_boxes: Optional[List[Dict]],
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        \"\"\"Return (x1, y1, x2, y2) for a finding.

        Strategy – sliding-window over consecutive OCR word boxes:
        PaddleOCR returns word boxes in reading order (top-left to
        bottom-right), so a multi-word entity like 'Jan Pierre Balein'
        appears as a contiguous run of adjacent boxes.  We slide a window
        of sizes [n-1, n, n+1] (n = number of tokens in the finding value)
        and accept the first window whose concatenated text contains the
        finding value (or vice-versa).  This gives a tight bounding rect
        covering exactly the matched words, nothing more.

        Falls back to _resolve_box (character-offset estimation) when no
        window matches.
        \"\"\"
        value = (finding.get("value") or "").strip()
        if not word_boxes or not value:
            return RedactionEngine._resolve_box(finding, img_w, img_h)

        val_lower = value.lower()
        val_tokens = val_lower.split()
        n_val = len(val_tokens)
        if n_val == 0:
            return RedactionEngine._resolve_box(finding, img_w, img_h)

        wb_texts = [(wb.get("text") or "").lower().strip() for wb in word_boxes]

        # Try window sizes close to the number of tokens in the value.
        for window_size in sorted(
            {max(1, n_val - 1), n_val, n_val + 1},
            key=lambda s: abs(s - n_val),   # prefer exact size first
        ):
            for i in range(len(word_boxes) - window_size + 1):
                joined = " ".join(wb_texts[i : i + window_size])
                # Accept if value appears in the joined window or vice-versa.
                if val_lower in joined or joined in val_lower:
                    bboxes = [
                        wb["bbox"]
                        for wb in word_boxes[i : i + window_size]
                        if wb.get("bbox") and len(wb["bbox"]) == 4
                    ]
                    if not bboxes:
                        continue
                    x1 = max(0, min(int(b[0]) for b in bboxes))
                    y1 = max(0, min(int(b[1]) for b in bboxes))
                    x2 = min(img_w, max(int(b[2]) for b in bboxes))
                    y2 = min(img_h, max(int(b[3]) for b in bboxes))
                    if x2 > x1 and y2 > y1:
                        return x1, y1, x2, y2

        # Fallback: estimate from character offsets or explicit box field.
        return RedactionEngine._resolve_box(finding, img_w, img_h)
""")

if OLD not in text:
    print("ERROR: OLD pattern not found in file.  Check for whitespace/EOL differences.")
    # Debug: show the actual block in the file
    idx = text.find("_coords_for_finding")
    print("=== File section around _coords_for_finding ===")
    print(repr(text[max(0,idx-10):idx+2000]))
    raise SystemExit(1)

text = text.replace(OLD, NEW, 1)

with open(path, 'wb') as f:
    f.write(text.encode('utf-8'))

print("Patch applied successfully.")
print(f"  File EOL: {'CRLF' if USE_CRLF else 'LF'}")
