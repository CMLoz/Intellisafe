"""Aggregate and compare findings from multiple engines.

Provides utilities to merge overlapping detections, compute source
agreement, and attach a simple consensus score so callers can compare
and contrast regex, GLiNER and Presidio outputs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def aggregate(findings_groups: List[List[Dict]]) -> Tuple[List[Dict], Dict]:
    """Aggregate multiple lists of findings.

    Args:
        findings_groups: ordered lists of findings (e.g., [regex, gliner, presidio])

    Returns:
        merged: list of merged findings with `sources` and `consensus_score`
        meta: summary statistics about agreement between engines
    """
    merged: List[Dict] = []
    meta = {"total_inputs": len(findings_groups), "by_engine": {}, "total_merged": 0}

    # Flatten with source tags
    tagged = []
    for idx, group in enumerate(findings_groups):
        engine_name = group[0]["engine"] if group else f"engine_{idx}"
        meta["by_engine"][engine_name] = len(group)
        for f in group:
            tagged.append({**f, "source_engine": f.get("engine", engine_name)})

    used = [False] * len(tagged)

    for i, base in enumerate(tagged):
        if used[i]:
            continue
        cluster = [base]
        used[i] = True
        for j in range(i + 1, len(tagged)):
            if used[j]:
                continue
            other = tagged[j]
            if _overlap(base["start"], base["end"], other["start"], other["end"]) or base.get("value") == other.get("value"):
                cluster.append(other)
                used[j] = True

        # Build merged record
        sources = list({c.get("source_engine", c.get("engine")) for c in cluster})
        confidences = [float(c.get("confidence", 0.0) or 0.0) for c in cluster]
        highest = max(confidences) if confidences else 0.0
        types = list({c.get("type") for c in cluster if c.get("type")})
        severities = [str(c.get("severity", "medium")).lower() for c in cluster]
        merged_severity = max(severities, key=lambda item: SEVERITY_RANK.get(item, 0)) if severities else "medium"

        merged_record = {
            "type": types[0] if types else cluster[0].get("type"),
            "value": cluster[0].get("value"),
            "start": min(c["start"] for c in cluster),
            "end": max(c["end"] for c in cluster),
            "sources": sources,
            "consensus_score": round((sum(confidences) / len(confidences)) if confidences else 0.0, 3),
            "highest_confidence": highest,
            "engines_count": len(sources),
            "severity": merged_severity,
            "representative_context": cluster[0].get("context"),
        }
        merged.append(merged_record)

    merged.sort(key=lambda r: (r["start"], r.get("type", "")))
    meta["total_merged"] = len(merged)
    logger.info("Entity aggregation completed: merged=%s", len(merged))
    return merged, meta
