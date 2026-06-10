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
ENGINE_BASE_CONFIDENCE = {
    "regex": 0.74,
    "presidio": 0.82,
    "gliner": 0.68,
    "transformer": 0.75,
}


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _confidence(finding: Dict) -> float:
    confidence = finding.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 0:
        return max(0.0, min(float(confidence), 1.0))
    engine = str(finding.get("engine", finding.get("source_engine", ""))).lower()
    return ENGINE_BASE_CONFIDENCE.get(engine, 0.6)


def _consensus_score(confidences: List[float], sources_count: int) -> float:
    if not confidences:
        return 0.0
    highest = max(confidences)
    average = sum(confidences) / len(confidences)
    agreement_bonus = min(0.16, max(0, sources_count - 1) * 0.08)
    return round(min(1.0, (highest * 0.7) + (average * 0.3) + agreement_bonus), 3)


def _remove_subsumed(findings: List[Dict]) -> List[Dict]:
    """Drop findings whose span is fully contained within a higher-confidence
    finding of a *different* type (e.g. 'john.smith' Person Name inside an
    Email span, or a Phone Number fragment inside a Credit Card span)."""
    result = []
    for i, finding in enumerate(findings):
        f_start = finding.get("start", 0)
        f_end = finding.get("end", 0)
        f_conf = finding.get("confidence", 0.0)
        f_type = finding.get("type")
        dominated = False
        for j, other in enumerate(findings):
            if i == j or other.get("type") == f_type:
                continue
            o_start = other.get("start", 0)
            o_end = other.get("end", 0)
            o_conf = other.get("confidence", 0.0)
            # Drop finding if it is fully subsumed by a >= confidence other.
            if o_start <= f_start and o_end >= f_end and o_conf >= f_conf:
                dominated = True
                break
        if not dominated:
            result.append(finding)
    return result


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
        base_type = str(base.get("type", "")).lower()
        for j in range(i + 1, len(tagged)):
            if used[j]:
                continue
            other = tagged[j]
            other_type = str(other.get("type", "")).lower()
            same_type = base_type == other_type
            spans_overlap = _overlap(base["start"], base["end"], other["start"], other["end"])
            # Merge when same type AND spans overlap, OR same exact value (cross-engine same entity).
            same_value = base.get("value") and base.get("value") == other.get("value") and same_type
            if (same_type and spans_overlap) or same_value:
                cluster.append(other)
                used[j] = True

        # Pick type and representative value from the highest-confidence member.
        best = max(cluster, key=lambda c: _confidence(c))
        sources = list({c.get("source_engine", c.get("engine")) for c in cluster})
        confidences = [_confidence(c) for c in cluster]
        source_confidences = {
            str(c.get("source_engine", c.get("engine"))): _confidence(c)
            for c in cluster
        }
        highest = max(confidences) if confidences else 0.0
        severities = [str(c.get("severity", "medium")).lower() for c in cluster]
        merged_severity = max(severities, key=lambda item: SEVERITY_RANK.get(item, 0)) if severities else "medium"

        raw_value = best.get("value") or cluster[0].get("value") or ""
        masked = raw_value[:3] + "***" + (raw_value[-3:] if len(raw_value) > 3 else "") if raw_value else ""
        merged_record = {
            "type": best.get("type") or cluster[0].get("type"),
            "value": raw_value,
            "masked_value": masked,
            "start": min(c["start"] for c in cluster),
            "end": max(c["end"] for c in cluster),
            "line": best.get("line") or cluster[0].get("line"),
            "sources": sources,
            "source_confidences": source_confidences,
            "confidence": _consensus_score(confidences, len(sources)),
            "consensus_score": _consensus_score(confidences, len(sources)),
            "highest_confidence": highest,
            "engines_count": len(sources),
            "severity": merged_severity,
            "representative_context": best.get("context") or cluster[0].get("context"),
        }
        merged.append(merged_record)

    merged.sort(key=lambda r: (r["start"], r.get("type", "")))
    merged = _remove_subsumed(merged)
    meta["total_merged"] = len(merged)
    logger.info("Entity aggregation completed: merged=%s", len(merged))
    return merged, meta
