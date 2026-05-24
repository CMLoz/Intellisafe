"""Transformer-based validation using DistilBERT NER."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class TransformerEngine:
    """Validate spaCy entities with a DistilBERT NER model."""

    DEFAULT_MODEL = "elastic/distilbert-base-uncased-finetuned-conll03-english"
    LABEL_MAP = {
        "Person Name": "PER",
        "Organization": "ORG",
        "Location": "LOC",
        "Group": "MISC",
    }


    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        local_files_only: bool = False,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.local_files_only = local_files_only
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Transformer model could not be loaded. "
                "Ensure transformers + torch are installed and the model can be downloaded."
            ) from exc

        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )

    def validate(self, findings: List[Dict]) -> List[Dict]:
        """Attach transformer confidence to spaCy findings."""
        if not findings:
            return findings

        for finding in findings:
            if finding.get("engine") != "spacy":
                finding["transformer_confidence"] = None
                finding["transformer_validated"] = None
                continue

            expected = self.LABEL_MAP.get(finding.get("type"))
            if not expected:
                finding["transformer_confidence"] = 0.0
                finding["transformer_validated"] = False
                continue

            context = finding.get("context", "")
            if not context:
                finding["transformer_confidence"] = 0.0
                finding["transformer_validated"] = False
                continue

            entities = self._pipeline(context)
            best_score = 0.0
            for entity in entities:
                label = entity.get("entity_group", "")
                if label == expected:
                    score = float(entity.get("score", 0.0))
                    best_score = max(best_score, score)

            finding["transformer_confidence"] = best_score
            finding["transformer_validated"] = best_score >= 0.55

        logger.info("Transformer validation completed for %s findings", len(findings))
        return findings
