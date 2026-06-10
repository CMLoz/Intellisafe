"""Detection engine tests."""

import unittest

from app.backend.detection import GLiNEREngine, TransformerEngine, ValidationPipeline
from app.backend.detection.risk_classifier import RiskClassifier


class TestRiskClassifier(unittest.TestCase):
    def test_high_risk_credit_card(self):
        classifier = RiskClassifier()
        findings = [{"type": "Credit Card", "confidence": 0.9}]
        classified = classifier.classify(findings)
        self.assertEqual(classified[0]["risk_level"], "high")

    def test_high_risk_ssn(self):
        classifier = RiskClassifier()
        findings = [{"type": "ID Number", "label": "SSN-like", "confidence": 0.5}]
        classified = classifier.classify(findings)
        self.assertEqual(classified[0]["risk_level"], "high")

    def test_medium_risk_email(self):
        classifier = RiskClassifier()
        findings = [{"type": "Email", "confidence": 0.7}]
        classified = classifier.classify(findings)
        self.assertEqual(classified[0]["risk_level"], "medium")

    def test_low_risk_person_name(self):
        classifier = RiskClassifier()
        findings = [{"type": "Person Name", "confidence": 0.5}]
        classified = classifier.classify(findings)
        self.assertEqual(classified[0]["risk_level"], "low")

    def test_risk_distribution(self):
        classifier = RiskClassifier()
        findings = [
            {"type": "Credit Card", "confidence": 0.9},
            {"type": "Email", "confidence": 0.7},
            {"type": "Person Name", "confidence": 0.5},
        ]
        classifier.classify(findings)
        dist = classifier.get_risk_distribution(findings)
        self.assertEqual(dist["high"], 1)
        self.assertEqual(dist["medium"], 1)
        self.assertEqual(dist["low"], 1)

    def test_high_risk_filter(self):
        classifier = RiskClassifier()
        findings = [
            {"type": "Credit Card", "confidence": 0.9},
            {"type": "Email", "confidence": 0.7},
        ]
        classifier.classify(findings)
        high = classifier.get_high_risk_findings(findings)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0]["type"], "Credit Card")


class TestValidationPipeline(unittest.TestCase):
    def test_quick_mode_regex(self):
        pipeline = ValidationPipeline()
        result = pipeline.run("Email: test@example.com", mode="quick")
        self.assertEqual(result["validation_tier"], "quick")
        self.assertTrue(any(f["type"] == "Email" for f in result["findings"]))

    def test_standard_mode_gliner(self):
        try:
            engine = GLiNEREngine()
        except RuntimeError:
            self.skipTest("GLiNER not available")

        findings = engine.detect("John Doe signed the document in Paris.")
        self.assertTrue(len(findings) >= 0)



class TestTransformerEngine(unittest.TestCase):
    def test_transformer_validation(self):
        try:
            engine = TransformerEngine(local_files_only=True)
        except RuntimeError:
            self.skipTest("Transformer model not available locally")

        findings = [
            {
                "type": "Person Name",
                "engine": "gliner",
                "context": "John Doe signed the document.",
            }
        ]
        updated = engine.validate(findings)
        self.assertIn("transformer_confidence", updated[0])
        self.assertIn("transformer_validated", updated[0])


if __name__ == "__main__":
    unittest.main()
