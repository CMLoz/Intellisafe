"""Detection engine tests."""

import unittest

from app.backend.detection import SpacyEngine, TransformerEngine, ValidationPipeline


class TestValidationPipeline(unittest.TestCase):
    def test_quick_mode_regex(self):
        pipeline = ValidationPipeline()
        result = pipeline.run("Email: test@example.com", mode="quick")
        self.assertEqual(result["validation_tier"], "quick")
        self.assertTrue(any(f["type"] == "Email" for f in result["findings"]))

    def test_standard_mode_spacy(self):
        try:
            engine = SpacyEngine()
        except RuntimeError:
            try:
                engine = SpacyEngine(model_name="en_core_web_sm")
            except RuntimeError:
                self.skipTest("spaCy model not available")

        findings = engine.detect("John Doe signed the document in Paris.")
        self.assertTrue(any(f["type"] == "Person Name" for f in findings))



class TestTransformerEngine(unittest.TestCase):
    def test_transformer_validation(self):
        try:
            engine = TransformerEngine(local_files_only=True)
        except RuntimeError:
            self.skipTest("Transformer model not available locally")

        findings = [
            {
                "type": "Person Name",
                "engine": "spacy",
                "context": "John Doe signed the document.",
            }
        ]
        updated = engine.validate(findings)
        self.assertIn("transformer_confidence", updated[0])
        self.assertIn("transformer_validated", updated[0])


if __name__ == "__main__":
    unittest.main()
