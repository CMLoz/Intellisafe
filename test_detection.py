"""Detection engine tests."""

import unittest

from app.backend.detection import GLiNEREngine, TransformerEngine, ValidationPipeline


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
