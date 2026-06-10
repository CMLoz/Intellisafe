"""Live standard-mode scan test — verifies GLiNER + Presidio + Regex all contribute."""
import sys
sys.path.insert(0, '.')

from app.backend.detection import ValidationPipeline

SAMPLE = """
Patient: John Smith, DOB: 1985-03-22
Email: john.smith@example.com  Phone: +1-555-867-5309
SSN: 123-45-6789  Credit Card: 4111 1111 1111 1111
IP: 192.168.1.100  API Key: sk-abc123XYZ789secret
Address: 742 Evergreen Terrace, Springfield, IL 62701
"""

print("Initializing ValidationPipeline (standard mode)...")
pipeline = ValidationPipeline()
result = pipeline.run(SAMPLE, mode="standard")

findings = result["findings"]
breakdown = result["confidence_breakdown"]
print(f"\nTotal findings: {len(findings)}\n")

for f in findings:
    src = f.get("engine") or ",".join(f.get("sources", []))
    conf = f.get("confidence") or f.get("source_confidences", {})
    sev = f.get("severity", "?").upper()[:6]
    typ = f.get("type", "?")[:25]
    val = str(f.get("masked_value", "?"))[:20]
    print(f"  [{sev:6}] {typ:25} | {val:20} | engine={src} conf={conf}")

print("\nConfidence breakdown:")
for engine, stats in breakdown.items():
    print(f"  {engine}: {stats}")

print("\nRisk distribution:", result.get("risk_distribution", {}))
