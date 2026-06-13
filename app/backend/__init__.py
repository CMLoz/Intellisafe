"""Backend package exports."""

from .privacy_risk_manager import PrivacyRiskManager
from .sensitive_data_tracker import SensitiveDataTracker, compute_file_hash

__all__ = ["PrivacyRiskManager", "RedactionEngine", "SensitiveDataTracker", "compute_file_hash"]


def __getattr__(name):
    if name == "RedactionEngine":
        from .redaction_engine import RedactionEngine

        return RedactionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
