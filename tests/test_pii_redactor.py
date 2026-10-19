"""
Unit tests for ClinicalPIIRedactor (Microsoft Presidio).
Run with: .venv/Scripts/pytest tests/ -v
"""
import pytest
from src.pii_redaction.presidio_service import ClinicalPIIRedactor


@pytest.fixture(scope="module")
def redactor():
    """Load redactor once for all tests in this module (heavy model init)."""
    return ClinicalPIIRedactor()


class TestPIIRedaction:
    def test_person_name_redacted(self, redactor):
        text = "Patient John Smith was admitted on Monday."
        result = redactor.redact_clinical_context(raw_text=text)
        assert "John Smith" not in result
        assert "<PERSON>" in result

    def test_ssn_redacted(self, redactor):
        text = "SSN: 123-45-6789"
        result = redactor.redact_clinical_context(raw_text=text)
        assert "123-45-6789" not in result

    def test_no_phi_unchanged(self, redactor):
        """Clinical terminology with no PII should pass through."""
        text = "Furosemide 40mg IV administered. Diagnosis: hepatic encephalopathy."
        result = redactor.redact_clinical_context(raw_text=text)
        assert "Furosemide" in result
        assert "hepatic encephalopathy" in result

    def test_hospital_name_redacted(self, redactor):
        text = "Referred to Massachusetts General Hospital for follow-up."
        result = redactor.redact_clinical_context(raw_text=text)
        assert "Massachusetts General Hospital" not in result

    def test_returns_string(self, redactor):
        result = redactor.redact_clinical_context(raw_text="Test input.")
        assert isinstance(result, str)
        assert len(result) > 0
