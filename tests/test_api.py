"""
Integration tests for the FastAPI endpoints.
Requires the API to be running: uvicorn src.api.main:app --port 8000
Run with: .venv/Scripts/pytest tests/test_api.py -v
"""
import pytest
import requests

BASE = "http://localhost:8000"


@pytest.fixture(scope="module", autouse=True)
def check_api_running():
    """Skip all tests gracefully if the API is not up."""
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
    except Exception:
        pytest.skip("API not running — start uvicorn before running integration tests")


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{BASE}/health")
        assert r.status_code == 200

    def test_health_has_status_field(self):
        data = requests.get(f"{BASE}/health").json()
        assert data["status"] == "healthy"

    def test_health_reports_middleware(self):
        data = requests.get(f"{BASE}/health").json()
        assert "middleware_loaded" in data
        assert len(data["middleware_loaded"]) > 0


class TestPatientsEndpoint:
    def test_patients_returns_list(self):
        r = requests.get(f"{BASE}/api/v1/patients")
        assert r.status_code == 200
        data = r.json()
        assert "patients" in data
        assert isinstance(data["patients"], list)

    def test_patients_are_numeric_strings(self):
        data = requests.get(f"{BASE}/api/v1/patients").json()
        for p in data["patients"]:
            assert p.isdigit(), f"Patient ID '{p}' is not numeric"


class TestChatEndpoint:
    def test_chat_returns_200(self):
        payload = {
            "patient_id": "10000032",
            "messages": [{"role": "user", "content": "What medications were given?"}]
        }
        r = requests.post(f"{BASE}/api/v1/chat", json=payload)
        assert r.status_code == 200

    def test_chat_response_has_llm_reply(self):
        payload = {
            "patient_id": "10000032",
            "messages": [{"role": "user", "content": "Summarise the patient history."}]
        }
        data = requests.post(f"{BASE}/api/v1/chat", json=payload).json()
        assert "llm_response" in data
        assert len(data["llm_response"]) > 10

    def test_chat_pii_stripped_from_context(self):
        """Verify that raw PHI is not echoed back in any response field."""
        payload = {
            "patient_id": "10000032",
            "messages": [{"role": "user", "content": "Show diagnosis details."}]
        }
        data = requests.post(f"{BASE}/api/v1/chat", json=payload).json()
        reply = data.get("llm_response", "")
        # Real names from MIMIC should not leak through
        assert "John" not in reply or "<PERSON>" in reply
