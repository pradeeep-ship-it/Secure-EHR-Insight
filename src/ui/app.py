import sys
import streamlit as st
import requests

API_CHAT_URL = "http://localhost:8000/api/v1/chat"
API_STREAM_URL = "http://localhost:8000/api/v1/chat/stream"
API_PATIENTS_URL = "http://localhost:8000/api/v1/patients"
API_HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(page_title="Zero-Trust Clinical EHR", layout="centered")
st.title("🏥 Enterprise EHR Chat")
st.caption("Protected by Presidio Zero-Trust & NeMo Guardrails — by Pradeep")


# --- HEALTH STATUS BADGE (Gap 3 improvement) ---
try:
    h = requests.get(API_HEALTH_URL, timeout=3)
    if h.status_code == 200:
        info = h.json()
        mode = "🟢 Live LLM" if info.get("llm_mode") == "live" else "🟡 Demo Mode"
        st.sidebar.success(f"API: Healthy | {mode}")
    else:
        st.sidebar.error("API: Unhealthy")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ API offline — start uvicorn first")
    st.stop()
except Exception as e:
    st.sidebar.warning(f"Health check failed: {e}")


# --- FETCH PATIENTS DYNAMICALLY ---
@st.cache_data(ttl=300)
def fetch_patient_list():
    """Fetch embedded patient IDs from the API with explicit error messages."""
    try:
        response = requests.get(API_PATIENTS_URL, timeout=10)
        response.raise_for_status()
        patients = response.json().get("patients", [])
        if not patients:
            return ["No embedded patients — run script 04 first"]
        return patients
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API at localhost:8000. Is the FastAPI server running?")
        return []
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned error {e.response.status_code}: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching patient list: {e}")
        return []


patient_list = fetch_patient_list()
if not patient_list:
    st.stop()

patient_id = st.sidebar.selectbox("Select Patient File (Type to search)", patient_list)
st.sidebar.info(f"🔒 Database locked to Patient: {patient_id}")

# Streaming toggle
use_streaming = st.sidebar.toggle("⚡ Streaming mode", value=False,
                                   help="Yields tokens word-by-word via SSE")

# --- CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# --- CHAT INPUT & API CALL ---
if prompt := st.chat_input(f"Ask about patient {patient_id}'s history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    payload = {"patient_id": patient_id, "messages": st.session_state.messages}

    with st.spinner("Analyzing securely..."):
        bot_reply = ""
        if use_streaming:
            # --- STREAMING PATH ---
            try:
                with requests.post(API_STREAM_URL, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    box = st.empty()
                    for line in resp.iter_lines():
                        if line:
                            raw = line.decode("utf-8")
                            if raw.startswith("data: "):
                                data = raw[6:]
                                if data == "[DONE]":
                                    break
                                import json
                                chunk = json.loads(data)
                                if "token" in chunk:
                                    bot_reply += chunk["token"]
                                    box.markdown(bot_reply + "▌")
                                elif "error" in chunk:
                                    bot_reply = f"⚠️ Stream error: {chunk['error']}"
                                    break
                    box.markdown(bot_reply)
            except requests.exceptions.ConnectionError:
                bot_reply = "❌ Cannot reach API. Is the server running?"
            except requests.exceptions.HTTPError as e:
                bot_reply = f"⚠️ API error {e.response.status_code}: {e.response.text}"
            except Exception as e:
                bot_reply = f"⚠️ Unexpected error: {e}"
        else:
            # --- STANDARD PATH ---
            try:
                response = requests.post(API_CHAT_URL, json=payload, timeout=60)
                response.raise_for_status()
                bot_reply = response.json().get("llm_response", "No response received.")
            except requests.exceptions.ConnectionError:
                bot_reply = "❌ Cannot reach API. Is the FastAPI server running on port 8000?"
            except requests.exceptions.HTTPError as e:
                bot_reply = f"⚠️ API returned {e.response.status_code}: {e.response.text}"
            except requests.exceptions.Timeout:
                bot_reply = "⏱️ Request timed out — the model may still be loading."
            except Exception as e:
                bot_reply = f"⚠️ Unexpected error: {e}"

    bot_reply += "\n\n*⚕️ AI-generated summary. Do not use for diagnostic purposes.*"
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)