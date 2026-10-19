# uvicorn src.api.main:app --reload
import asyncio
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from typing import List

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.pii_redaction.presidio_service import ClinicalPIIRedactor

load_dotenv()

app = FastAPI(title="Zero-Trust Clinical RAG API — by Pradeep")

# Global state to hold our heavy ML models so they only load once at startup
middleware = {}

DEEPSEEK_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")
LLM_ENABLED   = bool(DEEPSEEK_KEY and not DEEPSEEK_KEY.startswith("your_"))
GEMINI_ENABLED = bool(GEMINI_KEY and not GEMINI_KEY.startswith("your_"))


async def _gemini_llm_response(context: str, question: str) -> str:
    """Call Google Gemini Flash for a real clinical natural-language answer.
    Retries with exponential backoff on 503; falls back to mock on persistent failure."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_KEY)

    system_instruction = (
        "You are a clinical decision-support assistant. "
        "You will be given de-identified EHR records (PII already removed by Presidio). "
        "Answer the clinician's question concisely, citing only what is in the context. "
        "Never speculate beyond the provided data. "
        "Always end with: ⚕️ AI-generated summary. Do not use for diagnostic purposes."
    )
    prompt = (
        f"De-identified Clinical Context:\n{context}\n\n"
        f"Clinician Question: {question}"
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=512,
    )

    # Confirmed working with AQ. key type (tested live)
    model_chain = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    last_error = None

    for model in model_chain:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return response.text
            except Exception as e:
                err = str(e)
                last_error = err
                if "503" in err or "UNAVAILABLE" in err or "high demand" in err:
                    wait = 2 ** attempt          # 1s, 2s, 4s
                    await asyncio.sleep(wait)
                    continue                      # retry same model
                else:
                    break                        # non-503 error → try next model

    # All models exhausted → graceful fallback to structured mock
    print(f"WARNING: Gemini unavailable ({last_error[:80]}...) — falling back to mock")
    return await _mock_llm_response(context, question)


async def _mock_llm_response(context: str, question: str) -> str:
    """Smart mock that parses retrieved RAG context and produces a structured
    clinical summary — fully demonstrable without an LLM API key."""
    import re

    # --- Extract unique values from the redacted context records ---
    drugs     = list(dict.fromkeys(re.findall(r"Drug:\s*([^\(,]+)", context)))
    diagnoses = list(dict.fromkeys(re.findall(r"Diagnosis:\s*([^\n,D]+?)(?=\s*Drug:|$)", context)))
    events    = list(dict.fromkeys(re.findall(r"Event:\s*([^\s,<][^,]*?)(?=,\s*Test:|$)", context)))
    tests     = list(dict.fromkeys(re.findall(r"Test:\s*([^,]+)", context)))
    comments  = list(dict.fromkeys(re.findall(r"Comments:\s*([^,]+)", context)))

    def clean(lst):
        return [x.strip() for x in lst if x.strip() and "<" not in x]

    drugs     = clean(drugs)
    diagnoses = clean(diagnoses)
    events    = clean(events)
    tests     = clean(tests)
    comments  = clean(comments)

    lines = ["**📋 Clinical Summary  (Demo Mode — Presidio + pgvector pipeline active)**", ""]

    if drugs:
        lines.append(f"**💊 Medications:** {', '.join(drugs)}")
    if diagnoses:
        lines.append(f"**🩺 Diagnosis:** {', '.join(diagnoses)}")
    if events:
        lines.append(f"**📅 Encounter Types:** {', '.join(set(events))}")
    if tests:
        lines.append(f"**🧪 Lab Tests:** {', '.join(tests)}")
    if comments:
        lines.append(f"**📝 Lab Results:** {comments[0]}")

    lines += [
        "",
        f"*Query: \"{question}\"*",
        "",
        "> ℹ️ PII redacted by Microsoft Presidio · Context retrieved via pgvector semantic search · "
        "Add `DEEPSEEK_API_KEY` to `.env` for full natural-language answers.",
        "",
        "⚕️ *AI-generated summary. Do not use for diagnostic purposes.*"
    ]
    return "\n".join(lines)


@app.on_event("startup")
async def startup_event():
    print("Booting Enterprise AI Middlewares...")

    # 1. Load Presidio (spaCy NER for PII redaction)
    middleware["redactor"] = ClinicalPIIRedactor()

    # 2. Load NeMo Guardrails only if API key present
    if LLM_ENABLED:
        from nemoguardrails import RailsConfig, LLMRails
        config = RailsConfig.from_path("./src/guardrails")
        middleware["rails"] = LLMRails(config)
        print("NeMo Guardrails loaded (DeepSeek LLM active)")
    else:
        middleware["rails"] = None
        print("WARNING: DEEPSEEK_API_KEY not set — running in DEMO MODE (mock LLM)")

    # 3. Load local BioClinical ModernBERT embedding model
    print("Loading BioClinical ModernBERT embedding model...")
    middleware["embedder"] = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')

    print("System Ready on port 8000.")



# --- DATABASE CONNECTION HELPER ---
def get_db_engine():
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if db_host:
        DB_URL = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        DB_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:password@localhost:5432/clinical_db")

    return create_engine(DB_URL)


# --- AUTH MIDDLEWARE (Gap 7: Zero-Trust API key) ---
import json
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
_AUTH_ENABLED = bool(API_SECRET_KEY and not API_SECRET_KEY.startswith("your_"))

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce X-API-Key header on protected routes when API_SECRET_KEY is set."""
    protected = request.url.path.startswith("/api/v1/chat") or \
                request.url.path == "/api/v1/clinical-query"
    if _AUTH_ENABLED and protected:
        key = request.headers.get("X-API-Key", "")
        if key != API_SECRET_KEY:
            return JSONResponse({"detail": "Unauthorized: invalid or missing X-API-Key"}, status_code=401)
    return await call_next(request)


# --- HEALTH CHECK (Gap 2) ---
@app.get("/health")
async def health_check():
    """Liveness probe — confirms API and all middleware are loaded."""
    return {
        "status": "healthy",
        "llm_mode": "deepseek" if LLM_ENABLED else ("gemini-3.8-flash" if GEMINI_ENABLED else "demo_mock"),
        "auth_enabled": _AUTH_ENABLED,
        "middleware_loaded": list(middleware.keys()),
    }


# --- MODELS ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    patient_id: str
    messages: List[ChatMessage]

class ClinicalQuery(BaseModel):
    patient_id: str
    prompt: str


# --- ENDPOINT 1: HEALTH CHECK (MOCK DB) ---
@app.post("/api/v1/clinical-query")
async def process_clinical_query(query: ClinicalQuery):
    try:
        raw_db_context = f"""
        Patient John Doe (ID: {query.patient_id}) was admitted on March 15th.
        Last recorded Furosemide dosage was 40mg IV. 
        Attending physician: Dr. Gregory House, ID: 20043.
        """

        redactor = middleware["redactor"]
        safe_context = redactor.redact_clinical_context(raw_text=raw_db_context)

        augmented_prompt = f"Clinical Context:\n{safe_context}\n\nUser Question: {query.prompt}"

        rails = middleware["rails"]
        if rails:
            response = await rails.generate_async(messages=[{"role": "user", "content": augmented_prompt}])
            llm_response = response['content']
        else:
            llm_response = await _mock_llm_response(safe_context, query.prompt)

        return {
            "status": "success",
            "redacted_context_used": safe_context.strip(),
            "llm_response": llm_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 2: PRODUCTION CHAT (NATIVE PGVECTOR DB) ---
@app.post("/api/v1/chat")
async def process_chat(request: ChatRequest):
    try:
        latest_question = request.messages[-1].content
        engine = get_db_engine()

        # 1. Embed the user's question into a 768-dim vector
        embedder = middleware["embedder"]
        query_vector = embedder.encode(latest_question).tolist()

        # 2. Native pgvector similarity search on patient_encounters table
        with engine.connect() as conn:
            query = text("""
                SELECT drug, dose_val_rx, dose_unit_rx, route, eventtype, test_name, comments, description 
                FROM patient_encounters 
                WHERE subject_id = :subject_id AND clinical_embedding IS NOT NULL
                ORDER BY clinical_embedding <=> CAST(:query_embedding AS vector)
                LIMIT 5;
            """)

            result = conn.execute(query, {
                "subject_id": int(request.patient_id),
                "query_embedding": str(query_vector)
            })
            rows = result.fetchall()

            if not rows:
                real_db_context = f"No historical records found for patient {request.patient_id}."
            else:
                context_lines = []
                for row in rows:
                    context_lines.append(
                        f"Drug: {row.drug} ({row.dose_val_rx} {row.dose_unit_rx}), Route: {row.route}, "
                        f"Event: {row.eventtype}, Test: {row.test_name}, Comments: {row.comments}, Diagnosis: {row.description}"
                    )
                real_db_context = f"[Records for Patient ID: {request.patient_id}]\n" + "\n".join(context_lines)

        # 3. Redact the real context via Presidio
        redactor = middleware["redactor"]
        safe_context = redactor.redact_clinical_context(raw_text=real_db_context)

        # 4. Assemble Prompt
        augmented_prompt = f"Clinical Context:\n{safe_context}\n\nUser Question: {latest_question}"

        # 5. Format History for Guardrails
        nemo_history = [{"role": msg.role, "content": msg.content} for msg in request.messages[:-1]]
        nemo_history.append({"role": "user", "content": augmented_prompt})

        # 6. Route through LLM: DeepSeek (NeMo) > Gemini Flash > Smart Mock
        rails = middleware["rails"]
        if rails:
            response = await rails.generate_async(messages=nemo_history)
            llm_response = response['content']
        elif GEMINI_ENABLED:
            llm_response = await _gemini_llm_response(safe_context, latest_question)
        else:
            llm_response = await _mock_llm_response(safe_context, latest_question)

        return {"status": "success", "llm_response": llm_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3: FETCH UNIQUE PATIENTS (EMBEDDINGS ONLY) ---
@app.get("/api/v1/patients")
async def get_unique_patients():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Only fetch patients who actually have generated embeddings
            query = text("""
                SELECT DISTINCT subject_id 
                FROM patient_encounters 
                WHERE subject_id IS NOT NULL AND clinical_embedding IS NOT NULL 
                ORDER BY subject_id;
            """)
            result = conn.execute(query)
            patients = [str(row[0]) for row in result]
            
            return {"patients": patients if patients else ["No embedded patients found"]}
            
    except Exception as e:
        return {"patients": [], "error": str(e)}


# --- ENDPOINT 4: STREAMING CHAT (Gap 8: SSE token-by-token) ---
@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Streaming variant of /api/v1/chat using Server-Sent Events.
    Yields tokens word-by-word so the UI can render progressively.
    Swap the word-split loop for a provider streaming SDK when API key is set.
    """
    latest_question = request.messages[-1].content

    async def _event_generator():
        try:
            engine = get_db_engine()
            embedder = middleware["embedder"]
            query_vector = embedder.encode(latest_question).tolist()

            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT drug, dose_val_rx, dose_unit_rx, route, eventtype,
                               test_name, comments, description
                        FROM patient_encounters
                        WHERE subject_id = :sid AND clinical_embedding IS NOT NULL
                        ORDER BY clinical_embedding <=> CAST(:vec AS vector)
                        LIMIT 5;
                    """),
                    {"sid": int(request.patient_id), "vec": str(query_vector)}
                )
                rows = result.fetchall()

            if rows:
                ctx_lines = [
                    f"Drug: {r.drug}, Event: {r.eventtype}, Diagnosis: {r.description}"
                    for r in rows
                ]
                context = "\n".join(ctx_lines)
            else:
                context = f"No records found for patient {request.patient_id}."

            redactor = middleware["redactor"]
            safe_context = redactor.redact_clinical_context(raw_text=context)

            rails = middleware["rails"]
            if rails:
                nemo_msgs = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
                nemo_msgs.append({"role": "user", "content": f"Context:\n{safe_context}\n\nQ: {latest_question}"})
                full_response = await rails.generate_async(messages=nemo_msgs)
                reply = full_response['content']
            else:
                reply = await _mock_llm_response(safe_context, latest_question)

            # Stream word-by-word; swap for provider streaming SDK with a real key
            for word in reply.split():
                payload = json.dumps({"token": word + " "})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.02)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")