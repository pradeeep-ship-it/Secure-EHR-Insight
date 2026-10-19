# Code Walkthrough — File by File

---

## Project Layout

```
src/
  api/main.py              ← FastAPI app (entry point)
  pii_redaction/
    presidio_service.py    ← Microsoft Presidio wrapper
  guardrails/
    config.yml             ← NeMo model config (DeepSeek + embedder)
    rails.co               ← Colang intent/flow definitions
  database/
    schema.sql             ← PostgreSQL DDL
  ui/
    app.py                 ← Streamlit frontend

scripts/
  01_ingest_baseline_data.py    ← CSV → PostgreSQL
  02_verify_ingestion.py        ← sanity check query
  03_apply_vector_schema.py     ← ALTER TABLE ADD COLUMN vector(768)
  04_generate_embeddings.py     ← full-table embedding (batch)
  04b_embed_per_patient.py      ← smart per-patient sample embedding
  05_test_vector_search.py      ← manual pgvector search test
  06_test_guardrails.py         ← end-to-end guardrails smoke test

tests/
  test_pii_redactor.py     ← 5 unit tests for Presidio
  test_api.py              ← 8 integration tests for FastAPI
```

---

## `src/api/main.py` — FastAPI Application

**Key functions:**

### `startup_event()` — `@app.on_event("startup")`
Loads all three heavy ML models **once** into the `middleware` dict (global state):
- `middleware["redactor"]` — `ClinicalPIIRedactor` (spaCy NER)
- `middleware["rails"]` — `LLMRails` (NeMo Guardrails + DeepSeek) — skipped if no API key
- `middleware["embedder"]` — `SentenceTransformer` (BioClinical ModernBERT)

> **Why global dict?** FastAPI is async; loading 2 GB models per-request would be catastrophic. One-time load at startup, then shared across all coroutines.

### `api_key_middleware(request, call_next)` — `@app.middleware("http")`
Checks `X-API-Key` header on `/api/v1/chat*` routes. Returns `401` if key is wrong.
Only active when `API_SECRET_KEY` is set in `.env`.

### `GET /health`
Liveness probe — returns `middleware_loaded`, `llm_mode` (live vs demo), `auth_enabled`.

### `POST /api/v1/chat` — **Main production endpoint**
Pipeline:
1. `embedder.encode(question)` → `[float] × 768`
2. SQL: `ORDER BY clinical_embedding <=> CAST(:vec AS vector) LIMIT 5` (cosine ANN)
3. `redactor.redact_clinical_context(context)` → PII replaced with entity tags
4. `rails.generate_async(messages)` → guarded LLM call OR `_mock_llm_response()`
5. Returns `{"status": "success", "llm_response": "..."}`

### `POST /api/v1/chat/stream`
SSE variant — same pipeline, but yields `data: {"token": "word "}` events.
UI reads the stream and renders progressively with a `st.empty()` placeholder.

### `_mock_llm_response(context, question)` — async helper
Returns redacted context with a `[DEMO MODE]` banner. Lets the full pipeline run without a paid LLM key.

---

## `src/pii_redaction/presidio_service.py` — PII Redaction

### `ClinicalPIIRedactor.__init__()`
Sets up `AnalyzerEngine` and `AnonymizerEngine` with **two custom recognisers**:
1. **SSN override** — regex `\d{3}-\d{2}-\d{4}` at score 0.9 (overrides Presidio's checksum-requiring default)
2. **Hospital deny-list** — forces "Massachusetts General Hospital", "Mayo Clinic", etc. to be recognised as `ORGANIZATION`

> **Interview point:** Why override defaults? Presidio's built-in SSN recogniser requires a valid Luhn checksum. MIMIC-IV synthetic SSNs don't pass it. Custom recognisers fix this.

### `redact_clinical_context(raw_text)`
Calls `analyzer.analyze()` then `anonymizer.anonymize()`. Returns text with entities replaced by `<ENTITY_TYPE>` tags.

---

## `src/guardrails/config.yml` — NeMo Model Config

```yaml
models:
  - type: main       # LLM for response generation
    engine: openai
    model: deepseek-chat
    parameters:
      base_url: "https://api.deepseek.com/v1"
      api_key: $DEEPSEEK_API_KEY   # read from env

  - type: embeddings  # local model for intent classification
    engine: SentenceTransformers
    model: NeuML/bioclinical-modernbert-base-embeddings
```

---

## `src/guardrails/rails.co` — Colang Intent Rules

```colang
define user ask for medical advice
  "Can you prescribe me a new medication?"
  "What is the best treatment for this?"
  ...

define bot refuse medical advice
  "I am an enterprise EHR retrieval system. For legal and compliance
   reasons, I cannot provide new medical diagnoses..."

define flow prevent medical advice
  user ask for medical advice
  bot refuse medical advice
```

**How it works:** NeMo embeds the user message with BioClinical ModernBERT, cosine-matches against the example utterances, classifies intent, and fires the matching flow **before** the LLM sees the prompt.

---

## `src/database/schema.sql` — DDL

Creates `patient_encounters` with:
- Structured EHR columns: `subject_id`, `admission_type`, `drug`, `dose_val_rx`, `test_name`, `description`, etc.
- `clinical_embedding vector(768)` — pgvector column for ANN search
- Index: `CREATE INDEX … USING ivfflat (clinical_embedding vector_cosine_ops)` (if present)

---

## `src/ui/app.py` — Streamlit Frontend

| Component | Purpose |
|---|---|
| Health badge | Calls `/health` on every page load; stops app if API is offline |
| `fetch_patient_list()` | Cached (5 min TTL); fetches embedded patient IDs from `/api/v1/patients` |
| `st.sidebar.selectbox` | Searchable dropdown — Streamlit renders this natively |
| `use_streaming` toggle | Switches between `/api/v1/chat` and `/api/v1/chat/stream` |
| Streaming path | `requests.post(..., stream=True)` → `iter_lines()` → `st.empty().markdown()` |
| Disclaimer footer | Appended to every assistant message: *"AI-generated. Not for diagnosis."* |

---

## `scripts/` — Setup Pipeline

| Script | Purpose | Key technique |
|---|---|---|
| `01_ingest_baseline_data.py` | Reads MIMIC CSV → `pd.read_csv` → `df.to_sql` | Bulk insert via SQLAlchemy |
| `02_verify_ingestion.py` | Sample query to confirm row count and schema | Quick sanity check |
| `03_apply_vector_schema.py` | `ALTER TABLE ADD COLUMN clinical_embedding vector(768)` | Idempotent (IF NOT EXISTS) |
| `04_generate_embeddings.py` | Full-table embedding with `batch_size=50` | Batched PyTorch encoding |
| `04b_embed_per_patient.py` | 500 rows per patient (all 7) in ~4 min | Per-patient SQL LIMIT |
| `05_test_vector_search.py` | Direct pgvector ANN query | Tests cosine distance |
| `06_test_guardrails.py` | End-to-end LLM guardrails test | Requires DeepSeek key |
