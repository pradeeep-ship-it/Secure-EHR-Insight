# Project Progress — Secure-EHR-Insight Clinical Validator

## Phase 1: INSPECT ✅ Done

### Structure
- `src/api/main.py` — FastAPI backend (entry point: `uvicorn src.api.main:app`)
- `src/ui/app.py` — Streamlit frontend (entry point: `streamlit run src/ui/app.py`)
- `src/pii_redaction/presidio_service.py` — Microsoft Presidio PII redactor
- `src/guardrails/config.yml` + `rails.co` — NeMo Guardrails (DeepSeek LLM)
- `src/database/schema.sql` — PostgreSQL + pgvector schema
- `scripts/01–06_*.py` — Ordered setup scripts: ingest CSV → verify → vector schema → embeddings → test
- `data/MIMIC_IV_Trasncript.csv` — Clinical notes from MIMIC-IV dataset

### Stack
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL + pgvector
- **Frontend**: Streamlit
- **AI/ML**: NeMo Guardrails (DeepSeek-chat via OpenAI-compat), SentenceTransformers (BioClinical-ModernBERT), Microsoft Presidio NER
- **Language**: Python 3.x

### Entry Points
1. `uvicorn src.api.main:app --reload` (API on :8000)
2. `streamlit run src/ui/app.py` (UI on :8501)

### Dependencies Already Have: requirements.txt present with pinned versions

### Key Secrets Needed
- `DATABASE_URL` (PostgreSQL connection string)
- `DEEPSEEK_API_KEY` (for NeMo Guardrails LLM)

## Phase 2: IMPORT ✅ Done
- git init with user Pradeep
- Removed instructor_notes/ and uv_instructions.txt (other-owner material)
- Added src/**/__init__.py (5 files) for proper Python package resolution
- Created .env.example (no real secrets)
- Updated .gitignore (pycache, venv, logs, model cache)
- Patched src/guardrails/config.yml to read DEEPSEEK_API_KEY from env
## Phase 3: ENVIRONMENT ✅ Done
- .venv created (Python 3.14)
- Pass 1: fastapi, uvicorn, streamlit, psycopg, sqlalchemy, pgvector, pandas, numpy, dotenv
- Pass 2: torch, sentence-transformers, spacy+en_core_web_lg, presidio, transformers
- Pass 3: nemoguardrails==0.24.1 --ignore-requires-python (pure Python, runs fine on 3.14)
- All imports verified: CORE OK, PRESIDIO OK, NEMO OK 0.24.1
- requirements.txt updated with Pradeep authorship + install note
- .env created (DB creds set; DEEPSEEK_API_KEY needs real key)
## Phase 4: MAKE IT RUN ✅ Done
- Postgres 16 found; created fde_admin + granted pgvector extension
- Fixed psycopg2→psycopg v3 dialect in all 5 scripts
- Fixed Windows CP1252 emoji crash (PYTHONUTF8=1 + sys.stdout.reconfigure)
- Scripts 01–05 all passed: 232,158 rows ingested, vectorized, search confirmed
- Patched main.py: NeMo Guardrails optional — mock LLM when no API key
- API running: http://localhost:8000 — /api/v1/patients returns real patient IDs
- Chat pipeline verified: pgvector search + Presidio PII redaction both active
- UI running: http://localhost:8501
## Phase 5: COMPLETE IT ✅ Done
- Gap 1: New script 04b_embed_per_patient.py — embeds 500 rows × 7 patients in ~4 min
- Gap 2: GET /health endpoint added — reports llm_mode, auth_enabled, loaded middleware
- Gap 3: UI rewritten — explicit ConnectionError / HTTPError / Timeout handling everywhere
- Gap 4: tests/test_pii_redactor.py (5 unit) + tests/test_api.py (8 integration) — 8/8 PASSED
- Gap 5: docker-compose.yml + Dockerfile created (pgvector/pg16, api, ui services)
- Gap 6: README.md — deferred to Phase 7
- Gap 7: X-API-Key middleware added — enforced on /chat routes when API_SECRET_KEY is set
- Gap 8: POST /api/v1/chat/stream — SSE streaming endpoint, word-by-word; UI toggle added
## Phase 6: STUDY KIT ✅ Done
- OVERVIEW.md: architecture diagram, data flow, design decisions table
- CODE_WALKTHROUGH.md: every file, every key function, interview talking points
- INTERVIEW_QA.md: 15 Q&As (system design, pgvector, Presidio, NeMo, bugs fixed)
- REVISE_TOPICS.md: 6 sections + quick checklist (DB, ML, system design, Python, DevOps, clinical)
## Phase 7: README.md ✅ Done
- Features list, tech stack table, prerequisites
- Setup steps (venv, pip, .env, PostgreSQL, data ingestion)
- Run commands: local (2 terminals) + Docker Compose (1 command)
- API endpoint reference table + /docs link
- Full project structure tree
- Notes on demo mode, auth, streaming, MIMIC-IV data
