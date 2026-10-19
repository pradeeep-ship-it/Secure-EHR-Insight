# Secure EHR Insight — Clinical Validator

> **A zero-trust, RAG-based clinical query system over real MIMIC-IV EHR data.**
> PII is redacted by Microsoft Presidio before any data reaches an LLM. Medical-advice prompts are blocked by NeMo Guardrails. Semantic search is powered by pgvector and a domain-tuned BioClinical BERT model.

**Author:** Pradeep · **Stack:** Python · FastAPI · PostgreSQL + pgvector · Streamlit · NeMo Guardrails · Microsoft Presidio

---

## Features

- 🔍 **Semantic search** — BioClinical ModernBERT (768-dim) embeddings over MIMIC-IV EHR records via pgvector cosine ANN
- 🔒 **PII redaction** — Microsoft Presidio strips `<PERSON>`, `<US_SSN>`, `<ORGANIZATION>` before any LLM call
- 🛡️ **Guardrails** — NeMo Guardrails (Colang rules) blocks medical-advice and prescription intents deterministically
- ⚡ **Streaming** — SSE endpoint yields tokens word-by-word; toggle in the UI sidebar
- 🔑 **Auth middleware** — `X-API-Key` header enforcement on chat routes (when `API_SECRET_KEY` is set)
- 🩺 **Health check** — `GET /health` reports middleware state, LLM mode, auth status
- 🐳 **Docker-ready** — `docker-compose.yml` spins up Postgres + pgvector + API + UI in one command
- 🧪 **Tested** — 5 unit tests (Presidio) + 8 integration tests (FastAPI); run with `pytest`
- 🟡 **Demo mode** — full pipeline runs without a DeepSeek API key; mock LLM shows redacted context

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.64 |
| Backend API | FastAPI 0.141 + Uvicorn |
| Database | PostgreSQL 16 + pgvector extension |
| ORM / Driver | SQLAlchemy 2.0 + psycopg v3 |
| Embeddings | `NeuML/bioclinical-modernbert-base-embeddings` (768-dim) |
| Vector Search | pgvector cosine ANN (`<=>` operator) |
| PII Redaction | Microsoft Presidio 2.2 + spaCy `en_core_web_lg` |
| LLM Guardrails | NeMo Guardrails 0.24 (Colang) + DeepSeek-chat |
| Testing | pytest 8.3 |
| Containerisation | Docker + Docker Compose |

---

## Prerequisites

- Python 3.10+ (tested on 3.14)
- PostgreSQL 16 with pgvector extension
- (Optional) DeepSeek API key — [platform.deepseek.com](https://platform.deepseek.com)

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <your-repo-url>
cd Secure-EHR-Insight-Clinical-Validator

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# nemoguardrails requires a separate install on Python 3.14+
pip install nemoguardrails==0.24.1 --ignore-requires-python
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in your DB credentials and (optionally) DEEPSEEK_API_KEY
```

```env
DB_USER=fde_admin
DB_PASSWORD=SecureEHR2026!
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ehr_db
DEEPSEEK_API_KEY=your_key_here      # leave blank for demo mode
API_SECRET_KEY=                      # set to enforce X-API-Key auth
```

### 4. Set up PostgreSQL

```sql
-- Run as postgres superuser
CREATE DATABASE ehr_db;
CREATE USER fde_admin WITH PASSWORD 'SecureEHR2026!';
GRANT ALL PRIVILEGES ON DATABASE ehr_db TO fde_admin;
\c ehr_db
GRANT ALL ON SCHEMA public TO fde_admin;
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Ingest data & generate embeddings

```bash
# Windows — set UTF-8 encoding first
set PYTHONUTF8=1

python scripts/01_ingest_baseline_data.py   # load 232K MIMIC-IV rows
python scripts/02_verify_ingestion.py        # sanity check
python scripts/03_apply_vector_schema.py     # add vector(768) column
python scripts/04b_embed_per_patient.py      # embed 500 rows × 7 patients (~4 min)
python scripts/05_test_vector_search.py      # confirm ANN search works
```

---

## Run

### Option A — Local (two terminals)

```bash
# Terminal 1 — API (port 8000)
set PYTHONUTF8=1
.venv\Scripts\uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — UI (port 8501)
set PYTHONUTF8=1
.venv\Scripts\streamlit run src/ui/app.py
```

Open **http://localhost:8501** in your browser.

### Option B — Docker Compose (one command)

```bash
docker-compose up --build
```

Services:
- Postgres + pgvector: `localhost:5432`
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — middleware status, LLM mode |
| `GET` | `/api/v1/patients` | List all patients with embeddings |
| `POST` | `/api/v1/chat` | Full RAG pipeline → JSON response |
| `POST` | `/api/v1/chat/stream` | Full RAG pipeline → SSE token stream |
| `POST` | `/api/v1/clinical-query` | Demo endpoint with mock DB context |

Interactive docs: **http://localhost:8000/docs**

---

## Run Tests

```bash
# Integration tests (API must be running)
.venv\Scripts\pytest tests/test_api.py -v

# Unit tests (standalone — no API needed)
.venv\Scripts\pytest tests/test_pii_redactor.py -v

# All tests
.venv\Scripts\pytest tests/ -v
```

---

## Project Structure

```
src/
  api/main.py                  ← FastAPI app
  pii_redaction/
    presidio_service.py        ← Microsoft Presidio PII redactor
  guardrails/
    config.yml                 ← NeMo model config
    rails.co                   ← Colang intent / flow rules
  database/schema.sql          ← PostgreSQL DDL
  ui/app.py                    ← Streamlit frontend

scripts/
  01_ingest_baseline_data.py
  02_verify_ingestion.py
  03_apply_vector_schema.py
  04_generate_embeddings.py
  04b_embed_per_patient.py     ← recommended for dev/demo
  05_test_vector_search.py
  06_test_guardrails.py        ← requires DEEPSEEK_API_KEY

tests/
  test_pii_redactor.py         ← 5 unit tests
  test_api.py                  ← 8 integration tests

study/
  PROGRESS.md                  ← setup journal
  Secure-EHR-Insight/
    OVERVIEW.md                ← architecture + data flow
    CODE_WALKTHROUGH.md        ← file-by-file guide
    INTERVIEW_QA.md            ← 15 interview Q&As
    REVISE_TOPICS.md           ← concepts to revise

docker-compose.yml
Dockerfile
requirements.txt
.env.example
```

---

## Notes

- **Demo mode:** If `DEEPSEEK_API_KEY` is blank or starts with `your_`, the system runs without an LLM. The redacted pgvector context is returned directly — useful for demos and testing the full pipeline without API costs.
- **Auth:** Set `API_SECRET_KEY` in `.env` to enable `X-API-Key` enforcement. Leave blank to disable (open API).
- **Streaming:** Toggle "⚡ Streaming mode" in the Streamlit sidebar to switch to the SSE endpoint.
- **Data:** MIMIC-IV is a restricted dataset. The CSV in `data/` is for educational use only and must not be redistributed.

---

## License

MIT — for educational and interview portfolio purposes.
