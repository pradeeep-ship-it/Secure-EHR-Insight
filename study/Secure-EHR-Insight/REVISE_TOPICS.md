# Revise Topics — Secure EHR Insight

Topics to revise before interviews on this project. Each section lists the core concept, why it appears in this codebase, and what to be able to explain.

---

## 1. Databases & SQL

### PostgreSQL
- `ALTER TABLE … ADD COLUMN vector(768)` — schema migration pattern
- `CREATE EXTENSION IF NOT EXISTS vector` — pgvector install
- `DISTINCT`, `WHERE … IS NOT NULL`, `ORDER BY … LIMIT` — used in all queries
- Connection strings: `postgresql+psycopg://user:pass@host:port/db`
- **Understand:** transactions (`engine.begin()` vs `engine.connect()`), connection pooling

### pgvector
- How IVFFlat index works: k-means partitioning, `lists` and `probes` params
- How HNSW differs (graph-based, better recall, higher memory)
- `<=>` cosine distance vs `<->` Euclidean vs `<#>` inner product — when to use each
- **Understand:** ANN vs exact kNN, recall/speed tradeoff

### SQLAlchemy
- `create_engine`, `text()`, `conn.execute()`
- `engine.begin()` (auto-commit transaction) vs `engine.connect()` (manual commit)
- Named bind parameters: `text("WHERE id = :id")` + `{"id": val}`

---

## 2. Machine Learning & NLP

### Transformer Embeddings
- How sentence embeddings work: CLS token pooling or mean pooling over last hidden state
- Why domain-specific models (BioClinical ModernBERT) outperform general ones on clinical text
- Embedding dimension: 768 (BERT-base) vs 1536 (OpenAI ada-002)
- **Understand:** cosine similarity, dot product similarity, L2 norm

### RAG (Retrieval-Augmented Generation)
- **Core loop:** Embed query → vector search → augment prompt with retrieved context → LLM
- Why RAG vs fine-tuning: no retraining needed, fresh data, source attribution possible
- Chunking strategies: sentence-level, paragraph-level, sliding window
- **Understand:** what goes wrong when retrieval fails (hallucination, irrelevant context)

### NLP — Named Entity Recognition (NER)
- How Presidio uses spaCy's NER to detect `PERSON`, `ORG`, `GPE` entities
- Difference between rule-based NER (deny-list, regex) and ML-based NER
- Why custom recognisers were needed (Luhn checksum, hospital names)

---

## 3. System Design

### Microservices Architecture
- How the 3 services interact: Streamlit ↔ FastAPI ↔ PostgreSQL
- Stateless API design: all state in DB and `middleware` dict (loaded at startup)
- Health check patterns: liveness vs readiness probes

### API Design (FastAPI / REST)
- Path params vs query params vs request body (Pydantic models)
- `@app.on_event("startup")` — deferred heavy initialization
- `@app.middleware("http")` — cross-cutting concerns (auth, logging)
- `StreamingResponse` + `media_type="text/event-stream"` for SSE
- HTTP status codes: 200, 401, 422 (Pydantic validation), 500

### Streaming — Server-Sent Events (SSE)
- SSE vs WebSockets: SSE is unidirectional (server → client), simpler, HTTP/1.1 compatible
- Event format: `data: <json>\n\n`
- `requests.post(..., stream=True)` + `iter_lines()` on the client
- **Understand:** when to use SSE vs WebSocket vs long-polling

### Security
- Zero-trust principle: never trust input; validate and redact at every layer
- API key auth vs OAuth2 / JWT — tradeoffs (simplicity vs granularity)
- PII redaction architecture: redact before any external call
- HIPAA relevance: PHI must not leave controlled environment

---

## 4. Python & Software Engineering

### Async Python
- `async def`, `await`, `asyncio` — how FastAPI runs coroutines
- `asyncio.sleep(0.02)` in streaming generator — yields control to event loop
- Why blocking calls (like `model.encode()`) should use `run_in_executor` for true async
- **Understand:** event loop, coroutine vs thread

### Python Packaging
- `__init__.py` — makes a directory a Python package; needed for `from src.x import y`
- `sys.stdout.reconfigure(encoding='utf-8')` — Windows console encoding fix
- `PYTHONIOENCODING=1` / `PYTHONUTF8=1` env vars

### Testing
- `pytest` fixtures: `scope="module"` for expensive one-time setup (model loading)
- `autouse=True` fixture for skip logic (`pytest.skip()` if API offline)
- Unit test vs integration test distinction
- **Understand:** test isolation, mocking (when to use `unittest.mock`)

---

## 5. Infrastructure & DevOps

### Docker & Docker Compose
- `FROM python:3.11-slim` — why slim over full (image size, attack surface)
- `depends_on` + `healthcheck` — don't start API until Postgres is ready
- `volumes` for local dev: mount `./src:/app/src` so code changes don't require rebuild
- Named volumes (`pgdata`) vs bind mounts — persistence vs convenience

### Environment Variables
- `.env` → `python-dotenv` → `os.getenv()` pattern
- Never commit `.env`; always commit `.env.example`
- `${VAR:-default}` syntax in `docker-compose.yml`

---

## 6. Clinical / Domain Knowledge (to contextualise the project)

- **MIMIC-IV:** de-identified ICU EHR dataset; `subject_id` = patient; records include medications, labs, diagnoses
- **HIPAA:** US law protecting health data; "minimum necessary" principle; PHI = any info linking data to a patient
- **PII in clinical notes:** names of physicians, patients, hospitals; dates; SSNs — all must be redacted before external processing
- **Why RAG for EHR:** LLMs hallucinate clinical facts; grounding answers in retrieved records reduces this; patient-locked retrieval (`WHERE subject_id = :pid`) prevents data leakage across patients

---

## Quick Revision Checklist

- [ ] Explain cosine similarity formula and why it's used for embeddings
- [ ] Draw the RAG pipeline from scratch
- [ ] Explain IVFFlat vs HNSW tradeoffs in 2 minutes
- [ ] Write a basic FastAPI endpoint with Pydantic model and middleware
- [ ] Explain what `async def` and `await` actually do in Python
- [ ] Describe HIPAA and why PII redaction placement matters
- [ ] Explain the difference between unit and integration tests with examples from this project
- [ ] Walk through docker-compose.yml and explain each service
