# Interview Q&A — Secure EHR Insight Clinical Validator

**15 likely questions with concise answers for technical interviews.**

---

### 1. Walk me through the system end-to-end.

A clinician selects a patient in the Streamlit UI and types a question. The FastAPI backend embeds the question using BioClinical ModernBERT into a 768-dimension vector, runs a cosine ANN search against pgvector to retrieve the 5 most semantically relevant EHR records, redacts all PII from those records using Microsoft Presidio, routes the clean context through NeMo Guardrails (which blocks medical-advice intents via Colang rules), and finally calls the DeepSeek LLM to produce a natural-language answer.

---

### 2. Why pgvector instead of a dedicated vector DB like Pinecone or Weaviate?

**Tradeoff:** pgvector keeps vectors co-located with structured EHR data in PostgreSQL. This means you can do SQL JOINs (e.g., `WHERE subject_id = :pid`) combined with ANN search in a single query — no cross-service round-trip. The downside is scalability: pgvector's IVFFlat index doesn't match Pinecone's throughput at billions of vectors. For a hospital with millions of patients, a dedicated vector DB makes sense. For a department-level RAG system with thousands of records, pgvector is simpler, cheaper, and sufficient.

---

### 3. What is the `<=>` operator in your SQL query?

It's pgvector's cosine distance operator. `ORDER BY clinical_embedding <=> CAST(:vec AS vector) LIMIT 5` returns the 5 rows whose embedding vector has the smallest cosine distance to the query vector — i.e., the most semantically similar records. Cosine distance is preferred over Euclidean here because it measures angle (semantic direction), not magnitude — important since sentence embeddings can vary in length.

---

### 4. Why did you choose BioClinical ModernBERT over general-purpose models like `text-embedding-ada-002`?

Domain adaptation matters for clinical text. General embeddings are trained on web/book corpora where "Furosemide" appears rarely and in non-clinical contexts. BioClinical ModernBERT is fine-tuned on PubMed and MIMIC notes, so it understands clinical synonyms, abbreviations (e.g., "Lasix" = Furosemide), and severity nuances. In benchmarks on clinical NLP tasks, domain-tuned models consistently outperform general ones by 5–15% on retrieval precision.

---

### 5. What is NeMo Guardrails and how does it work here?

NeMo Guardrails is NVIDIA's library for adding programmable safety rails to LLM applications. It uses **Colang** — a declarative conversation flow language. You define example utterances for intents (`user ask for medical advice`) and actions (`bot refuse medical advice`). At runtime, NeMo embeds the user message, cosine-matches it against all defined example utterances, classifies the intent, and fires the matching flow **before** the prompt ever reaches the LLM. This is a deterministic safety layer — unlike prompting the LLM to "be safe", it can't be jailbroken through prompt injection.

---

### 6. Where does the PII redaction happen and why does placement matter?

Presidio runs **after** pgvector retrieval but **before** the LLM call. This is critical: the raw MIMIC-IV records contain real clinical notes with names, dates, and SSNs. If we sent those to an external LLM API, we'd be violating HIPAA. By redacting first, no PHI ever leaves the local environment. The `<PERSON>`, `<LOCATION>`, `<US_SSN>` tags in the redacted text are safe to transmit. This is "privacy by architecture" — the safety guarantee is structural, not reliant on the LLM's behaviour.

---

### 7. How does your custom Presidio SSN recogniser work and why was it needed?

Presidio's default SSN recogniser applies a Luhn-like checksum to validate SSNs. MIMIC-IV uses synthetic SSNs that don't pass this checksum, so they were silently ignored. I added a custom `PatternRecognizer` with the regex `\d{3}-\d{2}-\d{4}` at confidence score 0.9, which overrides the default. The lesson: always test redaction on your actual data — default recognisers are tuned for real-world patterns that may not match synthetic datasets.

---

### 8. What's the time complexity of the pgvector ANN search?

pgvector uses **IVFFlat** (Inverted File with Flat quantization). Index build is O(n) for k-means clustering into `lists` partitions. Query time is O(√n) on average — it probes `probes` clusters (default: 1) and does exact search within them. For our 3,500 vectors, it's effectively O(n) (linear scan) since the dataset is tiny. At 1M+ vectors, IVFFlat with `lists=1000, probes=10` gives ~95% recall at ~10× speedup over brute force. For even better performance, HNSW (also supported by pgvector) gives O(log n) queries with higher recall.

---

### 9. How does the streaming endpoint work?

`POST /api/v1/chat/stream` returns a `StreamingResponse` with `media_type="text/event-stream"` (SSE). An async generator yields `data: {"token": "word "}` events word-by-word with 20ms delays between them. The Streamlit UI reads the stream with `requests.post(..., stream=True)` → `resp.iter_lines()`, accumulates tokens, and updates an `st.empty()` placeholder on each chunk. This reduces perceived latency dramatically — the user sees the response forming rather than waiting for the full completion.

---

### 10. Why load the ML models at startup rather than per-request?

BioClinical ModernBERT is ~500MB; spaCy `en_core_web_lg` is ~700MB. Loading them per request would add 5–10 seconds of latency and crash under concurrent load. FastAPI's `@app.on_event("startup")` hook loads them once into a module-level `middleware` dict that all request handlers share. This is safe because `SentenceTransformer.encode()` and Presidio's analyzer are both stateless — they don't modify shared state, so there are no race conditions in async context.

---

### 11. What does your `/health` endpoint return and how would you use it in production?

```json
{
  "status": "healthy",
  "llm_mode": "demo_mock",
  "auth_enabled": false,
  "middleware_loaded": ["redactor", "rails", "embedder"]
}
```

In production: (1) Kubernetes liveness probe hits `/health` every 30s — restarts pod if it returns non-200. (2) Readiness probe checks `middleware_loaded` has all 3 keys before routing traffic. (3) A monitoring dashboard (e.g., Grafana + Prometheus) tracks `llm_mode` to alert if the LLM key expires and the system silently falls back to demo mode.

---

### 12. What's the difference between your two chat endpoints?

| | `/api/v1/chat` | `/api/v1/chat/stream` |
|---|---|---|
| Response | Full JSON after LLM completes | SSE events, token-by-token |
| Latency | Blocks until full response | First token in ~1s |
| Use case | Programmatic API clients | Streamlit UI, web frontends |
| Protocol | Standard HTTP | Server-Sent Events |

Both share identical pipeline logic (embed → pgvector → Presidio → Guardrails → LLM); only the delivery mechanism differs.

---

### 13. How does your auth middleware work and what are its limitations?

`api_key_middleware` is an `@app.middleware("http")` function that intercepts every request before routing. For protected paths (`/api/v1/chat*`), it checks the `X-API-Key` header against `API_SECRET_KEY` from `.env`. If wrong/missing, it returns 401 before the request handler runs.

**Limitations:** (1) Single static key — no per-user identity or RBAC. (2) Key in `.env` — rotations require a restart. **Production improvement:** use JWT tokens with short expiry, or OAuth2 with a proper identity provider (Auth0, Cognito). FastAPI has built-in `OAuth2PasswordBearer` for this.

---

### 14. If you had to scale this to 1 million patients, what would you change?

1. **Embeddings:** Move from pgvector IVFFlat to HNSW index, or a dedicated Pinecone/Qdrant cluster with horizontal sharding
2. **Database:** Partition `patient_encounters` by `subject_id` hash
3. **API:** Add a Redis cache for frequent queries (same patient, same question); add rate limiting per-user
4. **PII Redaction:** Presidio is CPU-bound; run it as a separate microservice with a pool of workers
5. **LLM:** Use async batching — accumulate requests over 50ms windows, batch them to the LLM API
6. **Embeddings at ingest:** Compute vectors at write-time, not query-time, using a Kafka consumer

---

### 15. What bug did you fix that was most interesting?

The **psycopg2 → psycopg v3 dialect mismatch**. The original scripts used `postgresql://` as the SQLAlchemy URL prefix, which routes to the `psycopg2` driver. We installed `psycopg` v3 (the new async-native driver), which requires `postgresql+psycopg://`. The error was `ModuleNotFoundError: No module named 'psycopg2'` — clear once you know the mapping, but a classic gotcha when migrating between driver versions.

Second interesting one: Python 3.14's console uses CP1252 encoding on Windows, causing `UnicodeEncodeError` on `✅` emoji in print statements. Fixed with `sys.stdout.reconfigure(encoding='utf-8')` at the top of every script and `PYTHONUTF8=1` in `.env`.
