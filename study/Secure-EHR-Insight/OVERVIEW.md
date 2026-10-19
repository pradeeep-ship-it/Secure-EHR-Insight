# Secure EHR Insight — Clinical Validator
### Project Overview
**Author:** Pradeep | **Stack:** Python · FastAPI · PostgreSQL + pgvector · Streamlit · NeMo Guardrails · Microsoft Presidio

---

## What It Does

A **zero-trust, RAG-based clinical query system** over real MIMIC-IV EHR data.

A clinician selects a patient, types a natural-language question ("What medications were prescribed at discharge?"), and the system:
1. Embeds the question with a domain-specific BioClinical BERT model
2. Retrieves the 5 most semantically relevant records from a pgvector database
3. Strips all PII from the retrieved context using Microsoft Presidio
4. Routes the safe, anonymised context through NeMo Guardrails (blocks medical-advice prompts)
5. Sends the guarded prompt to DeepSeek LLM and returns the answer

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (:8501)               │
│  Patient selector → chat input → response display   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (POST /api/v1/chat)
┌──────────────────────▼──────────────────────────────┐
│              FastAPI Backend (:8000)                 │
│                                                      │
│  ① Embed question  ──► BioClinical ModernBERT (768d) │
│  ② Vector search   ──► pgvector (<=> cosine)         │
│  ③ Redact PII      ──► Microsoft Presidio NER        │
│  ④ Guard prompt    ──► NeMo Guardrails (Colang)      │
│  ⑤ LLM call        ──► DeepSeek-chat (OpenAI compat) │
└──────────────────────┬──────────────────────────────┘
                       │ SQLAlchemy + psycopg v3
┌──────────────────────▼──────────────────────────────┐
│         PostgreSQL 16 + pgvector extension           │
│    Table: patient_encounters                         │
│    Columns: subject_id, drug, diagnosis, …           │
│             clinical_embedding vector(768)           │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow (per query)

```
User question
    │
    ▼
SentenceTransformer.encode()  →  768-dim vector
    │
    ▼
pgvector: SELECT … ORDER BY embedding <=> query_vec LIMIT 5
    │
    ▼
Presidio AnalyzerEngine + AnonymizerEngine
    │  (replaces PERSON, SSN, ORG with <PERSON>, <US_SSN>, <ORGANIZATION>)
    ▼
NeMo Guardrails (rails.generate_async)
    │  (Colang rules: blocks "prescribe", "diagnose", "recommend drug" intents)
    ▼
DeepSeek LLM → natural language answer
    │
    ▼
Streamlit chat bubble
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| pgvector over Pinecone | Co-located with structured EHR data; no extra service; SQL joins possible |
| BioClinical ModernBERT | Domain-tuned on clinical notes → better semantic match than generic BERT |
| Presidio before LLM | PII never reaches the cloud LLM — privacy by architecture, not policy |
| NeMo Guardrails | Rule-based intent detection runs before LLM — deterministic safety layer |
| Streaming SSE | Progressive token delivery improves perceived latency in the UI |
| Mock LLM fallback | Full pipeline demoed without a paid API key — useful for interviews |

---

## Dataset

**MIMIC-IV** (Medical Information Mart for Intensive Care) — de-identified ICU patient data from Beth Israel Deaconess Medical Center.
- 7 patients, 232,158 rows ingested
- 500 rows per patient embedded (3,500 total vectors) for demo
- Fields: `subject_id`, `admission_type`, `drug`, `dose_val_rx`, `test_name`, `description`, `comments`
