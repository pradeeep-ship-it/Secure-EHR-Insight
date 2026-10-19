"""
04b_embed_per_patient.py
Ensures every patient in the DB has at least SAMPLE_PER_PATIENT embedded rows.
Far faster than a single pass over 232K rows — useful for demo/dev environments.
Run: .venv/Scripts/python scripts/04b_embed_per_patient.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SAMPLE_PER_PATIENT = 500   # embed 500 rows per patient — enough for rich semantic search
BATCH_SIZE = 50

def embed_per_patient():
    load_dotenv()
    db_url = (
        f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(db_url)

    print("Loading BioClinical ModernBERT model...")
    model = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')
    assert model.get_sentence_embedding_dimension() == 768, "Dimension mismatch!"

    with engine.begin() as conn:
        patients = [
            row[0] for row in conn.execute(
                text("SELECT DISTINCT subject_id FROM patient_encounters ORDER BY subject_id")
            )
        ]
        print(f"Found {len(patients)} patients: {patients}")

        for pid in patients:
            # Count already embedded for this patient
            already = conn.execute(
                text("SELECT COUNT(*) FROM patient_encounters WHERE subject_id=:p AND clinical_embedding IS NOT NULL"),
                {"p": pid}
            ).scalar()

            need = max(0, SAMPLE_PER_PATIENT - already)
            if need == 0:
                print(f"  Patient {pid}: already has >={SAMPLE_PER_PATIENT} embeddings. Skipping.")
                continue

            print(f"  Patient {pid}: embedding {need} more rows (already have {already})...")
            processed = 0

            with tqdm(total=need, desc=f"  P{pid}", unit="rows") as pbar:
                while processed < need:
                    batch = conn.execute(
                        text("""
                            SELECT id, admission_type, drug, test_name,
                                   drg_severity, description, comments
                            FROM patient_encounters
                            WHERE subject_id = :p AND clinical_embedding IS NULL
                            LIMIT :bs
                        """),
                        {"p": pid, "bs": min(BATCH_SIZE, need - processed)}
                    ).mappings().fetchall()

                    if not batch:
                        break

                    texts, ids = [], []
                    for row in batch:
                        parts = []
                        if row['admission_type']: parts.append(f"Admission: {row['admission_type']}")
                        if row['drug']:           parts.append(f"Drug: {row['drug']}")
                        if row['test_name']:      parts.append(f"Test: {row['test_name']}")
                        if row['drg_severity']:   parts.append(f"Severity: {row['drg_severity']}")
                        if row['description']:    parts.append(f"Diagnosis: {row['description']}")
                        if row['comments']:       parts.append(f"Notes: {row['comments'][:250]}")
                        texts.append(" | ".join(parts))
                        ids.append(row['id'])

                    embeddings = model.encode(texts, batch_size=BATCH_SIZE).tolist()

                    conn.execute(
                        text("UPDATE patient_encounters SET clinical_embedding = :emb WHERE id = :id"),
                        [{"id": rid, "emb": str(emb)} for rid, emb in zip(ids, embeddings)]
                    )
                    processed += len(batch)
                    pbar.update(len(batch))

    print("\nAll patients now have embeddings. Run 05_test_vector_search.py to verify.")

if __name__ == "__main__":
    embed_per_patient()
