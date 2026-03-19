# src/retrieve.py
import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME  = "ClinicalTrial"

model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.2")


def get_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )


def build_patient_query(patient: dict) -> str:
    """Turn a patient profile dict into a natural language query string."""
    parts = []
    if patient.get("condition"):
        parts.append(f"Patient diagnosed with {patient['condition']}.")
    if patient.get("age"):
        parts.append(f"Age: {patient['age']}.")
    if patient.get("gender"):
        parts.append(f"Gender: {patient['gender']}.")
    if patient.get("history"):
        parts.append(f"Medical history: {patient['history']}.")
    if patient.get("medications"):
        parts.append(f"Current medications: {patient['medications']}.")
    return " ".join(parts)


def retrieve_trials(patient: dict, top_k: int = 10) -> list[dict]:
    """
    Given a patient profile, retrieve the top_k most relevant trial chunks.
    Uses dense vector search (semantic similarity).
    """
    query_text = build_patient_query(patient)
    print(f"\nQuery: {query_text}")

    # Embed the query
    query_vector = model.encode(query_text).tolist()

    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)

    # Dense vector search
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
        return_properties=[
            "nct_id", "title", "section", "text",
            "status", "phase", "conditions",
            "interventions", "min_age", "max_age", "gender"
        ],
    )
    client.close()

    # Format results
    retrieved = []
    for obj in results.objects:
        props = obj.properties
        retrieved.append({
            **props,
            "score": round(1 - obj.metadata.distance, 4),
        })

    print(f"Retrieved {len(retrieved)} chunks.")
    return retrieved


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Keep only the most relevant chunk per trial (by NCT ID)."""
    seen = {}
    for chunk in chunks:
        nct_id = chunk["nct_id"]
        if nct_id not in seen or chunk["score"] > seen[nct_id]["score"]:
            seen[nct_id] = chunk
    deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    print(f"After deduplication: {len(deduped)} unique trials.")
    return deduped


if __name__ == "__main__":
    # Test with a sample patient
    sample_patient = {
        "condition": "non-small cell lung cancer",
        "age": "58",
        "gender": "Male",
        "history": "Stage III, previously treated with chemotherapy",
        "medications": "carboplatin, paclitaxel",
    }

    chunks = retrieve_trials(sample_patient, top_k=20)
    trials  = deduplicate(chunks)

    print("\nTop 5 matched trials:")
    for i, t in enumerate(trials[:5], 1):
        print(f"\n{i}. {t['title']}")
        print(f"   NCT ID : {t['nct_id']}")
        print(f"   Score  : {t['score']}")
        print(f"   Phase  : {t['phase']}")
        print(f"   Status : {t['status']}")
        print(f"   Section: {t['section']}")