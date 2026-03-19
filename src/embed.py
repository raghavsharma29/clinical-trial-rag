# src/embed.py
import os, json
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME  = "ClinicalTrial"
BATCH_SIZE       = 50

# We use a general biomedical model — works well, no login required
model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.2")


def get_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )


def create_collection(client):
    """Create the ClinicalTrial collection if it doesn't exist."""
    existing = [c.name for c in client.collections.list_all().values()]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),  # we supply our own vectors
        properties=[
            Property(name="nct_id",        data_type=DataType.TEXT),
            Property(name="title",         data_type=DataType.TEXT),
            Property(name="section",       data_type=DataType.TEXT),
            Property(name="text",          data_type=DataType.TEXT),
            Property(name="status",        data_type=DataType.TEXT),
            Property(name="phase",         data_type=DataType.TEXT),
            Property(name="conditions",    data_type=DataType.TEXT_ARRAY),
            Property(name="interventions", data_type=DataType.TEXT_ARRAY),
            Property(name="min_age",       data_type=DataType.TEXT),
            Property(name="max_age",       data_type=DataType.TEXT),
            Property(name="gender",        data_type=DataType.TEXT),
        ],
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def embed_and_upload(chunks: list[dict]):
    """Embed all chunks and upload to Weaviate in batches."""
    client = get_client()
    create_collection(client)

    collection = client.collections.get(COLLECTION_NAME)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(chunks)} chunks with BioBERT...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

    print(f"Uploading to Weaviate in batches of {BATCH_SIZE}...")
    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for chunk, vector in tqdm(zip(chunks, vectors), total=len(chunks)):
            batch.add_object(
                properties={
                    "nct_id":        chunk.get("nct_id", ""),
                    "title":         chunk.get("title", ""),
                    "section":       chunk.get("section", ""),
                    "text":          chunk.get("text", ""),
                    "status":        chunk.get("status", ""),
                    "phase":         chunk.get("phase", ""),
                    "conditions":    chunk.get("conditions", []),
                    "interventions": chunk.get("interventions", []),
                    "min_age":       chunk.get("min_age", ""),
                    "max_age":       chunk.get("max_age", ""),
                    "gender":        chunk.get("gender", ""),
                },
                vector=vector.tolist(),
            )

    client.close()
    print(f"Done! {len(chunks)} chunks uploaded to Weaviate.")


def count_objects():
    """Check how many objects are in the collection."""
    client = get_client()
    collection = client.collections.get(COLLECTION_NAME)
    count = collection.aggregate.over_all(total_count=True).total_count
    client.close()
    print(f"Total objects in Weaviate: {count}")
    return count


if __name__ == "__main__":
    chunks_path = Path("data/processed/cancer_chunks.json")
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    embed_and_upload(chunks)
    count_objects()