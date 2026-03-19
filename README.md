# Clinical Trial Matching RAG System

A production-grade Retrieval-Augmented Generation (RAG) system that matches patients to eligible clinical trials using real data from ClinicalTrials.gov.

## What it does

Given a patient summary (age, diagnosis, medical history), the system:
1. Retrieves the most relevant clinical trials from a 500k+ trial index
2. Reranks candidates using semantic similarity
3. Uses an LLM to assess each trial against patient criteria with cited evidence
4. Returns a structured match report explaining why each trial is or isn't suitable

## Architecture
```
ClinicalTrials.gov API → Parser → Chunker → ClinicalBERT Embeddings
                                                      ↓
Patient Summary → Query Embedding → Weaviate Hybrid Search → Cohere Reranker
                                                      ↓
                                          LLM Reasoning Layer (Gemini)
                                                      ↓
                                     Structured Match Report + Citations
```

## Tech stack

| Layer | Tool |
|-------|------|
| Embeddings | ClinicalBERT (domain-specific) |
| Vector DB | Weaviate (hybrid dense + BM25 search) |
| Reranking | Cohere Rerank |
| LLM | Google Gemini |
| Evaluation | RAGAS (faithfulness, context recall) |
| Backend | FastAPI |
| Frontend | Streamlit |

## Project structure
```
clinical-trial-rag/
├── data/
│   └── processed/        # Chunked trial JSON (gitignored)
├── src/
│   ├── ingest.py         # ClinicalTrials.gov → JSON chunks
│   ├── embed.py          # Embed + index into Weaviate
│   ├── retrieve.py       # Query pipeline + reranking
│   ├── llm.py            # LLM reasoning + explainability
│   └── app.py            # Streamlit UI
├── tests/
├── .env.example          # Environment variable template
└── requirements.txt
```

## Setup

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/clinical-trial-rag.git
cd clinical-trial-rag
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Fill in your keys in .env
```

### 3. Ingest data
```bash
python src/ingest.py
```

### 4. Run the app
```bash
streamlit run src/app.py
```

## Evaluation results
*RAGAS scores will be added after Phase 4*

## Author
Built as a portfolio project demonstrating production RAG patterns for healthcare.