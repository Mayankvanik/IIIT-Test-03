# CUAD Contract Analysis Chatbot

> **⚠️ Before reading this file, please read [`Project_architecture.md`](./pipeline_documentation.md) first
> for a deep-dive into the project architecture, ingestion pipeline, retrieval design, and LangGraph flow.**

---

## What Is This?

A **RAG (Retrieval-Augmented Generation) chatbot** built on top of the [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad) — 510 real commercial contracts annotated with 41 legal clause categories.

The chatbot can:
- **Answer questions** about any contract (e.g. *"What can Chase NOT do with the subaffiliate list?"*)
- **Extract clauses** (e.g. *"Extract the indemnification clause from Agreement X"*)
- **Compare clauses** across multiple contracts
- **Risk-flag** contracts with missing critical protections

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py  (Gradio UI)                    │
│   Tab 1: Config → saves credentials to .env                 │
│   Tab 2: Data Ingestion → runs ingest pipeline              │
│   Tab 3: Chat → opens chatbot terminal                      │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
   LOCAL mode                CLOUD mode
   ─────────────────          ─────────────────
   Embedding: mixedbread-ai   Embedding: OpenAI text-embedding-3-small
   VectorDB:  Docker Qdrant   VectorDB:  Qdrant Cloud
   Ingest:    ingest-local.py Ingest:    ingest.py
   Retrieval: retrieval-local.py  Retrieval: retrieval.py
   Chatbot:   chatbot-local.py    Chatbot:   chatbot.py
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   LangGraph Pipeline  │
         │  Router → Retrieve →  │
         │  Extract/Compare/Risk │
         │  → Answer → Grade →   │
         │  Respond              │
         └───────────────────────┘
```

### Hybrid Retrieval (both modes)
| Signal | Method | Purpose |
|--------|--------|---------|
| **Semantic** | Dense vector (cosine) | Understands meaning — finds "termination at will" even if user says "exit clause" |
| **Keyword** | Sparse BM25 | Exact legal terms — "indemnification", party names, jurisdiction |
| **Fusion** | Qdrant RRF | Reciprocal Rank Fusion merges both ranked lists for best results |
| **Metadata** | Payload filters | Pre-filters by `has_indemnification`, `has_arbitration`, etc. before vector search |

---

## Project Structure

```
practical-Task-03/
│
├── main.py                        # 🚀 Entry point — Gradio UI
│
├── data_preprocess/
│   ├── hugging-datasets.py        # Step 1 — Download CUAD dataset metadata from HuggingFace
│   ├── download_pdf_list.py       # Step 2 — Download actual contract PDFs
│   ├── ingest-local.py            # Step 3a — LOCAL ingestion pipeline (mxbai + Docker Qdrant)
│   └── ingest.py                  # Step 3b — CLOUD ingestion pipeline (OpenAI + Qdrant Cloud)
│
├── retrieval-local.py             # Hybrid retrieval — LOCAL (mxbai BM25 + Qdrant Docker)
├── retrieval.py                   # Hybrid retrieval — CLOUD (OpenAI BM25 + Qdrant Cloud)
│
├── chatbot-local.py               # LangGraph chatbot — LOCAL retrieval + OpenAI LLM
├── chatbot.py                     # LangGraph chatbot — CLOUD retrieval + OpenAI LLM
│
├── requirements.txt               # Python dependencies
├── .env                           # Credentials (git-ignored)
├── pipeline_documentation.md      # 📖 Detailed architecture documentation
└── README.md                      # This file
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for local mode only)
- OpenAI API key (for the LLM reasoning — required in both modes)
- Qdrant Cloud account (for cloud mode only)

---

### 1. Clone & Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download the CUAD Dataset

**Step A — Download dataset metadata from HuggingFace:**
```bash
python data_preprocess/hugging-datasets.py
```
This downloads the `theatticusproject/cuad` dataset index and saves it to `data_preprocess/data/dataset/`.

**Step B — Download the contract PDFs:**
```bash
python data_preprocess/download_pdf_list.py
```
This reads the dataset info and downloads all PDF contracts to `data_preprocess/data/pdfs/CUAD_v1/`.

> ℹ️ The full CUAD dataset contains **510 contracts**. Download may take 10–30 minutes depending on your connection.

---

### 4. (Local Mode Only) Start Qdrant via Docker

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify it's running: open [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### 5. Launch the Application

```bash
python main.py
```

This opens the **Gradio UI** at [http://127.0.0.1:7860](http://127.0.0.1:7860) automatically.

---

## Using the Application

### Tab 1 — Configuration ⚙️

Choose your environment and fill in the credentials:

| Field | Local Mode | Cloud Mode |
|-------|-----------|------------|
| `LOCAL_QDRANT_URL` | `http://localhost:6333` | _(not needed)_ |
| `LOCAL_QDRANT_API_KEY` | _(leave empty)_ | _(not needed)_ |
| `LOCAL_COLLECTION` | `pdfs-store` | _(not needed)_ |
| `QDRANT_URL` | _(not needed)_ | Your Qdrant Cloud URL |
| `QDRANT_API_KEY` | _(not needed)_ | Your Qdrant Cloud API key |
| `COLLECTION` | _(not needed)_ | `pdfs-store` |
| `OPENAI_API_KEY` | Required (for LLM) | Required (for LLM + embeddings) |

Click **"Save to .env"** — credentials are written to `.env` and the `user_type` flag is set.

> **Why local embedding?** OpenAI's embedding API has rate/cost limits. For local mode, we use `mixedbread-ai/mxbai-embed-large-v1` running on CUDA/CPU — no API calls needed for retrieval.

---

### Tab 2 — Data Ingestion 📥

Click **"Data Ingestion"** — this opens a new terminal window and runs:
- **Local:** `data_preprocess/ingest-local.py` — extracts PDFs, generates dense (mxbai) + sparse (BM25) vectors, stores in Docker Qdrant
- **Cloud:** `data_preprocess/ingest.py` — same pipeline using OpenAI embeddings, stores in Qdrant Cloud

The ingestion pipeline:
1. Discovers all PDFs recursively
2. Extracts text with PyMuPDF
3. Applies hierarchical chunking (section-aware + sliding window)
4. Generates dense embeddings (batched, GPU-accelerated if CUDA available)
5. Builds BM25 vocabulary and sparse vectors over the full corpus
6. Upserts chunks with rich metadata to Qdrant

> ⏱️ For 510 contracts on GPU: ~15–30 min. On CPU: ~2–4 hours.

---

### Tab 3 — Chat (Q&A) 💬

Click **"Chat (Q&A)"** — opens the chatbot in a new terminal window.

**Example questions:**
```
What can Chase NOT do with the subaffiliate list?
Extract the indemnification clause from BIRCH_COMMUNICATIONS_10-K
Compare the governing law clauses in Contract_A and Contract_B
Run a risk analysis on AMAZON_AFFILIATE_AGREEMENT
What is the termination notice period in [contract name]?
```

The chatbot uses a **6-node LangGraph pipeline**:

```
Router → Retrieve → [Extract | Compare/Risk | Answer] → Grade → Respond
```

- **Router** — classifies intent (qa / extract / compare / risk) and extracts entity names
- **Retrieve** — hybrid Qdrant search (dense + BM25 + RRF fusion)
- **Extract/Compare/Risk** — specialized LLM nodes for structured outputs
- **Grade** — scores answer on groundedness + relevance (retries if score < 0.6)
- **Respond** — finalizes and cites sources

---

## A Note on Code Structure

> Due to time constraints, **separate files were created** for local and cloud modes (`chatbot-local.py` / `chatbot.py`, `retrieval-local.py` / `retrieval.py`, `ingest-local.py` / `ingest.py`).
>
> In a production setting, this would follow a **"Don't Repeat Yourself" (DRY)** pattern — a single codebase with a config-driven backend that switches between local/cloud providers via dependency injection or an abstract base class.

---

## Environment Variables Reference

```env
# Mode selector (set automatically by main.py UI)
user_type=local          # or "cloud"

# Local mode
LOCAL_QDRANT_URL=http://localhost:6333
LOCAL_QDRANT_API_KEY=    # empty for Docker
LOCAL_COLLECTION=pdfs-store

# Cloud mode
QDRANT_URL=https://xxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION=pdfs-store

# LLM (required in both modes)
OPENAI_API_KEY=sk-...
```

---

## Thank You

Thank you for your time reviewing this project. The full technical breakdown of the pipeline, retrieval strategy, and LangGraph architecture is in [`Project_architecture.md`](./Project_architecture.md).
