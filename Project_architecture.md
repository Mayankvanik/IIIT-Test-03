# CUAD Contract Analysis Pipeline — Complete Technical Documentation

> **Scope:** This document explains every stage of the system end-to-end — from raw PDF files on disk to a streamed natural-language answer — covering the preprocessing design, chunking strategy, metadata mapping, hybrid vector search, and why LangGraph was chosen over simple RAG.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Stage 1 — PDF Discovery & Folder Mapping](#2-stage-1--pdf-discovery--folder-mapping)
3. [Stage 2 — Text Extraction & Cleaning](#3-stage-2--text-extraction--cleaning)
4. [Stage 3 — Hierarchical Chunking Strategy](#4-stage-3--hierarchical-chunking-strategy)
5. [Stage 4 — Metadata Schema & Payload Design](#5-stage-4--metadata-schema--payload-design)
6. [Stage 5 — Embedding Strategy (Dense + Sparse)](#6-stage-5--embedding-strategy-dense--sparse)
7. [Stage 6 — Qdrant Storage & Index Design](#7-stage-6--qdrant-storage--index-design)
8. [Stage 7 — Hybrid Retrieval with RRF Fusion](#8-stage-7--hybrid-retrieval-with-rrf-fusion)
9. [Why LangGraph Instead of Simple RAG](#9-why-langgraph-instead-of-simple-rag)
10. [LangGraph Node-by-Node Walkthrough](#10-langgraph-node-by-node-walkthrough)
11. [End-to-End Flow Summary](#11-end-to-end-flow-summary)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. System Overview

The system ingests 510 real commercial contracts from SEC EDGAR (the CUAD dataset), extracts 41 clause categories, and exposes a conversational Q&A interface. The pipeline has two distinct phases:

**Ingestion Phase** (runs once, offline)

```
PDFs on disk
  → PyMuPDF text extraction
    → Page cleaning
      → Hierarchical chunking (section-aware + sliding window)
        → OpenAI dense embedding (1536-dim)
          → BM25 sparse vectorisation
            → Qdrant upsert with rich metadata payload
```

**Retrieval Phase** (runs on every query, online)

```
User question
  → LangGraph router (classify intent)
    → Hybrid Qdrant search (dense + sparse → RRF fusion)
      → LLM reasoning node (extract / answer / compare / risk)
        → Grade node (hallucination check)
          → Response to user
```

The two phases are completely decoupled. Ingestion is a one-time batch job. Retrieval is a real-time graph execution.

---

## 2. Stage 1 — PDF Discovery & Folder Mapping

### The Folder Structure

The CUAD dataset stores contracts in a three-level folder hierarchy:

```
CUAD_v1/
  full_contract_pdf/
    Part_I/
      Affiliate_Agreements/      ← category
        ContractA.pdf
        ContractB.pdf
      License_Agreements/
        ContractC.pdf
    Part_II/
      Commercial Contracts (Part II-A)/
        Agency Agreements/
        Collaboration/
    Part_III/
      Joint Venture _ Filing/
      Manufacturing/
```

Each folder level carries semantic meaning:

| Level | Field stored | Meaning |
|---|---|---|
| `Part_I / Part_II / Part_III` | `part` | Dataset partition |
| `Affiliate_Agreements` etc. | `category` | Contract type / clause domain |
| `ContractA.pdf` (stem) | `contract_name` | Unique contract identifier |

### Why This Mapping Matters

At query time, a user can ask: *"Show me all non-compete clauses across affiliate agreements."* Because category is stored as an indexed payload field in Qdrant, this becomes a pre-filter on the vector search — you only search the ~80 affiliate agreement chunks instead of all 50,000+ chunks. This reduces latency and improves precision dramatically.

### Discovery Implementation

`Path.rglob("*.pdf")` recursively walks the entire tree. Each PDF is assigned a `contract_id` using `md5(file_path)` — a stable hash that does not change across re-ingestion runs, so re-ingesting one contract does not duplicate it in Qdrant (Qdrant's upsert is idempotent on the same ID).

---

## 3. Stage 2 — Text Extraction & Cleaning

### Tool: PyMuPDF (`fitz`)

PyMuPDF is the best-in-class Python PDF library for text extraction from SEC filings because:

- It respects the internal text-object ordering of the PDF (critical for multi-column layouts common in legal documents)
- It exposes page-level access, which is necessary to record `page_start` and `page_end` for every chunk
- It handles non-standard encodings common in scanned-then-OCR'd EDGAR filings

### Cleaning Steps Applied Per Page

```
Raw page text (from fitz)
  → Remove lone page-number lines  (regex: ^\s*[-–—]?\s*\d{1,3}\s*$)
  → Collapse 3+ consecutive newlines to double newline
  → Normalise intra-line whitespace (split on spaces, rejoin)
  → Discard pages shorter than MIN_CHUNK_CHARS (80 chars)
```

**Why page numbers are stripped explicitly:** SEC filings frequently centre-stamp page numbers on a line by themselves. If kept, they appear as orphaned tokens inside chunks and confuse BM25 scoring (the number "12" gets the same IDF treatment as a meaningful legal term).

**Why 80 chars minimum:** Pages that are purely decorative (exhibit cover pages, blank separators) produce very short text. Storing these wastes vector space and degrades retrieval precision because they match loosely against almost any query.

---

## 4. Stage 3 — Hierarchical Chunking Strategy

This is the most critical design decision in the entire pipeline. Poor chunking produces irretrievable answers regardless of how good the embedding model is.

### The Problem with Naive Chunking

A naive fixed-size chunker (e.g., split every 500 tokens) would routinely:

- Split a clause like *"Termination for Convenience"* across two chunks — putting the heading in one chunk and the actual terms in the next
- Merge unrelated sections (e.g., the end of the indemnification clause and the start of the governing law clause) into a single chunk, confusing the retriever
- Produce chunks with no semantic coherence, making the LLM's job harder

### The Two-Pass Approach

**Pass A — Section-aware splitting**

A regex detects natural section boundaries in legal documents:

```python
section_pattern = re.compile(
    r"(?m)^(?:"
    r"\d{1,2}(?:\.\d{1,2})*\s+[A-Z]"   # "1.2 SECTION TITLE"
    r"|[A-Z][A-Z\s]{5,50}$"              # "GOVERNING LAW"
    r"|(?:ARTICLE|SECTION)\s+[IVXLC\d]" # "ARTICLE IV"
    r")"
)
```

These patterns match the three most common section-heading formats in SEC commercial contracts:

- Numbered decimal headings: `4.3 Limitation of Liability`
- ALL-CAPS headings: `GOVERNING LAW`
- Article/Section headings: `ARTICLE IV — TERMINATION`

The boundaries from this regex are used to divide the full document text into **candidate sections** — each section is one logical unit of the contract.

**Pass A — Merging and flushing logic**

Once candidate sections are identified, a buffer-based pass merges or splits them:

```
If current section + buffer < TARGET_CHARS (2400):
    Merge into buffer
Else:
    Flush buffer as a chunk candidate
    Start new buffer with current section
```

This ensures:

- Tiny sections (e.g., a 3-line "Notices" clause) do not become standalone one-sentence chunks
- Very large sections (e.g., a 5-page schedule) do not become a single 5000-token chunk that exceeds the LLM context window

**Pass B — Sliding window on oversized candidates**

Any candidate that still exceeds TARGET_CHARS after merging is split with a sliding window:

```
Chunk 1:  chars [0     → 2400]
Chunk 2:  chars [1920  → 4320]   ← 480 char overlap with Chunk 1
Chunk 3:  chars [3840  → 6240]   ← 480 char overlap with Chunk 2
```

The 120-token (≈480 char) overlap ensures that a clause spanning a chunk boundary is fully represented in at least one of the two chunks. Without overlap, a retriever could fetch `Chunk 1` and miss the last sentence of a clause that continues in `Chunk 2`.

### Why This Is Better Than Recursive Character Splitting

LangChain's `RecursiveCharacterTextSplitter` is a good general tool, but it only looks at character separators (`\n\n`, `\n`, ` `) — it has no awareness of legal document structure. The hierarchical approach here:

- Preserves clause boundaries (a clause extracted from a single chunk is much more likely to be complete and coherent)
- Produces semantically meaningful chunks that the retriever can match against clause-type queries
- Reduces the number of chunks needed (fewer, denser chunks = faster retrieval)

For a typical 40-page contract, naive splitting at 500 tokens produces ~80 chunks. The hierarchical approach produces ~25–40 chunks, each corresponding to an actual contract section.

---

## 5. Stage 4 — Metadata Schema & Payload Design

Every chunk stored in Qdrant carries a rich payload. This payload serves two purposes: **post-retrieval context** (shown to the LLM and the user) and **pre-retrieval filtering** (applied before the vector search to narrow the candidate set).

### Full Payload Schema

```json
{
  "chunk_id":        "uuid-v4",
  "contract_id":     "md5-of-filepath",
  "contract_name":   "CreditcardscomInc_..._Affiliate Agreement",

  "part":            "Part_I",
  "category":        "Affiliate_Agreements",
  "file_path":       "/data/pdfs/CUAD_v1/.../Affiliate Agreement.pdf",

  "page_start":      3,
  "page_end":        4,
  "chunk_index":     3,
  "total_chunks":    15,
  "char_count":      2399,

  "text":            "... raw chunk text ...",

  "detected_clauses": ["Indemnification", "Termination for Convenience"],

  "has_governing_law":   false,
  "has_termination":     true,
  "has_cap_liability":   false,
  "has_non_compete":     false,
  "has_indemnification": true,
  "has_arbitration":     false,
  "has_ip_ownership":    false,
  "has_exclusivity":     false
}
```

### The Two-Tier Clause Tagging Design

**Tier 1 — Keyword detection at ingest time (fast, no LLM)**

During ingestion, each chunk is scanned for a curated list of legal keywords:

```python
CLAUSE_KEYWORDS = {
    "Indemnification": ["indemnif", "hold harmless", "defend and indemnify"],
    "Arbitration":     ["arbitration", "binding arbitration", "arbitrator"],
    ...
}
```

This populates `detected_clauses[]` and the `has_*` boolean fields. These are stored as indexed payload fields in Qdrant (using `PayloadSchemaType.BOOL`).

**Why this matters:** When a user asks *"Does this contract have a non-compete clause?"*, the retrieval node can pass `has_non_compete=True` as a payload pre-filter to Qdrant. The vector search then only runs over chunks that the lightweight keyword scanner already flagged — dramatically reducing the search space and improving precision.

**Tier 2 — LLM extraction at query time (accurate, with context)**

The keyword scanner produces boolean signals but no extracted text. When a user wants the actual clause content, the LangGraph `extract_node` calls GPT-4o with the retrieved chunks and asks it to:

- Confirm presence (true/false)
- Extract the verbatim clause text
- Identify the page and section number
- Explain what the clause means in plain English

This two-tier design avoids calling the LLM for every chunk at ingest time (which would cost ~$200+ for 510 contracts) while still delivering accurate, contextual extraction at query time.

### Payload Indexes (Required by Qdrant Strict Mode)

Qdrant's strict mode requires explicit indexes on any payload field used in filters. The following indexes are created once after ingestion:

| Field | Index Type | Used For |
|---|---|---|
| `contract_name` | KEYWORD | Scoping search to one contract |
| `category` | KEYWORD | Filtering by contract type |
| `part` | KEYWORD | Filtering by dataset partition |
| `has_*` (8 fields) | BOOL | Pre-filtering by clause presence |
| `page_start`, `page_end` | INTEGER | Page-range filtering |
| `chunk_index` | INTEGER | Ordered chunk retrieval |

---

## 6. Stage 5 — Embedding Strategy (Dense + Sparse)

### Dense Embeddings: OpenAI `text-embedding-3-small`

Each chunk's text is embedded as a 1536-dimensional float32 vector using OpenAI's `text-embedding-3-small` model. Dense embeddings capture **semantic meaning** — they encode what a passage is *about* rather than what words it contains.

This is essential for contract analysis because the same legal concept appears in radically different language across contracts:

| Contract A | Contract B | Dense embedding similarity |
|---|---|---|
| "Either party may terminate this Agreement at any time without cause upon 30 days written notice" | "This Agreement is terminable for convenience by either Party upon thirty (30) days prior written notice" | Very high (same concept) |

Keyword search would find neither if the user queries *"termination for convenience clause"* — because neither contains that exact phrase. Dense search finds both because the embeddings are close in vector space.

**Batching:** Embeddings are generated in batches of 16 chunks per OpenAI API call, with exponential back-off on rate limits. This keeps ingest cost low (~$2–5 for 510 contracts) while staying within API rate limits.

### Sparse Embeddings: Custom BM25

A from-scratch BM25 vectoriser is implemented without external dependencies. BM25 (Best Match 25) is a probabilistic ranking function that weighs terms by:

- **Term Frequency (TF):** How many times a term appears in a chunk, normalised by chunk length
- **Inverse Document Frequency (IDF):** How rare the term is across all chunks (rare terms get higher weight)

The formula for a term's score in a chunk:

```
score = IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × dl/avgdl))
```

Where:
- `k1 = 1.5` — term frequency saturation (diminishing returns for repeated terms)
- `b = 0.75` — length normalisation factor
- `dl` = document (chunk) length, `avgdl` = average chunk length across corpus

**Why BM25 complements dense embeddings:**

Dense embeddings are excellent at semantic similarity but can miss exact legal terminology. BM25 is excellent at exact term matching but misses paraphrases. Together in hybrid search, they cover both failure modes:

| Query | Dense wins | BM25 wins |
|---|---|---|
| "can either party walk away without cause" | ✓ (paraphrase) | ✗ |
| "liquidated damages clause" | ✗ (rare term) | ✓ (exact match) |
| "ROFR right of first refusal" | ✓ (semantic) | ✓ (exact) |

**Two-pass BM25 fitting:** BM25 requires knowing the full corpus statistics (N = total docs, df = document frequency per term, avgdl = average document length) before it can score any document. The pipeline therefore makes two passes:

- **Pass 1 (fit):** Iterate all chunks, build vocabulary and document frequency counts
- **Pass 2 (transform):** Convert each chunk's text to a sparse vector using the fitted statistics

The fitted vocabulary is saved to `bm25_vocab.json` so query-time BM25 scoring can reconstruct vectors without re-fitting.

---

## 7. Stage 6 — Qdrant Storage & Index Design

### Collection Configuration

```
dense-vector:   size=1536, distance=Cosine, HNSW m=24, ef_construct=256
sparse-vector:  on_disk=true, inverted index
```

The HNSW parameters `m=24` and `ef_construct=256` are higher than Qdrant's defaults (`m=16`, `ef_construct=100`). These settings trade indexing time for better recall:

- Higher `m` = more connections per node in the HNSW graph = better approximate nearest-neighbour accuracy
- Higher `ef_construct` = more candidates evaluated during index construction = higher quality graph

For legal documents where missing the most relevant clause has real consequences, higher recall is worth the extra indexing cost.

### Storage Layout: One Point Per Chunk

Each `PointStruct` contains:

```python
PointStruct(
    id      = chunk.chunk_id,           # uuid-v4 string
    vector  = {
        "dense-vector":  [float × 1536],
        "sparse-vector": SparseVector(indices=[...], values=[...]),
    },
    payload = { ...full metadata dict... },
)
```

Points are upserted in batches of 64. With ~25,000 chunks for 510 contracts (average 50 chunks per contract), the total collection size is manageable on the free Qdrant cloud tier.

---

## 8. Stage 7 — Hybrid Retrieval with RRF Fusion

### Architecture: Prefetch + Fuse

Qdrant's native hybrid search uses a two-stage architecture:

**Stage 1 — Parallel prefetch (two arms)**

```python
Prefetch(query=dense_vector,  using="dense-vector",  limit=20)
Prefetch(query=sparse_vector, using="sparse-vector", limit=20)
```

Both prefetches run simultaneously. Each returns its top-20 candidates independently. At this point there are up to 40 candidate chunks (some may appear in both lists).

**Stage 2 — Reciprocal Rank Fusion (RRF)**

RRF merges the two ranked lists using the formula:

```
RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)
```

Where `k = 60` (Qdrant's default) and `rank_in_list_i` is the chunk's rank in the dense or sparse results (1-indexed). A chunk that ranks 1st in the dense list and 3rd in the sparse list scores:

```
RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
```

A chunk that only appears in one list scores half as much. The final top-K chunks are selected from the RRF-ranked merged list.

**Why RRF instead of a weighted sum?**

A weighted sum (`α × dense_score + β × sparse_score`) requires careful tuning of α and β, which are corpus-dependent and can drift as new contracts are added. RRF is parameter-free (except for `k=60` which is robust across domains) and has been shown empirically to match or outperform weighted fusion in most retrieval benchmarks.

### Single-Contract vs Cross-Contract Search

**Single-contract search** (`retrieve_for_contract`): Adds a `FieldCondition` filter on `contract_name` to the Qdrant query. This pre-filters the candidate set to only that contract's chunks before the HNSW traversal, making it both faster and more precise.

**Cross-contract search** (`retrieve_cross_contract`): Runs one independent hybrid search per contract, with per-contract filters, then returns a dictionary keyed by contract name. This is used by the compare and risk nodes which need clause text from multiple contracts simultaneously.

---

## 9. Why LangGraph Instead of Simple RAG

### What Simple RAG Looks Like

A simple RAG pipeline has three steps:

```
User query → Vector search → LLM (query + chunks) → Answer
```

This works well for single-turn factual Q&A over a single document. It fails for this project because the CUAD use case requires several capabilities that simple RAG cannot provide.

### Problem 1 — Multiple Distinct Task Types

A user might ask any of these:

- *"Does this contract have a non-compete clause?"* → clause extraction
- *"What is the governing law?"* → Q&A
- *"Compare termination clauses across 5 contracts"* → cross-contract comparison
- *"Flag all contracts with no liability cap"* → risk analysis

Each task requires a fundamentally different retrieval strategy, a different prompt, and a different output format. Simple RAG has no concept of routing — it would send all four questions through the same retrieve-then-answer pipeline, producing poor results for three of them.

**LangGraph solution:** The `router_node` classifies intent into one of four categories. The graph then routes to the appropriate retrieval strategy and reasoning node.

### Problem 2 — Multi-Step Clause Extraction

Extracting 10 clause types from a single contract is not a one-shot operation. It requires:

1. A targeted hybrid search for each clause type (10 searches, each with different query text)
2. For each clause: a separate LLM call to extract verbatim text, locate the page, and explain the clause
3. Aggregation of 10 results into a structured report

Simple RAG retrieves once and answers once. This cannot be expressed as a single retrieve-then-answer step.

**LangGraph solution:** The `extract_node` loops over target clause types, running one hybrid search and one LLM call per clause, then aggregates into `clause_results`.

### Problem 3 — Grounding and Hallucination Control

Legal contracts are a domain where hallucination is catastrophic. A RAG pipeline that says *"yes, this contract has a liability cap of $500,000"* when the contract actually has no cap could cause real financial harm.

Simple RAG has no self-correction mechanism. It generates one answer and returns it.

**LangGraph solution:** The `grade_node` independently evaluates the draft answer on two axes:

- **Groundedness:** Is every factual claim traceable to the retrieved chunks?
- **Relevance:** Does the answer actually address the question?

If either score is below 0.6, the graph loops back to `retrieve_node` with an incremented retry counter (max 2 retries). This implements a self-correcting feedback loop that is structurally impossible in simple RAG.

### Problem 4 — Conversation History

A user conducting due diligence on a set of contracts will ask follow-up questions:

- *"What is the governing law of this contract?"*
- *"And what about the termination clause?"* ← requires knowing which contract "this" refers to
- *"Compare that to the affiliate agreement we looked at earlier"* ← requires remembering two contracts

Simple RAG is stateless — each query is independent. Without conversation history, follow-up questions cannot be resolved.

**LangGraph solution:** The `State` TypedDict accumulates `messages` (a list of `HumanMessage` / `AIMessage` pairs) across turns using LangGraph's `add_messages` reducer. The `answer_node` passes the last 4 turns as context to the LLM, enabling coherent multi-turn dialogue.

### Problem 5 — Cross-Contract Parallel Operations

Comparing termination clauses across 5 contracts requires:

1. Retrieving relevant chunks from each of the 5 contracts independently
2. Extracting the termination clause from each
3. Sending all 5 extracted clauses to the LLM for a comparative analysis

Simple RAG retrieves from a single context and cannot express "run retrieval per contract, then aggregate."

**LangGraph solution:** `retrieve_cross_contract()` runs one hybrid search per contract. `compare_risk_node` extracts clauses per contract, builds a comparison matrix, and applies risk rules. These are separate nodes connected by explicit graph edges.

### Decision Summary

| Requirement | Simple RAG | LangGraph |
|---|---|---|
| Intent classification | ✗ | ✓ router node |
| Multi-step clause extraction | ✗ | ✓ extract node loop |
| Hallucination self-correction | ✗ | ✓ grade → retry loop |
| Conversation memory | ✗ | ✓ state accumulation |
| Cross-contract comparison | ✗ | ✓ parallel retrieve + compare node |
| Risk flagging with rules | ✗ | ✓ compare_risk node |

---

## 10. LangGraph Node-by-Node Walkthrough

### Node 1 — `router_node`

**Input:** Latest user message  
**Output:** `intent`, `contract_name`, `contract_names`, `clause_type`

Calls GPT-4o-mini with a structured prompt asking it to classify the message into one of four intents and extract entity hints. Returns JSON. On parse failure, defaults to `intent="qa"` (safe fallback).

**Why a separate router?** Keeping routing logic in a dedicated node means it can be updated independently of the retrieval or answer logic. It also makes the graph's decision points explicit and inspectable.

### Node 2 — `retrieve_node`

**Input:** `intent`, `contract_name`, `contract_names`, `clause_type`  
**Output:** `retrieved_chunks` or `cross_contract_chunks`

Selects the retrieval mode based on intent:

- `qa` or `extract` → `retrieve_for_contract()` with optional boolean pre-filter
- `compare` or `risk` → `retrieve_cross_contract()` across all named contracts

If `clause_type` is set, the query is augmented: `f"{clause_type} clause: {user_query}"`. This gives BM25 a strong lexical signal even when the user's phrasing is informal.

### Node 3a — `extract_node`

**Input:** `retrieved_chunks`, `contract_name`, `clause_type`  
**Output:** `clause_results`

Iterates over target clauses. For each clause, runs a fresh targeted hybrid search (the initial retrieve may have missed some clauses). Calls GPT-4o with a structured prompt that requests JSON output with four fields: `present`, `extracted_text`, `page`, `explanation`.

### Node 3b — `compare_risk_node`

**Input:** `cross_contract_chunks`, `clause_type`, `intent`  
**Output:** `comparison_table`, `risk_report`

For each contract in `cross_contract_chunks`, calls GPT-4o to extract a set of target clauses. Then:

- Builds a Markdown comparison table (contracts as rows, clause types as columns)
- If `intent == "risk"`, applies rule-based risk scoring: each missing HIGH_RISK clause adds 3 to the risk score; each missing MEDIUM_RISK clause adds 1. Score ≥ 3 = HIGH, ≥ 1 = MEDIUM, 0 = LOW.

### Node 4 — `answer_node`

**Input:** Depends on intent  
**Output:** `draft_answer`

Routes to three sub-paths:

- `extract` → formats `clause_results` as a structured Markdown report with extracted text and explanations
- `compare` → wraps `comparison_table` with a header
- `risk` → formats `risk_report` with colour-coded risk levels
- `qa` → calls GPT-4o with retrieved chunks + conversation history to generate a grounded answer

### Node 5 — `grade_node`

**Input:** `draft_answer`, `retrieved_chunks`, user question  
**Output:** `grade_score` (0–1)

Calls GPT-4o-mini with the draft answer, the source chunks, and the original question. Asks for JSON with `groundedness` and `relevance` scores. Returns the average of the two.

Skipped entirely for `extract`, `compare`, and `risk` intents because those produce structured outputs from the chunks directly — there is nothing to grade for hallucination.

### Node 6 — `respond_node`

**Input:** `draft_answer`  
**Output:** `final_answer`, appended `AIMessage` to conversation history

Light post-processing (strip, trim) and appends the final answer to the `messages` list so it becomes part of the conversation context for the next turn.

---

## 11. End-to-End Flow Summary

### Example: *"Does the CreditcardscomInc affiliate agreement have a non-compete clause?"*

```
router_node
  intent      = "extract"
  contract    = "CreditcardscomInc_..._Affiliate Agreement"
  clause_type = "Non-Compete"

retrieve_node
  query       = "Non-Compete clause: Does the CreditcardscomInc affiliate agreement have a non-compete clause?"
  filter      = contract_name = "CreditcardscomInc_..." AND has_non_compete = true
  → 6 chunks returned (pages 2–5)

extract_node
  clause      = "Non-Compete"
  targeted    = 5 chunks from hybrid search
  LLM call    → {present: true, text: "Affiliates will not use the following product keyword...",
                  page: 3, section: "5", explanation: "..."}

answer_node
  → Formats extraction report

grade_node
  → Skipped (intent = extract)

respond_node
  → "### Non-Compete\n**Present:** Yes | **Page:** 3 | **Section:** 5\n..."
```

---

## 12. Design Decisions & Trade-offs

### Why `text-embedding-3-small` over `text-embedding-3-large`?

`text-embedding-3-large` produces 3072-dim vectors and is marginally more accurate, but costs 5× more and requires 2× storage. For legal clause retrieval, `text-embedding-3-small` at 1536 dimensions is sufficiently accurate because the retrieval task is relatively coarse (find the relevant section, not find the exact sentence). The LLM does the fine-grained extraction after retrieval.

### Why not store full contract text and chunk at query time?

Some RAG systems store the full document and chunk dynamically at query time. This is simpler to implement but slower at query time (every search requires re-chunking and re-embedding the document) and more expensive (you pay for embedding on every query, not just once at ingest).

### Why GPT-4o for extraction/comparison but GPT-4o-mini for routing/grading?

Routing and grading are classification tasks — they require structured JSON output and a binary decision. GPT-4o-mini is fast and cheap for these. Clause extraction and cross-contract comparison require deep legal reasoning and precise text localisation — GPT-4o is significantly more accurate for these tasks. Using different models per node minimises cost while maximising quality where it matters.

### Why BM25 from scratch instead of a library?

Libraries like `rank_bm25` or `BM25Okapi` build in-memory indexes but cannot serialize their vocabulary to a format Qdrant can ingest as sparse vectors. Building BM25 from scratch gives full control over the vocabulary serialisation (saved as `bm25_vocab.json`) and the sparse vector format required by Qdrant's `SparseVector(indices=[...], values=[...])` API.

### Why RRF over a weighted linear combination?

A weighted combination (`α × dense_score + (1-α) × sparse_score`) requires tuning α. The optimal α changes with the query type (clause extraction benefits from higher BM25 weight; semantic Q&A benefits from higher dense weight). RRF is query-type agnostic — it ranks by position in each list rather than absolute score, so it handles both query types without tuning.
