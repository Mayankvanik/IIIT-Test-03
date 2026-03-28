text

# CUAD Contract Analysis Pipeline — Deep Technical Explanation

## Table of Contents

1. [Overview](#overview)
2. [Phase 1 — PDF Discovery & Folder Mapping](#phase-1--pdf-discovery--folder-mapping)
3. [Phase 2 — Text Extraction & Cleaning](#phase-2--text-extraction--cleaning)
4. [Phase 3 — Hierarchical Chunking Strategy](#phase-3--hierarchical-chunking-strategy)
5. [Phase 4 — Clause Detection & Metadata Tagging](#phase-4--clause-detection--metadata-tagging)
6. [Phase 5 — Embedding Generation](#phase-5--embedding-generation)
7. [Phase 6 — BM25 Sparse Vectorisation](#phase-6--bm25-sparse-vectorisation)
8. [Phase 7 — Qdrant Storage & Payload Schema](#phase-7--qdrant-storage--payload-schema)
9. [Phase 8 — Hybrid Retrieval with RRF Fusion](#phase-8--hybrid-retrieval-with-rrf-fusion)
10. [Phase 9 — LangGraph Multi-Node Chatbot](#phase-9--langgraph-multi-node-chatbot)
11. [Why LangGraph over Simple RAG](#why-langgraph-over-simple-rag)
12. [End-to-End Data Flow Diagram](#end-to-end-data-flow-diagram)

---

## Overview

The CUAD pipeline processes 510 real-world commercial contracts (SEC EDGAR filings) and
makes them queryable through natural language. The core challenge is not reading — it is
**finding**: locating the one clause in a 40-page agreement that exposes a party to uncapped
liability. Standard keyword search fails because the same concept (e.g., indemnification,
termination for convenience) is expressed in wildly different language across contracts.

The pipeline is built in two stages:

| Stage | What happens | Files involved |
|---|---|---|
| Ingestion | PDF → text → chunks → embeddings → Qdrant | `ingest_pipeline.py` |
| Retrieval + QA | Query → hybrid search → LangGraph → answer | `retrieval.py`, `chatbot.py` |

---

## Phase 1 — PDF Discovery & Folder Mapping

### The Folder Structure

The CUAD dataset is organised into three parts, each subdivided by contract category:

```
CUAD_v1/full_contract_pdf/
├── Part_I/
│   ├── Affiliate_Agreements/
│   ├── License_Agreements/
│   ├── Joint_Venture/
│   └── ... (20+ categories)
├── Part_II/ (Commercial Contracts Part II-A)
│   ├── Agency Agreements/
│   ├── Franchise/
│   └── ...
└── Part_III/
    ├── Strategic Alliance/
    ├── Manufacturing/
    └── ...
```

### How Discovery Works

`discover_pdfs()` walks the entire directory tree using `Path.rglob("*.pdf")` and extracts
structural metadata from the path parts:

```
path parts[0] = "CUAD_v1"
path parts[1] = "full_contract_pdf"
path parts[2] = "Part_I"           → stored as `part`
path parts[3] = "Affiliate_Agreements"  → stored as `category`
path stem     = "CreditcardscomInc_20070810..."  → stored as `contract_name`
```

### Why This Metadata Matters

Storing `part` and `category` in the Qdrant payload allows payload-level filtering at query
time. For example:

- "Find all non-compete clauses in **franchise** agreements only" — filter `category = Franchise`
- "Compare termination clauses in **Part_I** agreements" — filter `part = Part_I`

This avoids scanning all 510 contracts for every query — instead Qdrant pre-filters to the
relevant subset before running vector search.

### Contract Identity — Stable Hash ID

Each contract gets a stable `contract_id` derived from an MD5 hash of its file path:

```python
contract_id = hashlib.md5(record["path"].encode()).hexdigest()
```

This is stable across re-runs. If you re-ingest the same PDF (e.g. after fixing a bug), the
same `contract_id` is produced, making it easy to do targeted upserts without duplicating
points in Qdrant.

---

## Phase 2 — Text Extraction & Cleaning

### Tool: PyMuPDF (fitz)

PyMuPDF is used for text extraction because it:

- Preserves the **reading order** of text on the page (unlike pdfplumber which reads in
  bounding-box order)
- Handles **multi-column layouts** common in SEC filings
- Is significantly faster than pdfminer for bulk processing
- Exposes page-level access so we can track exactly which page each chunk came from

Extraction is done page-by-page:

```python
doc = fitz.open(pdf_path)
for page_num, page in enumerate(doc, start=1):
    raw = page.get_text("text")
    cleaned = _clean_page_text(raw, page_num)
```

### Cleaning Rules

Raw PDF text has three categories of noise that hurt retrieval quality:

**1. Lone page number lines**

SEC filings often embed page numbers as standalone lines (`- 12 -` or just `12`). These
pollute chunks with meaningless tokens that inflate BM25 scores for number-heavy queries.

```python
re.sub(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$", "", text, flags=re.MULTILINE)
```

**2. Excessive blank lines**

Multiple consecutive newlines are collapsed to double newlines. This preserves paragraph
separation (which the chunker uses as a weak boundary signal) without wasting character budget.

**3. Intra-line whitespace normalisation**

Each line is split on whitespace and rejoined with single spaces. This handles cases where
the PDF renderer inserted extra spaces between characters (common in older SEC filings scanned
from paper).

### Page Offset Tracking

After cleaning, a `page_offsets` list is built:

```python
page_offsets: list[tuple[int, int]] = []  # (char_start_in_full_text, page_num)
```

This allows any character position in the concatenated full-document string to be mapped back
to its original page number, which is stored in the chunk's `page_start` and `page_end` fields.
This is what enables exact source citations like "Page 7–8" in answers.

---

## Phase 3 — Hierarchical Chunking Strategy

This is the most important part of the entire pipeline. Poor chunking is the single biggest
cause of retrieval failure in RAG systems applied to long legal documents.

### Why Simple Fixed-Size Chunking Fails on Legal Documents

A fixed 512-token sliding window would frequently:

- **Split a clause in half** — the indemnification obligation starts at token 500 of one chunk
  and the cap on liability appears at token 50 of the next. Neither chunk contains the complete
  clause.
- **Mix unrelated clauses** — two unrelated clauses that happen to appear near each other on
  the same page end up in the same chunk, reducing retrieval precision.
- **Fragment the governing law clause** — often a single sentence, a fixed-size chunker may
  pad it with 400 tokens of boilerplate from the previous section.

### The Two-Pass Approach

#### Pass A — Section-Aware Splitting

The chunker first finds **natural section boundaries** using a regex that detects:

```python
section_pattern = re.compile(
    r"(?m)^(?:"
    r"\d{1,2}(?:\.\d{1,2})*\s+[A-Z]"   # "1.2 SECTION TITLE"
    r"|[A-Z][A-Z\s]{5,50}$"              # "GOVERNING LAW"
    r"|(?:ARTICLE|SECTION)\s+[IVXLC\d]" # "ARTICLE IV"
    r")"
)
```

This detects:
- Numbered headings: `1.`, `1.2`, `4.3.1`
- ALL-CAPS section titles: `GOVERNING LAW`, `TERM AND TERMINATION`
- Explicit article/section markers: `ARTICLE IV`, `SECTION 7`

The text is split at these boundaries, producing a list of **candidate sections**.

#### Merging Tiny Sections

Many contracts have very short sections (e.g. a one-sentence `NOTICES` clause). Merging
adjacent tiny sections prevents creating chunks that are too small to contain useful semantic
content for retrieval.

The algorithm:

```
buffer = []
for each candidate section:
    if buffer + section <= CHUNK_SIZE (600 tokens ≈ 2400 chars):
        add section to buffer
    else:
        flush buffer as one chunk
        start new buffer with this section
```

This means semantically related short clauses that appear together (e.g. `NOTICES` followed
by `GOVERNING LAW`) are kept together if they fit.

#### Pass B — Sliding Window on Large Sections

Some sections are very long (e.g. a detailed IP ownership clause spanning 3 pages). These
are further split using a sliding window with overlap:

```
target_chars  = 600 tokens × 4 chars/token = 2400 chars
overlap_chars = 120 tokens × 4 chars/token = 480 chars

Window 1: chars 0    → 2400
Window 2: chars 1920 → 4320  (480 chars overlap with window 1)
Window 3: chars 3840 → 6240  (480 chars overlap with window 2)
```

The 120-token (≈480-char) overlap ensures that a clause boundary that falls near the end of
one chunk is also present at the beginning of the next. This is critical for retrieval
because the LLM will always see a complete clause even if the chunk boundary falls mid-section.

### Chunk Size Choice (600 tokens)

| Size | Too small problem | Too large problem |
|---|---|---|
| 256 tokens | Splits clauses, loses context | — |
| 512 tokens | Marginal | Some clause mixing |
| **600 tokens** | **Balances completeness vs precision** | — |
| 1024 tokens | — | Multiple clauses mixed, dilutes retrieval signal |
| 2048 tokens | — | Exceeds useful context, slow embedding |

600 tokens was chosen because the average individual clause in CUAD (based on annotation
analysis) is 200–500 tokens. At 600 tokens with 120-token overlap, a retrieval query for a
single clause will almost always land in a chunk that contains the entire clause.

### Chunk Metadata

Each `Chunk` object stores:

| Field | Purpose |
|---|---|
| `chunk_id` | UUID — unique identifier for this chunk in Qdrant |
| `contract_id` | MD5 of file path — groups all chunks from one contract |
| `contract_name` | Human-readable name — used for display and filtering |
| `part` | Part_I / Part_II / Part_III — structural filter |
| `category` | Subfolder name — subject-matter filter |
| `file_path` | Absolute path — for re-reading if needed |
| `page_start` | First page this chunk covers — for citations |
| `page_end` | Last page this chunk covers — for citations |
| `chunk_index` | Position in the document (0-based) — for ordering |
| `total_chunks` | Total chunks in this contract — for context |
| `char_count` | Length in characters — for debugging |
| `detected_clauses` | List of clause types detected by keyword scan |

---

## Phase 4 — Clause Detection & Metadata Tagging

### Lightweight Keyword Pre-Tagging

At ingest time, every chunk is scanned against a keyword map for 12 high-priority clause types:

```python
CLAUSE_KEYWORDS = {
    "Governing Law":               ["governing law", "governed by", "jurisdiction"],
    "Termination for Convenience": ["terminate for convenience", "without cause"],
    "Cap on Liability":            ["cap on liability", "shall not exceed", "aggregate liability"],
    "Non-Compete":                 ["non-compete", "not compete", "competing business"],
    "Indemnification":             ["indemnif", "hold harmless", "defend and indemnify"],
    "Audit Rights":                ["audit right", "right to audit", "inspect records"],
    "Arbitration":                 ["arbitration", "binding arbitration"],
    "Renewal Term":                ["automatically renew", "auto-renew", "evergreen"],
    "Exclusivity":                 ["exclusive", "exclusively", "sole and exclusive"],
    ...
}
```

This produces two things stored in the Qdrant payload:

**1. `detected_clauses` array** — list of clause types found in this chunk:

```json
"detected_clauses": ["Governing Law", "Arbitration"]
```

**2. Boolean flags** — one per clause type for fast payload filtering:

```json
"has_governing_law": true,
"has_arbitration": true,
"has_non_compete": false,
...
```

### Why Boolean Flags Instead of Just the Array?

Qdrant's strict mode requires a payload index on any field used for filtering. Arrays require
a `KEYWORD` index with `MATCH_ANY` semantics. Boolean fields require a `BOOL` index which is
faster and uses less memory. For the most common queries ("find all chunks with non-compete
clauses"), a boolean filter scan is more efficient than scanning a `KEYWORD` array field.

### Limitations & Why This Is Intentional

Keyword tagging produces false negatives (misses clauses expressed in unusual language) and
false positives (triggers on the word "arbitration" even in a sentence that says "this
agreement does NOT require arbitration").

This is acceptable because:

1. It runs at **ingest time** — zero additional latency at query time
2. It is used only as a **pre-filter hint**, not as the final answer
3. The LLM in the extract node makes the actual clause determination
4. False negatives just mean the pre-filter is skipped and the full hybrid search runs instead

---

## Phase 5 — Embedding Generation

### Model: text-embedding-3-small (OpenAI)

This model produces 1536-dimensional dense vectors with cosine similarity. It was chosen over
`text-embedding-ada-002` (also 1536-dim) because:

- Better out-of-box performance on legal/technical text
- Same price per token
- Slightly lower latency on batch requests

### Batching Strategy

OpenAI's embedding API has rate limits (tokens per minute). Chunks are embedded in batches of
16 to stay within rate limits while minimising the number of API round-trips:

```python
EMBED_BATCH = 16

for i in range(0, len(texts), EMBED_BATCH):
    batch = texts[i: i + EMBED_BATCH]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
```

Exponential backoff (1s, 2s, 4s, 8s, 16s) is applied on `RateLimitError`.

### What Dense Embeddings Capture

Dense embeddings capture **semantic meaning** — the vector for "the agreement shall be
governed by the laws of the State of Delaware" is close to the vector for "Delaware law
applies to this contract" even though they share no keywords.

This is essential for legal documents because:

- The same clause is drafted differently by every law firm
- Synonyms are common: "terminate" / "cancel" / "rescind" / "dissolve"
- Legal jargon and plain-language equivalents must be matched

---

## Phase 6 — BM25 Sparse Vectorisation

### Why BM25 in Addition to Dense?

Dense embeddings are great at semantic similarity but can miss exact term matching. Legal
documents contain specific terms that must be matched exactly:

- "ROFR" (Right of First Refusal) — a dense model may not know this acronym
- "liquidated damages" — a specific legal concept with precise meaning
- "non-solicitation" — the exact term matters for clause identification
- Party names, dates, specific dollar amounts

BM25 (Best Match 25) is a classic information retrieval scoring function that excels at
exact and near-exact term matching. Combining both gives the best of both worlds.

### Two-Pass Architecture

BM25 requires knowing the entire corpus before scoring any single document (because IDF
— Inverse Document Frequency — depends on how many documents contain each term).

This forces a two-pass architecture:

```
Pass 1 (fit):   Process all chunks → build vocabulary + document frequency table
Pass 2 (transform): Score each chunk against the vocabulary
```

In code:

```python
bm25 = BM25Vectoriser()
bm25.fit([c.text for c in all_chunks])      # Pass 1: builds df, N, avgdl
all_sparse = [bm25.transform(c.text) for c in all_chunks]  # Pass 2: score each chunk
```

### BM25 Formula

For a token `t` in document `d`:

```
TF(t, d) = freq(t, d) × (k1 + 1) / (freq(t, d) + k1 × (1 - b + b × |d|/avgdl))
IDF(t)   = log((N - df(t) + 0.5) / (df(t) + 0.5))
BM25(t, d) = IDF(t) × TF(t, d)
```

Where:
- `k1 = 1.5` — term frequency saturation (diminishing returns for repeated terms)
- `b = 0.75` — document length normalisation (penalises long documents slightly)
- `N` — total number of chunks in corpus
- `df(t)` — number of chunks containing token `t`
- `avgdl` — average chunk length in tokens

### Sparse Vector Format

The BM25 score for each token is stored as a sparse vector: only non-zero token weights are
stored. For a chunk mentioning "arbitration" twice:

```json
{
  "indices": [4821, 892, 1203, ...],
  "values":  [3.54, 1.23, 0.87, ...]
}
```

Index 4821 is the vocabulary index for "arbitration", value 3.54 is its BM25 weight.
Most of the 50,000+ vocabulary tokens have weight 0 and are not stored.

### Vocabulary Persistence

The vocabulary (token → index mapping) and document frequency table are saved to
`bm25_vocab.json` after ingestion:

```json
{
  "vocab":  {"arbitration": 4821, "governing": 892, ...},
  "df":     {"4821": 43, "892": 128, ...},
  "N":      7823,
  "avgdl":  142.3
}
```

This is loaded at query time to convert the user's query into a sparse vector using the
same vocabulary. Without this, the query vector and document vectors would use different
index spaces and dot products would be meaningless.

---

## Phase 7 — Qdrant Storage & Payload Schema

### Collection Configuration

```
Collection: "pdfs-store"
├── dense-vector  : 1536 dimensions, Cosine distance, HNSW index
└── sparse-vector : Sparse, on-disk index
```

HNSW (Hierarchical Navigable Small World) is the approximate nearest-neighbour index used
for the dense vector. The configuration `m=24, ef_construct=256, payload_m=24` means:

- `m=24` — each node in the graph has up to 24 bidirectional connections (higher = better
  recall, more memory)
- `ef_construct=256` — beam width during index construction (higher = better recall, slower
  build)
- `payload_m=24` — connections in the payload-filtering graph (enables filtered search
  without full scan)

### Payload Indexes (Required by Strict Mode)

Qdrant's strict mode (`"strict_mode_config": {"enabled": true}`) **requires** a payload index
on any field used in a filter. Without an index, the query returns a 400 error.

Indexes created:

| Field | Index type | Used for |
|---|---|---|
| `contract_name` | KEYWORD | Single-contract scoped search |
| `contract_id` | KEYWORD | Internal cross-reference |
| `part` | KEYWORD | Part_I / Part_II / Part_III filter |
| `category` | KEYWORD | Category filter (Franchise, etc.) |
| `has_governing_law` | BOOL | Pre-filter for governing law chunks |
| `has_termination` | BOOL | Pre-filter for termination chunks |
| `has_cap_liability` | BOOL | Pre-filter for liability cap chunks |
| `has_non_compete` | BOOL | Pre-filter for non-compete chunks |
| `has_indemnification` | BOOL | Pre-filter for indemnification chunks |
| `has_arbitration` | BOOL | Pre-filter for arbitration chunks |
| `has_ip_ownership` | BOOL | Pre-filter for IP ownership chunks |
| `has_exclusivity` | BOOL | Pre-filter for exclusivity chunks |
| `page_start` | INTEGER | Page range filtering |
| `page_end` | INTEGER | Page range filtering |
| `chunk_index` | INTEGER | Sequential ordering |

### Full Payload per Point

Every Qdrant point stores the complete chunk payload including the raw text. This means
retrieval returns everything needed to generate an answer without a secondary lookup:

```json
{
  "chunk_id":         "459c7583-...",
  "contract_id":      "83242bec...",
  "contract_name":    "CreditcardscomInc_20070810_..._Affiliate Agreement",
  "part":             "Part_I",
  "category":         "Affiliate_Agreements",
  "file_path":        "/home/.../CreditcardscomInc...pdf",
  "page_start":       3,
  "page_end":         4,
  "chunk_index":      3,
  "total_chunks":     15,
  "char_count":       2399,
  "text":             "Chase shall only use the list for...",
  "detected_clauses": ["Termination for Convenience"],
  "has_governing_law": false,
  "has_termination":   true,
  ...
}
```

---

## Phase 8 — Hybrid Retrieval with RRF Fusion

### The Core Idea

At query time, we run two retrieval arms in parallel and fuse their results:

```
Query: "does this contract have a non-compete clause?"
         │
         ├── Dense arm: embed_query() → top-20 by cosine similarity
         │   (captures: "restrictions on competing activities", "exclusivity provisions")
         │
         └── Sparse arm: bm25_query_vector() → top-20 by BM25 score
             (captures: "non-compete", "non compete", "competing business")
         │
         └── RRF fusion → re-rank → top-k results
```

### Qdrant Prefetch + FusionQuery

The hybrid search is implemented as a single Qdrant API call using `Prefetch`:

```python
results = qdrant.query_points(
    collection_name=COLLECTION,
    prefetch=[
        Prefetch(query=dense_vector,  using="dense-vector",  limit=20),
        Prefetch(query=sparse_vector, using="sparse-vector", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
    with_payload=True,
)
```

This runs entirely server-side in Qdrant — no round-trips back to the Python client between
the two arms. The prefetch candidates are merged and re-ranked by RRF before being returned.

### Reciprocal Rank Fusion (RRF)

RRF is a simple but highly effective fusion function:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
              arms
```

Where `k=60` (Qdrant default) and `rank_i(d)` is the position of document `d` in arm `i`'s
ranked list (1-indexed).

Example: a chunk that ranks 3rd in dense search and 1st in sparse search:

```
RRF = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
```

A chu