"""
CUAD Contract Ingestion Pipeline
=================================
- Recursively finds all PDFs in the dataset folder structure
- Extracts + cleans text with PyMuPDF
- Applies hierarchical chunking strategy (section-aware + sliding window)
- Generates OpenAI dense embeddings (1536-dim) + sparse BM25 vectors
- Stores everything in Qdrant with rich metadata for hybrid search
"""

import os
import re
import uuid
import json
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
)
from dotenv import load_dotenv
import os
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION     = os.getenv("COLLECTION", "pdfs-store")

BASE_DIR = Path(__file__).parent
PDF_ROOT = BASE_DIR / "data/pdfs/CUAD_v1/full_contract_pdf"

# Chunking hyperparameters
CHUNK_SIZE      = 600    # tokens (approx 450 words)
CHUNK_OVERLAP   = 120    # tokens overlap between consecutive chunks
MIN_CHUNK_CHARS = 80     # discard chunks shorter than this
EMBED_BATCH     = 16     # how many chunks to embed per OpenAI call
UPSERT_BATCH    = 64     # how many points to upsert per Qdrant call

# 41 CUAD clause categories for auto-tagging
CUAD_CATEGORIES = [
    "Governing Law", "Termination for Convenience", "Cap on Liability",
    "Non-Compete", "IP Ownership Assignment", "Revenue/Profit Sharing",
    "Minimum Commitment", "Audit Rights", "Indemnification",
    "Limitation of Liability", "Exclusivity", "Non-Solicitation",
    "Change of Control", "Renewal Term", "Notice Period to Terminate Renewal",
    "Anti-Assignment", "License Grant", "Warranty Duration",
    "Insurance", "Covenant Not to Sue", "Third Party Beneficiary",
    "Price Restrictions", "Most Favored Nation", "Competitive Restriction Exception",
    "Source Code Escrow", "Post-Termination Services", "Liquidated Damages",
    "Arbitration", "Uncapped Liability", "Affiliate License-Licensee",
    "Affiliate License-Licensor", "Irrevocable or Perpetual License",
    "Volume Restriction", "Joint IP Ownership", "Unlimited/All-You-Can-Eat License",
    "ROFR/ROFO/ROFN", "Sponsorship", "Development Agreement",
    "Parties", "Agreement Date", "Effective Date"
]

# Simple keyword map for lightweight clause-type tagging (no LLM needed at ingest)
CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "Governing Law":               ["governing law", "governed by", "jurisdiction", "laws of the state"],
    "Termination for Convenience": ["terminate for convenience", "terminate without cause", "termination for convenience"],
    "Cap on Liability":            ["cap on liability", "liability shall not exceed", "maximum liability", "aggregate liability"],
    "Non-Compete":                 ["non-compete", "non compete", "not compete", "competing business"],
    "IP Ownership Assignment":     ["intellectual property", "ip ownership", "assigns all right", "work for hire"],
    "Indemnification":             ["indemnif", "hold harmless", "defend and indemnify"],
    "Audit Rights":                ["audit right", "right to audit", "inspect records"],
    "Arbitration":                 ["arbitration", "binding arbitration", "arbitrator"],
    "Renewal Term":                ["renewal term", "automatically renew", "auto-renew", "evergreen"],
    "Exclusivity":                 ["exclusive", "exclusively", "sole and exclusive"],
    "Non-Solicitation":            ["non-solicit", "not solicit", "no solicitation"],
    "Limitation of Liability":     ["limitation of liability", "in no event shall", "consequential damages"],
    "Liquidated Damages":          ["liquidated damages", "penalty", "agreed damages"],
}


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class PageBlock:
    """One cleaned page of text."""
    page_num: int
    text: str


@dataclass
class Chunk:
    """A single chunk ready for embedding and storage."""
    chunk_id:        str
    contract_id:     str
    contract_name:   str
    part:            str          # Part_I / Part_II / Part_III
    category:        str          # subfolder name (e.g. "Affiliate_Agreements")
    file_path:       str
    page_start:      int
    page_end:        int
    chunk_index:     int
    total_chunks:    int          # filled in after all chunks are created
    text:            str
    char_count:      int
    detected_clauses: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# STEP 1 — PDF DISCOVERY
# ─────────────────────────────────────────────
def discover_pdfs(root: str) -> list[dict]:
    """
    Walk the directory tree and collect every PDF with its structural metadata.
    Returns a list of dicts: {path, part, category, contract_name}
    """
    records = []
    root_path = Path(root)

    for pdf_path in sorted(root_path.rglob("*.pdf")):
        parts = pdf_path.relative_to(root_path).parts
        # parts[0] = "CUAD_v1", parts[1] = "full_contract_pdf" or other
        # parts[2] = Part_I / Part_II / ... , parts[3] = category folder
        part     = parts[2] if len(parts) > 2 else "Unknown"
        category = parts[3] if len(parts) > 3 else "Unknown"
        records.append({
            "path":          str(pdf_path),
            "part":          part,
            "category":      category,
            "contract_name": pdf_path.stem,
        })

    print(f"[Discovery] Found {len(records)} PDFs under {root}")
    return records


# ─────────────────────────────────────────────
# STEP 2 — TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_pages(pdf_path: str) -> list[PageBlock]:
    """
    Extract text page-by-page with PyMuPDF.
    Applies light cleaning: strip headers/footers heuristics, normalise whitespace.
    """
    blocks: list[PageBlock] = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = _clean_page_text(raw, page_num)
            if len(cleaned) >= MIN_CHUNK_CHARS:
                blocks.append(PageBlock(page_num=page_num, text=cleaned))
        doc.close()
    except Exception as e:
        print(f"  [WARN] Could not extract {pdf_path}: {e}")
    return blocks


def _clean_page_text(text: str, page_num: int) -> str:
    """Remove common PDF noise: page numbers, long dashes, multiple newlines."""
    # Remove lone page-number lines (e.g. "- 12 -" or just "12")
    text = re.sub(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ newlines to double
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise whitespace within lines
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


# ─────────────────────────────────────────────
# STEP 3 — HIERARCHICAL CHUNKING
# ─────────────────────────────────────────────
def chunk_contract(
    pages: list[PageBlock],
    record: dict,
    contract_id: str,
) -> list[Chunk]:
    """
    Two-pass chunking strategy:
    
    Pass A — Section-aware split:
        Detect natural section boundaries (numbered headings, ALL-CAPS titles).
        Each section becomes a candidate chunk.  If a section is too long it is
        further split by sliding window (Pass B).  If two consecutive sections
        are both short they are merged until CHUNK_SIZE is approached.
    
    Pass B — Sliding-window fallback:
        For any candidate that exceeds CHUNK_SIZE tokens, apply a sliding
        window with CHUNK_OVERLAP overlap so no context is lost at boundaries.
    
    Result: chunks that respect section semantics wherever possible, never
    exceed ~800 tokens, and always overlap neighbouring chunks.
    """
    # Merge all pages into one string, keeping track of page boundaries
    full_text = ""
    page_offsets: list[tuple[int, int]] = []  # (char_start, page_num)
    for pb in pages:
        start = len(full_text)
        full_text += pb.text + "\n\n"
        page_offsets.append((start, pb.page_num))

    # ── Pass A: find section boundaries ──
    section_pattern = re.compile(
        r"(?m)^(?:"
        r"\d{1,2}(?:\.\d{1,2})*\s+[A-Z]"   # "1.2 SECTION TITLE"
        r"|[A-Z][A-Z\s]{5,50}$"              # "GOVERNING LAW"
        r"|(?:ARTICLE|SECTION)\s+[IVXLC\d]" # "ARTICLE IV"
        r")"
    )
    boundaries = [m.start() for m in section_pattern.finditer(full_text)]
    boundaries = [0] + boundaries + [len(full_text)]

    candidate_texts: list[tuple[int, int]] = []  # (char_start, char_end)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        candidate_texts.append((start, end))

    # ── Merge tiny sections, split large ones ──
    CHARS_PER_TOKEN = 4  # rough estimate
    target_chars    = CHUNK_SIZE   * CHARS_PER_TOKEN
    overlap_chars   = CHUNK_OVERLAP * CHARS_PER_TOKEN

    raw_chunks: list[tuple[int, int]] = []
    buffer_start, buffer_end = 0, 0
    buffer_len = 0

    for cs, ce in candidate_texts:
        seg_len = ce - cs
        if buffer_len == 0:
            buffer_start, buffer_end, buffer_len = cs, ce, seg_len
        elif buffer_len + seg_len <= target_chars:
            # merge into buffer
            buffer_end, buffer_len = ce, buffer_len + seg_len
        else:
            # flush buffer
            if buffer_len >= MIN_CHUNK_CHARS:
                raw_chunks.extend(_sliding_window(buffer_start, buffer_end, full_text, target_chars, overlap_chars))
            buffer_start, buffer_end, buffer_len = cs, ce, seg_len

    if buffer_len >= MIN_CHUNK_CHARS:
        raw_chunks.extend(_sliding_window(buffer_start, buffer_end, full_text, target_chars, overlap_chars))

    # ── Build Chunk objects ──
    chunks: list[Chunk] = []
    for idx, (cs, ce) in enumerate(raw_chunks):
        text = full_text[cs:ce].strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        p_start = _char_to_page(cs, page_offsets)
        p_end   = _char_to_page(ce, page_offsets)
        chunk = Chunk(
            chunk_id        = str(uuid.uuid4()),
            contract_id     = contract_id,
            contract_name   = record["contract_name"],
            part            = record["part"],
            category        = record["category"],
            file_path       = record["path"],
            page_start      = p_start,
            page_end        = p_end,
            chunk_index     = idx,
            total_chunks    = 0,     # set below
            text            = text,
            char_count      = len(text),
            detected_clauses= _detect_clauses(text),
        )
        chunks.append(chunk)

    # fill total_chunks now we know the count
    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def _sliding_window(cs: int, ce: int, text: str, target: int, overlap: int) -> list[tuple[int, int]]:
    """Split [cs, ce] into overlapping windows of ~target chars."""
    result = []
    start = cs
    while start < ce:
        end = min(start + target, ce)
        result.append((start, end))
        if end == ce:
            break
        start = end - overlap
    return result


def _char_to_page(char_pos: int, offsets: list[tuple[int, int]]) -> int:
    """Map a character offset back to a page number."""
    page = 1
    for (off, pnum) in offsets:
        if off <= char_pos:
            page = pnum
        else:
            break
    return page


def _detect_clauses(text: str) -> list[str]:
    """Keyword-based clause detection — fast, no LLM required at ingest."""
    tl = text.lower()
    found = []
    for clause, keywords in CLAUSE_KEYWORDS.items():
        if any(kw in tl for kw in keywords):
            found.append(clause)
    return found


# ─────────────────────────────────────────────
# STEP 4 — OPENAI DENSE EMBEDDINGS
# ─────────────────────────────────────────────
def embed_chunks(chunks: list[Chunk], client: openai.OpenAI) -> list[list[float]]:
    """
    Batch embed chunk texts with text-embedding-3-small (1536 dims).
    Retries with exponential back-off on rate-limit errors.
    """
    texts = [c.text for c in chunks]
    vectors: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i: i + EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                vectors.extend([e.embedding for e in resp.data])
                break
            except openai.RateLimitError:
                wait = 2 ** attempt
                print(f"  [RateLimit] waiting {wait}s ...")
                time.sleep(wait)
            except Exception as e:
                print(f"  [EmbedError] {e}")
                vectors.extend([[0.0] * 1536] * len(batch))
                break

    return vectors


# ─────────────────────────────────────────────
# STEP 5 — BM25 SPARSE VECTORS
# ─────────────────────────────────────────────
class BM25Vectoriser:
    """
    Lightweight BM25 sparse vector builder — no external library needed.
    Produces {token_index: bm25_weight} dicts compatible with Qdrant SparseVector.
    The vocabulary is built incrementally during ingestion.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1   = k1
        self.b    = b
        self.vocab: dict[str, int] = {}
        self.df:    dict[int, int] = {}   # document frequency per token
        self.N     = 0                    # total docs seen
        self._corpus_len = 0
        self._doc_lens: list[int] = []

    # ── Pass 1: fit on all texts ──
    def fit(self, texts: list[str]):
        self.N = len(texts)
        for text in texts:
            tokens = set(self._tokenise(text))
            self._doc_lens.append(len(self._tokenise(text)))
            for tok in tokens:
                idx = self._get_idx(tok)
                self.df[idx] = self.df.get(idx, 0) + 1
        self._avgdl = sum(self._doc_lens) / max(self.N, 1)

    # ── Pass 2: transform one text ──
    def transform(self, text: str) -> SparseVector:
        tokens  = self._tokenise(text)
        dl      = len(tokens)
        tf: dict[int, int] = {}
        for tok in tokens:
            idx = self._get_idx(tok)
            tf[idx] = tf.get(idx, 0) + 1

        indices, values = [], []
        for idx, freq in tf.items():
            df  = self.df.get(idx, 1)
            idf = max(0.0, (self.N - df + 0.5) / (df + 0.5))
            numerator   = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score = idf * numerator / denominator
            if score > 0:
                indices.append(idx)
                values.append(float(score))

        return SparseVector(indices=indices, values=values)

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _get_idx(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]


# ─────────────────────────────────────────────
# STEP 6 — UPSERT TO QDRANT
# ─────────────────────────────────────────────
def upsert_to_qdrant(
    chunks:        list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    qdrant:        QdrantClient,
):
    """
    Build PointStruct objects with named vectors + full metadata payload
    and upsert in batches.
    """
    points = []
    for chunk, dv, sv in zip(chunks, dense_vectors, sparse_vectors):
        payload = {
            # ── Identity ──
            "chunk_id":        chunk.chunk_id,
            "contract_id":     chunk.contract_id,
            "contract_name":   chunk.contract_name,
            # ── Location ──
            "part":            chunk.part,
            "category":        chunk.category,
            "file_path":       chunk.file_path,
            "page_start":      chunk.page_start,
            "page_end":        chunk.page_end,
            # ── Chunk position ──
            "chunk_index":     chunk.chunk_index,
            "total_chunks":    chunk.total_chunks,
            "char_count":      chunk.char_count,
            # ── Content ──
            "text":            chunk.text,
            "detected_clauses": chunk.detected_clauses,
            # ── Search helpers ──
            "has_governing_law":      "Governing Law"               in chunk.detected_clauses,
            "has_termination":        "Termination for Convenience" in chunk.detected_clauses,
            "has_cap_liability":      "Cap on Liability"            in chunk.detected_clauses,
            "has_non_compete":        "Non-Compete"                 in chunk.detected_clauses,
            "has_indemnification":    "Indemnification"             in chunk.detected_clauses,
            "has_arbitration":        "Arbitration"                 in chunk.detected_clauses,
            "has_ip_ownership":       "IP Ownership Assignment"     in chunk.detected_clauses,
            "has_exclusivity":        "Exclusivity"                 in chunk.detected_clauses,
        }

        point = PointStruct(
            id=chunk.chunk_id,
            vector={
                "dense-vector":  dv,   # ← was "dense"
                "sparse-vector": sv,   # ← was "sparse"
            },
            payload=payload,
        )
        points.append(point)

    # Upsert in batches
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i: i + UPSERT_BATCH]
        qdrant.upsert(collection_name=COLLECTION, points=batch)
        print(f"  [Qdrant] Upserted points {i}–{i+len(batch)-1}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(limit: Optional[int] = None):
    """
    Full end-to-end ingestion pipeline.
    Set limit=20 for a quick smoke test, None for all contracts.
    """
    print("=" * 60)
    print("  CUAD Contract Ingestion Pipeline")
    print("=" * 60)

    # ── Clients ──
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    oai    = openai.OpenAI(api_key=OPENAI_API_KEY)

    # ── Discover PDFs ──
    records = discover_pdfs(PDF_ROOT)
    if limit:
        records = records[:limit]
        print(f"[Pipeline] Limiting to {limit} PDFs for this run.")

    # ── Per-contract processing ──
    all_chunks:        list[Chunk]       = []
    all_dense:         list[list[float]] = []

    bm25 = BM25Vectoriser()
    # We need two passes for BM25: first collect all texts, then transform
    # So we store chunks first, embed dense in parallel, then do BM25 fit+transform

    for rec_idx, record in enumerate(records):
        print(f"\n[{rec_idx+1}/{len(records)}] {record['contract_name']}")

        # stable contract ID based on file path hash
        contract_id = hashlib.md5(record["path"].encode()).hexdigest()

        # extract
        pages = extract_pages(record["path"])
        if not pages:
            print("  [SKIP] No extractable text.")
            continue

        # chunk
        chunks = chunk_contract(pages, record, contract_id)
        print(f"  → {len(pages)} pages → {len(chunks)} chunks")

        # dense embed
        dense = embed_chunks(chunks, oai)

        all_chunks.extend(chunks)
        all_dense.extend(dense)

    print(f"\n[BM25] Fitting on {len(all_chunks)} chunks ...")
    bm25.fit([c.text for c in all_chunks])

    print("[BM25] Transforming ...")
    all_sparse = [bm25.transform(c.text) for c in all_chunks]

    print(f"\n[Qdrant] Upserting {len(all_chunks)} points to '{COLLECTION}' ...")
    upsert_to_qdrant(all_chunks, all_dense, all_sparse, qdrant)

    # ── Persist BM25 vocab for query-time use ──
    vocab_path = Path(PDF_ROOT).parent / "bm25_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({
            "vocab":  bm25.vocab,
            "df":     {str(k): v for k, v in bm25.df.items()},
            "N":      bm25.N,
            "avgdl":  bm25._avgdl,
        }, f)
    print(f"[BM25] Vocabulary saved → {vocab_path}")

    print("\n✅  Pipeline complete.")
    print(f"    Total contracts processed : {len(set(c.contract_id for c in all_chunks))}")
    print(f"    Total chunks stored       : {len(all_chunks)}")
    print(f"    BM25 vocabulary size      : {len(bm25.vocab):,} tokens")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CUAD ingestion pipeline")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N contracts (omit for all)")
    args = ap.parse_args()
    run_pipeline(limit=args.limit)