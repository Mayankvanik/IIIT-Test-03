"""
CUAD Contract Local Ingestion Pipeline
=======================================
- Recursively finds all PDFs in the dataset folder structure
- Extracts + cleans text with PyMuPDF
- Applies hierarchical chunking strategy (section-aware + sliding window)
- Generates LOCAL dense embeddings via mixedbread-ai/mxbai-embed-large-v1 (1024-dim) on CUDA
- Generates sparse BM25 vectors
- Stores everything in a LOCAL Docker Qdrant instance with rich metadata for hybrid search

Usage:
    python ingest-local.py [--limit N]

Requirements:
    pip install sentence-transformers torch qdrant-client pymupdf python-dotenv
    Docker: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""

import os
import re
import uuid
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG  (local overrides — no cloud credentials needed)
# ─────────────────────────────────────────────
QDRANT_URL      = os.getenv("LOCAL_QDRANT_URL",  "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("LOCAL_QDRANT_API_KEY", "")   # empty for local Docker
COLLECTION      = os.getenv("LOCAL_COLLECTION",  "pdfs-store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",   "mixedbread-ai/mxbai-embed-large-v1")
EMBED_DIM       = 1024   # mxbai-embed-large-v1 output dimension

BASE_DIR = Path(__file__).parent.parent
PDF_ROOT = BASE_DIR / "data/pdfs/CUAD_v1/full_contract_pdf"

# Chunking hyperparameters  (same as cloud ingest.py)
CHUNK_SIZE      = 600   # approx tokens
CHUNK_OVERLAP   = 120
MIN_CHUNK_CHARS = 80
EMBED_BATCH     = 32    # larger batch is fine locally on GPU
UPSERT_BATCH    = 64


# ─────────────────────────────────────────────
# CUAD CLAUSE TAXONOMY  (unchanged from ingest.py)
# ─────────────────────────────────────────────
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
    chunk_id:         str
    contract_id:      str
    contract_name:    str
    part:             str   # Part_I / Part_II / Part_III
    category:         str   # subfolder name (e.g. "Affiliate_Agreements")
    file_path:        str
    page_start:       int
    page_end:         int
    chunk_index:      int
    total_chunks:     int   # filled after all chunks are created
    text:             str
    char_count:       int
    detected_clauses: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# STEP 0 — LOCAL EMBEDDING MODEL
# ─────────────────────────────────────────────
def load_embedding_model() -> tuple[SentenceTransformer, str]:
    """
    Load mixedbread-ai/mxbai-embed-large-v1 onto CUDA if available,
    otherwise fall back to CPU.
    Returns (model, device_str).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"Loading dense model {EMBEDDING_MODEL} on GPU (CUDA)...")
    else:
        logger.warning(f"CUDA not available. Loading {EMBEDDING_MODEL} on CPU (slow)...")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info(f"Embedding model loaded. Output dim = {EMBED_DIM}")
    return model, device


# ─────────────────────────────────────────────
# STEP 0b — QDRANT COLLECTION SETUP
# ─────────────────────────────────────────────
def ensure_collection(qdrant: QdrantClient):
    """
    Create the Qdrant collection with named dense + sparse vector spaces
    if it does not already exist.
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        logger.info(f"Collection '{COLLECTION}' already exists — skipping creation.")
        return

    logger.info(f"Creating collection '{COLLECTION}' (dense={EMBED_DIM}d + sparse BM25)...")
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense-vector": VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse-vector": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(f"Collection '{COLLECTION}' created successfully.")


# ─────────────────────────────────────────────
# STEP 1 — PDF DISCOVERY
# ─────────────────────────────────────────────
def discover_pdfs(root: Path) -> list[dict]:
    """
    Walk the directory tree and collect every PDF with structural metadata.
    Returns: [{path, part, category, contract_name}, ...]
    """
    records = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        parts    = pdf_path.relative_to(root).parts
        part     = parts[2] if len(parts) > 2 else "Unknown"
        category = parts[3] if len(parts) > 3 else "Unknown"
        records.append({
            "path":          str(pdf_path),
            "part":          part,
            "category":      category,
            "contract_name": pdf_path.stem,
        })
    logger.info(f"[Discovery] Found {len(records)} PDFs under {root}")
    return records


# ─────────────────────────────────────────────
# STEP 2 — TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_pages(pdf_path: str) -> list[PageBlock]:
    """Extract text page-by-page with PyMuPDF and apply light cleaning."""
    blocks: list[PageBlock] = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            raw     = page.get_text("text")
            cleaned = _clean_page_text(raw)
            if len(cleaned) >= MIN_CHUNK_CHARS:
                blocks.append(PageBlock(page_num=page_num, text=cleaned))
        doc.close()
    except Exception as exc:
        logger.warning(f"Could not extract '{pdf_path}': {exc}")
    return blocks


def _clean_page_text(text: str) -> str:
    """Remove PDF noise: lone page numbers, redundant newlines, extra whitespace."""
    text = re.sub(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────
# STEP 3 — HIERARCHICAL CHUNKING
# ─────────────────────────────────────────────
def chunk_contract(pages: list[PageBlock], record: dict, contract_id: str) -> list[Chunk]:
    """
    Two-pass chunking:
      Pass A — section-aware splits on numbered/CAPS headings.
      Pass B — sliding-window fallback for oversized sections.
    """
    # Build full text + page offset map
    full_text    = ""
    page_offsets: list[tuple[int, int]] = []
    for pb in pages:
        start = len(full_text)
        full_text += pb.text + "\n\n"
        page_offsets.append((start, pb.page_num))

    # Pass A — detect section boundaries
    section_re = re.compile(
        r"(?m)^(?:"
        r"\d{1,2}(?:\.\d{1,2})*\s+[A-Z]"
        r"|[A-Z][A-Z\s]{5,50}$"
        r"|(?:ARTICLE|SECTION)\s+[IVXLC\d]"
        r")"
    )
    boundaries = [0] + [m.start() for m in section_re.finditer(full_text)] + [len(full_text)]
    candidates  = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    CHARS_PER_TOKEN = 4
    target_chars    = CHUNK_SIZE   * CHARS_PER_TOKEN
    overlap_chars   = CHUNK_OVERLAP * CHARS_PER_TOKEN

    # Merge tiny, split large
    raw_chunks: list[tuple[int, int]] = []
    buf_s = buf_e = buf_len = 0
    for cs, ce in candidates:
        seg = ce - cs
        if buf_len == 0:
            buf_s, buf_e, buf_len = cs, ce, seg
        elif buf_len + seg <= target_chars:
            buf_e, buf_len = ce, buf_len + seg
        else:
            if buf_len >= MIN_CHUNK_CHARS:
                raw_chunks.extend(_sliding_window(buf_s, buf_e, full_text, target_chars, overlap_chars))
            buf_s, buf_e, buf_len = cs, ce, seg
    if buf_len >= MIN_CHUNK_CHARS:
        raw_chunks.extend(_sliding_window(buf_s, buf_e, full_text, target_chars, overlap_chars))

    # Build Chunk objects
    chunks: list[Chunk] = []
    for idx, (cs, ce) in enumerate(raw_chunks):
        text = full_text[cs:ce].strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        chunks.append(Chunk(
            chunk_id        = str(uuid.uuid4()),
            contract_id     = contract_id,
            contract_name   = record["contract_name"],
            part            = record["part"],
            category        = record["category"],
            file_path       = record["path"],
            page_start      = _char_to_page(cs, page_offsets),
            page_end        = _char_to_page(ce, page_offsets),
            chunk_index     = idx,
            total_chunks    = 0,
            text            = text,
            char_count      = len(text),
            detected_clauses= _detect_clauses(text),
        ))
    for c in chunks:
        c.total_chunks = len(chunks)
    return chunks


def _sliding_window(cs, ce, text, target, overlap) -> list[tuple[int, int]]:
    result, start = [], cs
    while start < ce:
        end = min(start + target, ce)
        result.append((start, end))
        if end == ce:
            break
        start = end - overlap
    return result


def _char_to_page(char_pos: int, offsets: list[tuple[int, int]]) -> int:
    page = 1
    for off, pnum in offsets:
        if off <= char_pos:
            page = pnum
        else:
            break
    return page


def _detect_clauses(text: str) -> list[str]:
    tl = text.lower()
    return [clause for clause, kws in CLAUSE_KEYWORDS.items() if any(kw in tl for kw in kws)]


# ─────────────────────────────────────────────
# STEP 4 — LOCAL DENSE EMBEDDINGS (CUDA)
# ─────────────────────────────────────────────
def embed_chunks_local(
    chunks: list[Chunk],
    model:  SentenceTransformer,
) -> list[list[float]]:
    """
    Batch-encode chunk texts with the local SentenceTransformer model.
    Uses CUDA automatically if the model was loaded on GPU.
    Applies prompt_name='query' is NOT used here — these are document embeddings,
    so no instruction prefix is needed for mxbai-embed-large-v1.
    """
    texts   = [c.text for c in chunks]
    vectors: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        logger.info(f"  [Embed] Batch {i // EMBED_BATCH + 1} | chunks {i}–{i + len(batch) - 1}")
        encoded = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity works best with L2-normalised vecs
            convert_to_numpy=True,
        )
        vectors.extend(encoded.tolist())

    return vectors


# ─────────────────────────────────────────────
# STEP 5 — BM25 SPARSE VECTORS
# ─────────────────────────────────────────────
class BM25Vectoriser:
    """
    Lightweight BM25 sparse vector builder.
    Produces SparseVector objects compatible with Qdrant.
    Vocabulary is built incrementally during ingestion.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1    = k1
        self.b     = b
        self.vocab: dict[str, int] = {}
        self.df:    dict[int, int] = {}
        self.N      = 0
        self._avgdl = 0.0
        self._doc_lens: list[int] = []

    def fit(self, texts: list[str]):
        self.N = len(texts)
        for text in texts:
            tokens = set(self._tokenise(text))
            self._doc_lens.append(len(self._tokenise(text)))
            for tok in tokens:
                idx = self._idx(tok)
                self.df[idx] = self.df.get(idx, 0) + 1
        self._avgdl = sum(self._doc_lens) / max(self.N, 1)

    def transform(self, text: str) -> SparseVector:
        tokens = self._tokenise(text)
        dl     = len(tokens)
        tf: dict[int, int] = {}
        for tok in tokens:
            idx = self._idx(tok)
            tf[idx] = tf.get(idx, 0) + 1

        indices, values = [], []
        for idx, freq in tf.items():
            df  = self.df.get(idx, 1)
            idf = max(0.0, (self.N - df + 0.5) / (df + 0.5))
            num = freq * (self.k1 + 1)
            den = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score = idf * num / den
            if score > 0:
                indices.append(idx)
                values.append(float(score))
        return SparseVector(indices=indices, values=values)

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _idx(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]


# ─────────────────────────────────────────────
# STEP 6 — UPSERT TO LOCAL QDRANT
# ─────────────────────────────────────────────
def upsert_to_qdrant(
    chunks:         list[Chunk],
    dense_vectors:  list[list[float]],
    sparse_vectors: list[SparseVector],
    qdrant:         QdrantClient,
):
    """Build PointStructs with named vectors + full metadata and upsert in batches."""
    points = []
    for chunk, dv, sv in zip(chunks, dense_vectors, sparse_vectors):
        payload = {
            # Identity
            "chunk_id":       chunk.chunk_id,
            "contract_id":    chunk.contract_id,
            "contract_name":  chunk.contract_name,
            # Location
            "part":           chunk.part,
            "category":       chunk.category,
            "file_path":      chunk.file_path,
            "page_start":     chunk.page_start,
            "page_end":       chunk.page_end,
            # Chunk position
            "chunk_index":    chunk.chunk_index,
            "total_chunks":   chunk.total_chunks,
            "char_count":     chunk.char_count,
            # Content
            "text":           chunk.text,
            "detected_clauses": chunk.detected_clauses,
            # Search helpers
            "has_governing_law":   "Governing Law"               in chunk.detected_clauses,
            "has_termination":     "Termination for Convenience" in chunk.detected_clauses,
            "has_cap_liability":   "Cap on Liability"            in chunk.detected_clauses,
            "has_non_compete":     "Non-Compete"                 in chunk.detected_clauses,
            "has_indemnification": "Indemnification"             in chunk.detected_clauses,
            "has_arbitration":     "Arbitration"                 in chunk.detected_clauses,
            "has_ip_ownership":    "IP Ownership Assignment"     in chunk.detected_clauses,
            "has_exclusivity":     "Exclusivity"                 in chunk.detected_clauses,
        }
        points.append(PointStruct(
            id      = chunk.chunk_id,
            vector  = {
                "dense-vector":  dv,
                "sparse-vector": sv,
            },
            payload = payload,
        ))

    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i : i + UPSERT_BATCH]
        qdrant.upsert(collection_name=COLLECTION, points=batch)
        logger.info(f"  [Qdrant] Upserted points {i}–{i + len(batch) - 1}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(limit: Optional[int] = None):
    """
    Full end-to-end LOCAL ingestion pipeline.
    Set limit=5 for a smoke test, None for all contracts.
    """
    logger.info("=" * 60)
    logger.info("  CUAD Contract LOCAL Ingestion Pipeline")
    logger.info(f"  Embedding : {EMBEDDING_MODEL} ({EMBED_DIM}d)")
    logger.info(f"  Qdrant    : {QDRANT_URL}  collection='{COLLECTION}'")
    logger.info("=" * 60)

    # ── Load local embedding model ──
    embed_model, device = load_embedding_model()

    # ── Connect to local Docker Qdrant ──
    qdrant_kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        qdrant_kwargs["api_key"] = QDRANT_API_KEY
    qdrant = QdrantClient(**qdrant_kwargs)
    ensure_collection(qdrant)

    # ── Discover PDFs ──
    records = discover_pdfs(PDF_ROOT)
    if limit:
        records = records[:limit]
        logger.info(f"[Pipeline] Limiting to {limit} PDFs for this run.")

    # ── Per-contract processing ──
    all_chunks: list[Chunk]       = []
    all_dense:  list[list[float]] = []
    bm25 = BM25Vectoriser()

    t0 = time.time()
    for rec_idx, record in enumerate(records):
        logger.info(f"\n[{rec_idx + 1}/{len(records)}] {record['contract_name']}")

        contract_id = hashlib.md5(record["path"].encode()).hexdigest()

        pages = extract_pages(record["path"])
        if not pages:
            logger.warning("  [SKIP] No extractable text.")
            continue

        chunks = chunk_contract(pages, record, contract_id)
        logger.info(f"  → {len(pages)} pages → {len(chunks)} chunks")

        dense = embed_chunks_local(chunks, embed_model)

        all_chunks.extend(chunks)
        all_dense.extend(dense)

    logger.info(f"\n[BM25] Fitting on {len(all_chunks)} chunks ...")
    bm25.fit([c.text for c in all_chunks])

    logger.info("[BM25] Transforming ...")
    all_sparse = [bm25.transform(c.text) for c in all_chunks]

    logger.info(f"\n[Qdrant] Upserting {len(all_chunks)} points to '{COLLECTION}' ...")
    upsert_to_qdrant(all_chunks, all_dense, all_sparse, qdrant)

    # ── Persist BM25 vocab for query-time use ──
    vocab_path = "C:/python/i2e_consultancy/practical-Task-03/data/pdfs/CUAD_v1/bm25_vocab.json"
    #PDF_ROOT.parent / "bm25_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({
            "vocab": bm25.vocab,
            "df":    {str(k): v for k, v in bm25.df.items()},
            "N":     bm25.N,
            "avgdl": bm25._avgdl,
        }, f)
    logger.info(f"[BM25] Vocabulary saved → {vocab_path}")

    elapsed = time.time() - t0
    logger.info("\n✅  Local pipeline complete.")
    logger.info(f"    Device                    : {device.upper()}")
    logger.info(f"    Total contracts processed : {len(set(c.contract_id for c in all_chunks))}")
    logger.info(f"    Total chunks stored       : {len(all_chunks)}")
    logger.info(f"    BM25 vocabulary size      : {len(bm25.vocab):,} tokens")
    logger.info(f"    Wall-clock time           : {elapsed / 60:.1f} min")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CUAD LOCAL ingestion pipeline (no OpenAI)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N contracts (omit for all). Use 5 for smoke test.")
    args = ap.parse_args()
    run_pipeline(limit=args.limit)
