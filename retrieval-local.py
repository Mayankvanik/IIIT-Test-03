"""
retrieval-local.py
==================
Hybrid retrieval over Qdrant using LOCAL resources:
  - Dense  : mixedbread-ai/mxbai-embed-large-v1 (1024-dim, cosine) via SentenceTransformers on CUDA/CPU
  - Sparse : BM25 vectors built during ingestion
  - Fusion : Reciprocal Rank Fusion (RRF) — native Qdrant

Exposes two public functions used by LangGraph nodes/UI:
  retrieve_for_contract(query, contract_name, top_k, clause_filter)
  retrieve_cross_contract(query, contract_names, top_k_per_contract)
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)

from dotenv import load_dotenv
import os
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
QDRANT_URL      = os.getenv("LOCAL_QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("LOCAL_QDRANT_API_KEY", "")
COLLECTION      = os.getenv("LOCAL_COLLECTION", "pdfs-store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")

BASE_DIR = Path(__file__).parent

DENSE_VECTOR_NAME  = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"

BM25_VOCAB_PATH = BASE_DIR / "data/pdfs/CUAD_v1/bm25_vocab.json"

# ─── Singletons (initialised once) ────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_embed_model:   Optional[SentenceTransformer] = None
_bm25_vocab:    Optional[dict] = None


def _qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant_client = QdrantClient(**kwargs)
    return _qdrant_client


def _model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _embed_model


def _vocab() -> dict:
    global _bm25_vocab
    if _bm25_vocab is None:
        with open(BM25_VOCAB_PATH) as f:
            _bm25_vocab = json.load(f)
    return _bm25_vocab


# ─── Dense embedding ──────────────────────────────────────
def embed_query(text: str) -> list[float]:
    """Embed a query string locally using mxbai on GPU/CPU."""
    model = _model()
    # mxbai-embed-large-v1 recommends specific prompt instructions for queries versus documents.
    # The prompt_name="query" uses the internal prefix if configured.
    try:
        encoded = model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    except ValueError:
        # Fallback if sentence-transformers complains about prompt_name not existing
        instruction = "Represent this sentence for searching relevant passages: "
        encoded = model.encode(
            instruction + text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
    return encoded.tolist()


# ─── Sparse BM25 query vector ──────────────────────────────
def bm25_query_vector(text: str) -> SparseVector:
    """
    Convert a query string into a BM25 sparse vector using the vocabulary
    built during ingestion (stored in bm25_vocab.json).
    At query time we only need the IDF scores — no document-length
    normalisation because we don't know the document length — so we use
    a simplified IDF-only scoring: score = log(N / (df + 1) + 1).
    """
    v = _vocab()
    vocab:  dict = v["vocab"]
    df_map: dict = {int(k): val for k, val in v["df"].items()}
    N:      int  = v["N"]

    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    seen: dict[int, float] = {}
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is None:
            continue
        df    = df_map.get(idx, 1)
        score = (N / (df + 1)) ** 0.5     # sqrt-IDF is a stable query weight
        seen[idx] = max(seen.get(idx, 0), score)

    if not seen:
        return SparseVector(indices=[], values=[])
    indices = list(seen.keys())
    values  = list(seen.values())
    return SparseVector(indices=indices, values=values)


# ─── Core hybrid search ───────────────────────────────────
def _hybrid_search(
    query:       str,
    qdrant_filter: Optional[Filter],
    top_k:       int = 8,
) -> list[dict]:
    """
    Run Qdrant hybrid search using named-vector prefetch + RRF fusion.

    Architecture:
      Prefetch dense  → top 20 candidates (semantic meaning)
      Prefetch sparse → top 20 candidates (exact/legal term matching)
      Fuse with RRF   → re-rank and return top_k
    """
    dense  = embed_query(query)
    print('vectorrr',len(dense))
    sparse = bm25_query_vector(query)

    prefetch_dense = Prefetch(
        query=dense,
        using=DENSE_VECTOR_NAME,
        limit=20,
        filter=qdrant_filter,
    )
    prefetch_sparse = Prefetch(
        query=sparse,
        using=SPARSE_VECTOR_NAME,
        limit=20,
        filter=qdrant_filter,
    )

    results = _qdrant().query_points(
        collection_name=COLLECTION,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score":         hit.score,
            "text":          hit.payload.get("text", ""),
            "contract_name": hit.payload.get("contract_name", ""),
            "contract_id":   hit.payload.get("contract_id", ""),
            "page_start":    hit.payload.get("page_start", 0),
            "page_end":      hit.payload.get("page_end", 0),
            "chunk_index":   hit.payload.get("chunk_index", 0),
            "category":      hit.payload.get("category", ""),
            "part":          hit.payload.get("part", ""),
            "detected_clauses": hit.payload.get("detected_clauses", []),
        }
        for hit in results.points
    ]


# ─── Public: single-contract retrieval ────────────────────
def retrieve_for_contract(
    query:          str,
    contract_name:  str,
    top_k:          int = 6,
    clause_filter:  Optional[str] = None,
) -> list[dict]:
    """
    Hybrid search scoped to ONE contract.

    Args:
        query:         Natural language question or clause description.
        contract_name: Exact contract_name value stored in Qdrant payload.
        top_k:         Number of chunks to return.
        clause_filter: Optional boolean payload field e.g. 'has_non_compete'.
                       If provided, adds a must-match condition to pre-filter
                       chunks that the lightweight ingest tagger already flagged.
    """
    conditions = [
        FieldCondition(
            key="contract_name",
            match=MatchValue(value=contract_name),
        )
    ]
    if clause_filter:
        conditions.append(
            FieldCondition(key=clause_filter, match=MatchValue(value=True))
        )

    f = Filter(must=conditions)
    return _hybrid_search(query, f, top_k)


# ─── Public: cross-contract retrieval ─────────────────────
def retrieve_cross_contract(
    query:          str,
    contract_names: list[str],
    top_k_per:      int = 3,
) -> dict[str, list[dict]]:
    """
    Run one hybrid search PER contract and return results grouped by contract.
    Used for comparison and risk-flagging across multiple contracts.

    Returns:
        { contract_name: [chunk_dicts...], ... }
    """
    results: dict[str, list[dict]] = {}
    for name in contract_names:
        chunks = retrieve_for_contract(query, name, top_k=top_k_per)
        results[name] = chunks
    return results


# ─── Public: list available contracts ─────────────────────
def list_contracts(category_filter: Optional[str] = None) -> list[str]:
    """
    Scroll Qdrant to collect unique contract names.
    Optionally filter by category folder (e.g. 'Affiliate_Agreements').
    """
    f = None
    if category_filter:
        f = Filter(must=[
            FieldCondition(key="category", match=MatchValue(value=category_filter))
        ])

    seen: set[str] = set()
    offset = None

    while True:
        resp = _qdrant().scroll(
            collection_name=COLLECTION,
            scroll_filter=f,
            limit=250,
            offset=offset,
            with_payload=["contract_name"],
            with_vectors=False,
        )
        for point in resp[0]:
            name = point.payload.get("contract_name", "")
            if name:
                seen.add(name)
        offset = resp[1]
        if offset is None:
            break

    return sorted(seen)


# ─── Public: fetch all chunks for one contract ────────────
def fetch_all_chunks(contract_name: str) -> list[dict]:
    """
    Retrieve ALL stored chunks for a given contract (no query — ordered by
    chunk_index).  Used by the clause-extraction node to scan the full doc.
    """
    f = Filter(must=[
        FieldCondition(key="contract_name", match=MatchValue(value=contract_name))
    ])
    chunks = []
    offset = None
    while True:
        resp = _qdrant().scroll(
            collection_name=COLLECTION,
            scroll_filter=f,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in resp[0]:
            chunks.append(point.payload)
        offset = resp[1]
        if offset is None:
            break

    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    return chunks

# ─── Test Harness ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing local retrieval...")
    
    # 1. Fetch available contracts
    contracts = list_contracts()
    print(f"\\nFound {len(contracts)} contracts indexed in local Qdrant:")
    
    if contracts:
        sample_contract = contracts[0]
        print(f"\\nRetrieving context for query 'What is the governing law?' on contract: {sample_contract}")
        
        # 2. Hybrid search test
        results = retrieve_for_contract("What is the governing law?", sample_contract, top_k=2)
        
        for i, hit in enumerate(results):
            print(f"\\n--- Hit {i+1} : Score = {hit['score']:.4f} ---")
            print(hit['text'][:200] + "...")
