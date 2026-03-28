# create_indexes.py  — run once before using the chatbot
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION     = os.getenv("COLLECTION", "pdfs-store")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Keyword indexes (for MatchValue / MatchAny filters)
keyword_fields = [
    "contract_name",
    "contract_id",
    "part",
    "category",
]

# Boolean indexes (for has_* clause flags)
bool_fields = [
    "has_governing_law",
    "has_termination",
    "has_cap_liability",
    "has_non_compete",
    "has_indemnification",
    "has_arbitration",
    "has_ip_ownership",
    "has_exclusivity",
]

# Integer indexes (for page filtering later)
int_fields = [
    "page_start",
    "page_end",
    "chunk_index",
]

for field in keyword_fields:
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"✓ keyword index: {field}")

for field in bool_fields:
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=PayloadSchemaType.BOOL,
    )
    print(f"✓ bool index: {field}")

for field in int_fields:
    qdrant.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=PayloadSchemaType.INTEGER,
    )
    print(f"✓ integer index: {field}")

print("\nAll indexes created. You can now run the chatbot.")