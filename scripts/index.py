#!/usr/bin/env python3
"""
Index a document into Qdrant chunks_collection.
Supports .md, .txt, and .pdf files.
Run once before starting the API, or via POST /index/rebuild.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import CHUNKS_COLLECTION, EMBEDDING_DIM, QDRANT_API_KEY, QDRANT_URL

# Auto-detect: use .pdf if present, otherwise .md
_data_dir = Path(__file__).parent.parent / "data"
_pdf = next(_data_dir.glob("*.pdf"), None)
DATA_PATH = _pdf if _pdf else _data_dir / "source.md"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_enc = tiktoken.get_encoding("cl100k_base")


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        print(f"  PDF: {len(reader.pages)} pages extracted")
        return text
    return path.read_text(encoding="utf-8")


def token_count(text: str) -> int:
    return len(_enc.encode(text))


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {DATA_PATH} ...")
    text = _read_document(DATA_PATH)
    total_chars = len(text)
    total_tokens = token_count(text)
    print(f"  {total_chars:,} chars | ~{total_tokens:,} tokens")

    print("Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_count,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"  {len(chunks)} chunks")

    print("Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding ...")
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client.collection_exists(CHUNKS_COLLECTION):
        print(f"  Dropping existing collection '{CHUNKS_COLLECTION}' ...")
        client.delete_collection(CHUNKS_COLLECTION)

    client.create_collection(
        collection_name=CHUNKS_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection '{CHUNKS_COLLECTION}'")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "chunk_id": f"chunk_{i}",
                "text": chunk,
                "token_count": token_count(chunk),
                "source_file": str(DATA_PATH.name),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    client.upsert(collection_name=CHUNKS_COLLECTION, points=points, wait=True)
    indexed_tokens = sum(token_count(c) for c in chunks)
    print(f"\nDone. Indexed {len(chunks)} chunks | {indexed_tokens:,} tokens")


if __name__ == "__main__":
    main()
