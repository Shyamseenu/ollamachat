"""
ingest.py — document ingestion into the Chroma knowledge_base collection.

Changes from the original version:
  - Supports .pdf, .txt, and .md (previously PDF-only).
  - Refactored into importable functions (`ingest_file`) so app.py can call
    this directly from the /documents/upload endpoint, not just the CLI.
  - Every chunk is tagged with a `user_id` metadata field so retrieval can
    be scoped per-user (users only ever query their own uploads).
  - Still fully usable from the command line for bulk/manual ingestion.
  - Embeddings now use Google's Gemini embedding API instead of a local
    SentenceTransformer model, to avoid loading a large ML model into
    memory (important on memory-constrained deploys like Render's free tier).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
import fitz  # PyMuPDF
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_PATH = Path(__file__).parent / "chroma_db"
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

_chroma_client = None
_kb_collection = None


def get_kb_collection():
    """Lazily create a single shared PersistentClient/collection.
    Reused by app.py so we don't open the sqlite file twice."""
    global _chroma_client, _kb_collection
    if _kb_collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set — required for Gemini embeddings in ingest.py"
            )

        embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name=GEMINI_EMBEDDING_MODEL,
            task_type="RETRIEVAL_DOCUMENT",
        )
        _kb_collection = _chroma_client.get_or_create_collection(
            name="knowledge_base", embedding_function=embedding_fn
        )
    return _kb_collection


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def extract_text_from_pdf(path: Path) -> List[Dict]:
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_text_from_plain(path: Path) -> List[Dict]:
    """TXT / MD — no real page concept, so the whole file is 'page 1'."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"page": 1, "text": text}] if text else []


def extract_pages(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".txt", ".md", ".markdown"):
        return extract_text_from_plain(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def chunk_pages(pages: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = []
    for p in pages:
        for j, split in enumerate(splitter.split_text(p["text"])):
            chunks.append({"text": split, "page": p["page"], "chunk_id": j})
    return chunks


def ingest_file(
    path: Path,
    user_id: str,
    chunk_size: int = 500,
    overlap: int = 50,
    clear_existing: bool = True,
) -> int:
    """
    Ingest a single file into the shared Chroma collection, scoped to
    `user_id` via metadata. IDs are namespaced by user_id so two users
    uploading a file with the same name never collide.

    Returns the number of chunks stored.
    Raises ValueError for unsupported file types or empty documents.
    """
    kb_collection = get_kb_collection()

    if clear_existing:
        # Re-uploading the same filename replaces the old vectors, scoped
        # to this user only — never touches other users' documents.
        kb_collection.delete(where={"$and": [{"source": path.name}, {"user_id": user_id}]})

    pages = extract_pages(path)
    if not pages:
        raise ValueError(f"No extractable text found in {path.name}")

    chunks = chunk_pages(pages, chunk_size, overlap)

    batch_size = 100
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        kb_collection.upsert(
            ids=[f"{user_id}_{path.stem}_p{c['page']}_c{c['chunk_id']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": path.name,
                    "page": c["page"],
                    "chunk_id": c["chunk_id"],
                    "user_id": user_id,
                }
                for c in batch
            ],
        )
        total += len(batch)

    return total


def delete_file(filename: str, user_id: str) -> None:
    """Remove all vectors for a given filename, scoped to this user."""
    kb_collection = get_kb_collection()
    kb_collection.delete(where={"$and": [{"source": filename}, {"user_id": user_id}]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document (PDF/TXT/MD) into the Chroma knowledge base.")
    parser.add_argument("--file", required=True, help="Path to file")
    parser.add_argument("--user", required=True, help="User id/username to scope this document to")
    parser.add_argument("--chunk", type=int, default=500, help="Chunk size (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap (default: 50)")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing vectors for this filename first")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print("File not found: " + args.file)
        sys.exit(1)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    print("Ingesting: " + file_path.name + " for user=" + args.user)
    try:
        n = ingest_file(file_path, args.user, args.chunk, args.overlap, clear_existing=not args.no_clear)
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)

    print(f"  Stored {n} vectors in Chroma knowledge_base")
    print("  Chroma DB path: " + str(CHROMA_PATH.resolve()))
    print("Ingestion complete.")