"""Batch-ingest every .txt file in data/ via the running POST /ingest endpoint.

Run (with the API already running, locally or via API_BASE_URL for the live Render URL):
  python ingest_corpus.py
  API_BASE_URL=https://your-service.onrender.com python ingest_corpus.py
"""

import os
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent / "data"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def main():
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {DATA_DIR}")
        return

    total_chunks = 0
    for path in files:
        document_id = path.stem  # stable id derived from the filename, e.g. "bitcoin_overview"
        text = path.read_text(encoding="utf-8")

        response = httpx.post(
            f"{API_BASE_URL}/ingest",
            json={"text": text, "document_id": document_id, "source": path.name},
            timeout=60.0,
        )
        response.raise_for_status()
        chunks_indexed = response.json()["chunks_indexed"]
        total_chunks += chunks_indexed
        print(f"{document_id}: {chunks_indexed} chunks indexed")

    # Pinecone's stats endpoint is eventually consistent, so give it a moment before checking.
    time.sleep(3)
    stats = httpx.get(f"{API_BASE_URL}/health/pinecone", timeout=30.0).json()
    print(f"\nIngested {total_chunks} chunks across {len(files)} documents this run.")
    print(f"Vector store now reports {stats['total_vectors']} total vectors.")


if __name__ == "__main__":
    main()
