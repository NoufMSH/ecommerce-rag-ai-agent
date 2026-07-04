"""
Embedding index builder.

Reads data/processed/knowledge_base.jsonl (produced by etl_pyspark.py),
embeds each text chunk with a small local sentence-transformers model
(no API key needed), and saves the vectors + metadata for the agent to
search over with plain NumPy cosine similarity.

Run with:
    python src/build_index.py
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

KB_PATH = os.path.join("data", "processed", "knowledge_base.jsonl")
INDEX_DIR = "index"
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs locally on CPU


def load_chunks():
    chunks = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    # normalize_embeddings=True lets us use a plain dot product as cosine similarity later
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, np.asarray(embeddings, dtype=np.float32))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved embeddings {embeddings.shape} to {EMBEDDINGS_PATH}")
    print(f"Saved metadata for {len(chunks)} chunks to {METADATA_PATH}")


if __name__ == "__main__":
    main()
