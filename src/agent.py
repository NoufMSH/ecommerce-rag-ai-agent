"""
ProductQAAgent: the RAG agent.

Retrieval: embed the user's question with the same local sentence-transformers
model used to build the index, then rank stored chunks by cosine similarity
(plain NumPy dot product, since embeddings are normalized).

Augmentation: stuff the top matching chunks into a prompt as grounding context.

Generation: send that prompt to Google's Gemini API (free tier) and return
the answer, along with the chunks that were used, so the caller can show
"sources" for transparency.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_DIR = "index"
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are a helpful e-commerce customer support assistant. "
    "Answer the customer's question using ONLY the context provided below. "
    "If the context doesn't contain enough information to answer, say you "
    "don't have that information instead of guessing. Be concise and friendly."
)


class ProductQAAgent:
    def __init__(self):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embeddings = np.load(EMBEDDINGS_PATH)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your free "
                "Gemini API key from https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def list_products(self):
        """Return sorted unique (product_id, product_name) pairs, for a UI dropdown."""
        seen = {}
        for chunk in self.chunks:
            seen[chunk["product_id"]] = chunk["product_name"]
        return sorted(seen.items())

    def retrieve(self, question, product_id=None, top_k=4):
        """Return the top_k most relevant chunks for the question.

        If product_id is given, only search within that product's own chunks
        (there are only 4 per product, so this just re-ranks them by
        relevance to the question).
        """
        query_vector = self.embedder.encode([question], normalize_embeddings=True)[0]

        if product_id:
            candidate_indices = [i for i, c in enumerate(self.chunks) if c["product_id"] == product_id]
        else:
            candidate_indices = list(range(len(self.chunks)))

        candidate_embeddings = self.embeddings[candidate_indices]
        scores = candidate_embeddings @ query_vector  # cosine similarity (vectors are normalized)

        ranked = sorted(zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True)
        top_indices = [idx for idx, _score in ranked[:top_k]]
        return [self.chunks[i] for i in top_indices]

    def build_prompt(self, question, chunks):
        context = "\n".join(f"{i + 1}. [{c['doc_type']}] {c['text']}" for i, c in enumerate(chunks))
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Context:\n{context}\n\n"
            f"Customer question: {question}\n"
            f"Answer:"
        )

    def answer(self, question, product_id=None):
        chunks = self.retrieve(question, product_id=product_id)
        prompt = self.build_prompt(question, chunks)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            answer_text = response.text
        except Exception as exc:  # course-project-level error handling: fail gracefully, not silently
            answer_text = f"Sorry, I couldn't reach the language model right now ({exc})."

        return {"answer": answer_text, "sources": chunks}


if __name__ == "__main__":
    # Quick manual test without Streamlit.
    agent = ProductQAAgent()
    print("Products:", agent.list_products())

    result = agent.answer("Is this product available?", product_id="P004")
    print("\nQ: Is this product available? (Boom Mini speaker)")
    print("A:", result["answer"])
    print("Sources used:", [s["doc_type"] for s in result["sources"]])
