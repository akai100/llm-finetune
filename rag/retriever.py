import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_path)

    def retrieve(self, query, k=5):
        q = self.encoder.encode([query])
        scores, ids = self.index.search(q, k)
        return ids[0]

