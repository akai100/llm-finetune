import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(docs, index_path):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = encoder.encode(docs, show_progress_bar=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)
