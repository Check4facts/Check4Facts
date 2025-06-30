import ollama
import numpy as np


# class to compute embeddings using Ollama. If anything else is needed, the class can be extended.
class OllamaEmbeddings:
    def __init__(self):
        self.model_name = "paraphrase-multilingual"

    def compute_embedding(self, text):
        if not text.strip():
            return []
        response = ollama.embeddings(model=self.model_name, prompt=text)
        embedding = response.get("embedding", [])
        if not embedding:
            return None

        return np.array(embedding)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
