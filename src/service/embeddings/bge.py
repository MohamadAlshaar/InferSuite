from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    Local (offline) BAAI BGE embedder.
    - model_path must be a local directory.
    - normalize=True makes vectors unit-length, so COSINE similarity is well-behaved.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_path = model_path
        self.device = device
        self.normalize = normalize
        self.batch_size = int(batch_size)

        self._model = SentenceTransformer(model_path, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> List[float]:
        v = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return v[0].astype(np.float32).tolist()

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        v = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return v.astype(np.float32).tolist()
