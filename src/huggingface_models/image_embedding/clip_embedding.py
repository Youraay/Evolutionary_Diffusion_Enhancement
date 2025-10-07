import torch

from src.huggingface_models.base_strategy import EmbeddingModelStrategy


class ClipEmbeddingModel(EmbeddingModelStrategy):
    def embed(self) -> torch.Tensor:
        pass

    def embed_batch(self) -> torch.Tensor:
        pass