from abc import ABC, abstractmethod
import torch
from typing import List

from PIL import Image


class BaseStrategy(ABC):
    pass

class GenerativModelStrategy(BaseStrategy, ABC):

    @abstractmethod
    def generate(self,
                 noise_emds: torch.Tensor,
                 prompt: str):
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self,
                       noise_emds: List[torch.Tensor],
                       prompt: str):
        raise NotImplementedError

class EmbeddingModelStrategy(BaseStrategy, ABC):

    @abstractmethod
    def embed(self, pixel_image : Image.Image ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, pixel_images : list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

class CaptionModelStrategy(BaseStrategy, ABC):

    def caption(self, pixel_image : Image.Image) -> str:
        raise NotImplementedError

    def caption_batch(self, pixel_images : list[Image.Image]) -> List[str]:
        raise NotImplementedError