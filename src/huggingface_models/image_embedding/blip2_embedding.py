import torch
from PIL import Image

from src.huggingface_models.base_strategy import EmbeddingModelStrategy
from transformers import Blip2Model, AutoProcessor

class Blip2EmbeddingModel(EmbeddingModelStrategy):

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 cache_dir: str,
                 model: str = "Salesforce/blip2-opt-2.7b") -> None:


        self.device = device
        self.model = Blip2Model.from_pretrained(model,
                                                torch_dtype=dtype,
                                                cache_dir=cache_dir,
                                                use_safetensors=True,
                                                device_map="auto")

        self.processor = AutoProcessor.from_pretrained(model)


    def embed(self, pixel_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pixel_image, return_tensors="pt").to(
            self.device)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)


            return image_embeds

    def embed_batch(self, pixel_images: list[Image.Image]) -> list[torch.Tensor]:

        inputs = self.processor(images=pixel_images, return_tensors="pt").to(
            self.device)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)

        image_embeds = [ image_embeds[i] for i in range(pixel_images)]
        return image_embeds

