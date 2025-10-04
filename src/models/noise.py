from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from typing import Optional, Dict

@dataclass
class Noise:
    """
    Attributes:
    id (str): Unique identifier of the noise instance
    initial_noise (Optional[torch.Tensor]): Initial Gaussian noise
    initial_seed (int): Initial seed
    latent_representation (Optional[torch.Tensor]): Latent embedding of the generated image
    pil_image (Optional[PIL.Image.Image]): PIL image of the generated image
    blip2_embedding (Optional[torch.Tensor]): BLIP2 Image Encoder embedding vector of the generated image
    clip_embedding (Optional[torch.Tensor]): CLIP embedding vector of the generated image
    fitness (Optional[float]): Fitness score of the generated image
    evaluation_scores (Dict[str, float]): Individual evaluation scores from the evaluators
    start_generation (Optional[int]): Generation number of the first appearance of the noise instance
    end_generation (Optional[int]): Generation number of the last appearance of the noise instance
    """

    id: str
    initial_noise : torch.Tensor
    initial_seed : int = None
    latent_representation: torch.Tensor = None
    pil_image : Image.Image = None
    blip2_embedding : torch.Tensor = None
    clip_embedding : torch.Tensor = None
    fitness: float = None
    evaluation_scores: dict[str, float] = field(default_factory=dict)
    start_generation: int = None
    end_generation: int = None
    parent_1: "Noise" = None
    parent_2: "Noise" = None
    crossover: bool = None
    mutate: bool = None

    def _save_pil(self,
                  pil_image: Image.Image,
                  filepath: str,
                  file_format : str = "JPEG",
                  ) -> None:
        path = Path(filepath)
        fitness =  0.0 if self.fitness is None else self.fitness
        filename = Path(f"image_g{self.end_generation}_{self.id}_f{fitness}.{file_format}")
        full_path = path / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if pil_image is not None:
            pil_image.save(full_path, format=file_format)
        else:
            raise ValueError()

    def save_pil_image(self,
                       filepath: str,
                       file_format : str = "JPEG",
                       ) -> None:

        try:
            self._save_pil(pil_image=self.pil_image,
                           filepath=filepath,
                           file_format=file_format)
        except ValueError as e:
            raise ValueError(f"No pil image available for noise {self.id}") from e


    def save_noise_to_rgb(self,
                          filepath: str,
                          file_format : str = "JPEG",
                          ) -> None:

        try:
            weights = (
                (60, -60, 25, -70),
                (60, -5, 15, -50),
                (60, 10, -5, -35),
            )

            weights_tensor = torch.t(torch.tensor(weights, dtype=self.initial_noise.dtype)
                                     .to(self.initial_noise.device))
            biases_tensor = (torch.tensor((150, 140, 130), dtype=self.initial_noise.dtype)
                             .to(self.initial_noise.device))
            rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", self.initial_noise, weights_tensor) \
                         + biases_tensor.unsqueeze(-1).unsqueeze(-1)

            # Handle batch dimension - squeeze if batch size is 1
            if rgb_tensor.dim() == 4 and rgb_tensor.shape[0] == 1:
                rgb_tensor = rgb_tensor.squeeze(0)  # Remove batch dimension: [1, 3, H, W] -> [3, H, W]

            # Ensure we have [C, H, W] format before transpose to [H, W, C]
            if rgb_tensor.dim() == 3:
                image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected tensor shape: {rgb_tensor.shape}. Expected [C, H, W] or [1, C, H, W]")

            image = Image.fromarray(image_array)
        except Exception as e:
            raise Exception(f"Failed to convert initial noise to rgb image: {e}")

        try:
            self._save_pil(pil_image=image,
                           filepath=filepath,
                           file_format=file_format)
        except ValueError as e:
            raise ValueError(f"Error while saving image to RGB: {e}")


    def save_noise(self,
                 filepath: str,
    ) -> None:
        torch.save(self.latent_representation, f"{filepath}/{self.id}")

    def save_representation(self,
                 filepath: str,
    ) -> None:
        torch.save(self.latent_representation, f"{filepath}/img_{self.id}")

    def save_blip2(self,
                 filepath: str,
    ) -> None:
        torch.save(self.blip2_embedding, f"{filepath}/blip2_{self.id}")

    def save_clip(self,
                 filepath: str,
    ) -> None:
        torch.save(self.clip_embedding, f"{filepath}/clip_{self.id}")