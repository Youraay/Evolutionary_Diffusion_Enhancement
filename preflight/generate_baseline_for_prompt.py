import logging
import os
from pathlib import Path
from typing import Any, Generator

from dotenv import load_dotenv
from src.factorys import NoiseFactory
from src.huggingface_models import ModelLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_batches(self, embeds: list[Any]) -> Generator[list[Any], None, None]:
    num_samples = len(embeds)
    for i in range(0, num_samples, self.batch_size):
        # Schneidet die Liste der Tensoren in Chunks der Größe batch_size
        yield embeds[i:i + self.batch_size]


def generate_baseline_for_prompt(
        prompt : str,
        count : int,
        batch_size: int= 10,
) -> None:

    base_path = Path(os.environ.get("BASE_PATH"))
    blip_2_path = base_path / Path(os.environ.get("BLIP_2_BASELINE", "")) / prompt.replace(" ", " ")
    clip_path = base_path / Path(os.environ.get("CLIP_BASELINE", "")) / prompt.replace(" ", " ")
    sdxl_path = base_path / Path(os.environ.get("SDXL_BASELINE", "")) / prompt.replace(" ", " ")
    model_loader = ModelLoader(cache_dir=os.environ.get("HF_CACHE", ""))
    # logger.info("Blip: ", blip_2_path)
    # logger.info("CLIP: ", clip_path)
    logger.info(f"SDXL {sdxl_path}")
    logger.info(f"")
    sdxl = model_loader.load_sdxl()
    blip2 = model_loader.load_blip2_embeddings()
    clip = model_loader.load_clip_embeddings()

    nf = NoiseFactory(device=model_loader.device, dtype= model_loader.dtype)
    population = nf.create_batch(count)
    logger.info(f"Generating baseline for prompt {prompt}")
    logger.info(f"Set of {len(population)} noises")


    img_id = 0


    pils = []
    for batch in create_batches([candidate.initial_noise for candidate in population]):
        pil_images = sdxl.generate_batch(batch, prompt)
        pils.extend(pil_images)

    blip2_e = []
    for batch in create_batches(pils):
        batch_embeddings = blip2.batch_qformer_feature_extraction(batch)
        blip2_e.extend(batch_embeddings)

    clip_e = []
    for batch in create_batches(pils):
        batch_embeddings = clip.batch_image_features_extraction(batch)
        clip_e.extend(batch_embeddings)

load_dotenv()
generate_baseline_for_prompt(
    "a_chair",
    100
)


