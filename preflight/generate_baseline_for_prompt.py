import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.factorys import NoiseFactory
from src.huggingface_models import ModelLoader

logger = logging.getLogger(__name__)

def generate_baseline_for_prompt(
        prompt : str,
        count : int,
        batch_size: int= 10,
) -> None:

    base_path = Path(os.environ.get("BASE_PATH"))
    blip_2_path = Path(os.environ.get("BLIP_2_BASELINE", "")) / prompt.replace(" ", " ")
    clip_path = Path(os.environ.get("CLIP_BASELINE", "")) / prompt.replace(" ", " ")
    sdxl_path = Path(os.environ.get("SDXL_BASELINE", "")) / prompt.replace(" ", " ")
    model_loader = ModelLoader()
    logger.info(blip_2_path)
    logger.info(clip_path)
    logger.info(sdxl_path)
    logger.info(f"")
    sdxl = model_loader.load_sdxl()
    blip2 = model_loader.load_blip2_embeddings()
    clip = model_loader.load_clip_embeddings()

    nf = NoiseFactory(device=model_loader.device, dtype= model_loader.dtype)
    population = nf.create_batch(count)
    logger.info(f"Generating baseline for prompt {prompt}")
    logger.info(f"Set of {len(population)} noises")


    img_id = 0
    for noise in population:

        logger.info(noise.initial_noise.size)
        noise.pil_image = sdxl.generate(noise.initial_noise, prompt)
        noise.blip2_embedding = blip2.embed(noise.pil_image)
        noise.clip_embedding = clip.embed(noise.pil_image)

        noise.save_pil_image(sdxl_path / f"{img_id:4d}.png")
        noise.save_blip2(blip_2_path / f"{img_id:4d}.png")
        noise.save_clip(clip_path / f"{img_id:4d}.png")


load_dotenv()
generate_baseline_for_prompt(
    "a_cat",
    1000
)


