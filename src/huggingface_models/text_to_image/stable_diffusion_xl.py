import torch
from diffusers import StableDiffusionXLPipeline

from src.huggingface_models.base_strategy import GenerativModelStrategy


class StableDiffusionXLModel(GenerativModelStrategy):

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 cache_dir: str,
                 num_inference_steps: int = 50,
                 guidance_scale : float = 7.0,
                 compile_pipeline : bool = False,
                 model: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model = StableDiffusionXLPipeline.from_pretrained(model,
                                                               torch_dtype=dtype,
                                                               cache_dir=cache_dir,
                                                               use_safetensors=True)
        self.model.to(device=device)
        if compile_pipeline:
            self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)

    def generate(self,
                 noise_emds: torch.Tensor,
                 prompt: str):

        image = self.model(prompt=prompt,
                           latents =noise_emds,
                           output_type="pil",
                           num_inference_steps= self.num_inference_steps,
                           guidance_scale = self.guidance_scale).images[0]
        return image

    def generate_batch(self,
                       noise_emds: list[torch.Tensor],
                       prompt: str):

        images = self.model(prompt=[prompt]*len(noise_emds),
                           latents=noise_emds,
                           output_type="pil",
                           num_inference_steps=self.num_inference_steps,
                           guidance_scale=self.guidance_scale).images
        return images

class StableDiffusionXLRefinerStrategy(StableDiffusionXLModel):

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 cache_dir: str,
                 compile_pipeline: bool = False,
                 num_inference_steps: int = 50,
                 guidance_scale : float = 7.0,
                 high_noise_frac=0.8,
                 model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner : str = "stabilityai/stable-diffusion-xl-refiner-1.0"):
        super().__init__(device,
                         dtype,
                         cache_dir,
                         num_inference_steps,
                         guidance_scale,
                         compile_pipeline,
                         model)
        self.high_noise_frac = high_noise_frac
        self.refiner = StableDiffusionXLPipeline.from_pretrained(refiner,
                                                                 text_encoder_2 = self.model.text_encoder_2,
                                                                 vae = self.model.vae,
                                                                 torch_dtype=dtype,
                                                                 use_safetensors=True,
                                                                 cache_dir=cache_dir,)
        self.refiner.to(device=device)
        if compile_pipeline:
            self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    def generate(self,
                 noise_emds: torch.Tensor,
                 prompt: str):
        image = self.model(prompt=prompt,
                           latents=noise_emds,
                           output_type="latent",
                           denoising_end=self.high_noise_frac,
                           num_inference_steps=self.num_inference_steps,
                           guidance_scale=self.guidance_scale
                           ).images
        image = self.refiner(prompt=prompt,
                             num_inference_steps=self.num_inference_steps,
                             denoising_start=self.high_noise_frac,
                             image=image
                             ).images[0]
        return image


    def generate_batch(self,
                       noise_emds: list[torch.Tensor],
                       prompt: str):
        images = self.model(prompt=[prompt]*len(noise_emds),
                           latents=noise_emds,
                           output_type="latent",
                           denoising_end=self.high_noise_frac,
                           num_inference_steps=self.num_inference_steps,
                           guidance_scale=self.guidance_scale
                           ).images

        image = self.refiner(prompt=prompt,
                             num_inference_steps=self.num_inference_steps,
                             denoising_start=self.high_noise_frac,
                             image=images
                             ).images
        return image