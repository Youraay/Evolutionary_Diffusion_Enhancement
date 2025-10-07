import os
from pathlib import Path
from typing import Any, List
from torchkde import KernelDensity
import torch

from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel


class KernelDensityEstimationEvaluator(Evaluator):
    def __init__(self,
                 prompt: str,
                 bandwidth: float  =1.0,
                 kernel : str ='gaussian',



    ) -> None:
        self.bandwidth: float = bandwidth
        self.kernel: str = kernel
        self.prompt: str = prompt
        base_path = Path(os.environ['BASE_PATH'])
        metrics_path = Path(os.environ['BLIP_2_BASELINE'])
        self.metric_path  = base_path / metrics_path / self.prompt.replace(" ", "_")
        data = [torch.load(p)  for p in self.metric_path.glob("*.pt")]
        if len(data) == 0:
            raise FileNotFoundError(f"No metric found at {self.metric_path}")
        data_stack = torch.cat(data, dim=0)
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kde.fit(data_stack)
        self.name = "KernelDensityEstimation"

    def evaluate(self, embeds: torch.Tensor, *args, **kwargs) -> dict[str, float]:

        log_prob = self.kde.score_samples(embeds).item()
        return {"name": self.name,
                "score" : log_prob}

    def evaluate_batch(self, embeds: List[torch.Tensor], *args, **kwargs) -> list[dict[str,float]]:

        stack = torch.cat(embeds, dim=0)
        log_prob = self.kde.score_samples(stack)
        novelty_scores: List[float] = log_prob.tolist()

        return [{"name": self.name, "score": log_prob} for log_prob in novelty_scores]



    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel
