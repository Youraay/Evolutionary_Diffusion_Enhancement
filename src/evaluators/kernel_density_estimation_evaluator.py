import os
from pathlib import Path
from typing import Any, List
from torchkde import KernelDensity
import torch

from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
from src.utils.pca import PCA

class KernelDensityEstimationEvaluator(Evaluator):
    def __init__(self,
                 prompt: str,
                 bandwidth: float  =1.0,
                 kernel : str ='gaussian',
                 K : int =128



    ) -> None:
        self.bandwidth: float = bandwidth
        self.kernel: str = kernel
        self.prompt: str = prompt
        self.K = K
        base_path = Path(os.environ['BASE_PATH'])
        metrics_path = Path(os.environ['BLIP_2_BASELINE'])
        self.metric_path  = base_path / metrics_path / self.prompt.replace(" ", "_")
        print(self.metric_path)
        glob = self.metric_path.glob("*.pt")
        data =[]
        for p in glob:
            dp =torch.load(
                str(p),
                map_location=torch.device("cuda"),
                weights_only=False,
                )
            data.append(dp.pooler_output)
        

        if len(data) == 0:
            raise FileNotFoundError(f"No metric found at {self.metric_path}")


        data_stack = torch.cat(data, dim=0)

        self.pca = PCA(self.data+stack)
        self.reduced_data_stack = PCA.reduce_embeddings(data_stack, self.K)
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kde.fit(self.reduced_data_stack)
        self.name = "KernelDensityEstimation"

    def evaluate(self, embeds: torch.Tensor, *args, **kwargs) -> dict[str, float]:

        log_prob = self.kde.score_samples(embeds).item()
        return {"name": self.name,
                "score" : log_prob}

    def evaluate_batch(self, embeds: List[torch.Tensor], *args, **kwargs) -> list[dict[str,float]]:

        print(self.reduced_data_stack.size())
        stack = torch.stack(embeds)
        print(embeds[0].size())
        print(stack.size())
        self.reduced__stack = PCA.reduce_embeddings(stack, self.K)
        log_prob = self.kde.score_samples(reduced__stack)
        novelty_scores: List[float] = log_prob.tolist()

        return [{"name": self.name, "score": log_prob} for log_prob in novelty_scores]



    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel
