import os
from pathlib import Path
from typing import Any, List
from torchkde import KernelDensity
import torch
import numpy as np
from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
from src.utils.pca import PCA

class KernelDensityEstimationEvaluator(Evaluator):
    def __init__(self,
                 prompt: str,
                 rule_of_thumb: str = 'silverman',
                 bandwidth: float  | None = None,
                 kernel : str ='gaussian',
                 K : int = 128



    ) -> None:
        self.kernel: str = kernel
        self.prompt: str = prompt
        self.K : int = K

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
        data_stack = data_stack.to(torch.float32)

        self.pca = PCA(data_stack)

        self.reduced_data_stack = self.pca.reduce_embeddings(data_stack, self.K)

        if bandwidth is None:
            print(self.reduced_data_stack.shape)
            N = self.reduced_data_stack.size(0)
            print(N)
            bandwidth = self.calculate_bandwidth(
                self.reduced_data_stack,
                N, 
                rule_of_thumb
                )
        
        print(bandwidth)
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kde.to(self.reduced_data_stack.device)
        self.kde.fit(self.reduced_data_stack)
        self.name = "KernelDensityEstimation"


    def calculate_bandwidth(self, embeddings: torch.Tensor, N: int, rule_of_thumb: str = 'silverman'):
        embeddings = embeddings.cpu()
        std_dims = torch.std(embeddings, dim=0)
        sigma_median = np.median(std_dims.numpy())

        N_exponent = N**(-1/5)

        if rule_of_thumb == 'silverman':
            h = 0.9 * sigma_median * N_exponent
        else: 
            h = 1.06 * sigma_median * N_exponent
        
        return h

    def evaluate(self, embeds: torch.Tensor, *args, **kwargs) -> dict[str, float]:

        embeds = embeds.to(self.reduced_data_stack.dtype).to(self.reduced_data_stack.device)
        reduced__embeds = self.pca.reduce_embeddings(embeds, self.K)
        log_prob = self.kde.score_samples(reduced__embeds).item()
        return {"name": self.name,
                "score" : log_prob}

    def evaluate_batch(self, embeds: List[torch.Tensor], *args, **kwargs) -> list[dict[str,float]]:

        print(self.reduced_data_stack.size())
        stack = torch.stack(embeds)
        stack = stack.to(self.reduced_data_stack.dtype).to(self.reduced_data_stack.device)
        print(embeds[0].size())
        print(stack.size())
        reduced__stack = self.pca.reduce_embeddings(stack, self.K)
        log_prob = self.kde.score_samples(reduced__stack)
        novelty_scores: List[float] = log_prob.tolist()

        return [{"name": self.name, "score": log_prob} for log_prob in novelty_scores]



    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel
