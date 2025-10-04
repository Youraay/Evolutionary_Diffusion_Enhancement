from dataclasses import dataclass

import torch

from .base_crossover import CrossoverFunction


@dataclass
class ArithmeticCrossoverFunction(CrossoverFunction):
    alpha: float = 0.5


    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Perform weighted arithmetic crossover between two parent tensors.
        Args:
            parent_1 (torch.Tensor): [C,H,W] First parent tensor
            parent_2 (torch.Tensor): [C,H,W] Second parent tensor
            *args: Additional positional arguments (not used)
            **kwargs: Additional keyword arguments with optional 'weight' parameter

        Returns:
            torch.Tensor: [C,H,W] Child tensor created with weighted sum of parents

        """
        alpha = kwargs.pop("weight", self.weight)
        parent_2 = parent_2.to(parent_1.device)
        return alpha * parent_1 + (1 - alpha) * parent_2


