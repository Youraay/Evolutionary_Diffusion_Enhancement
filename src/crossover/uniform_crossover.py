from dataclasses import dataclass

import torch

from src.crossover.base_crossover import CrossoverFunction

@dataclass
class UniformCrossover(CrossoverFunction):

    swap_rate: float = 0.5

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            Perform uniform crossover between parent_1 and parent_2 tensors.

            Args:
                parent_1 (torch.Tensor): [C,H,W] First parent tensor
                parent_2 (torch.Tensor): [C,H,W] Second parent tensor
                *args: Additional positional arguments (not used)
                **kwargs (float): Additional keyword arguments with optional 'swap_rate' parameter

            Returns:
                torch.Tensor: [C,H,W] Child tensor created with weighted mix of parents
            """
        assert parent_1.shape == parent_2.shape, "Noises must have the same size for crossover."

        swap_rate = kwargs.pop("swap_rate", self.swap_rate)
        crossover_mask = torch.rand(parent_1) < swap_rate
        child = torch.where(crossover_mask, parent_1, parent_2)

        return child


