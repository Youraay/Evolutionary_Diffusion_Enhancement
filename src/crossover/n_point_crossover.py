from dataclasses import dataclass

import torch
from sympy.physics.units import length

from src.crossover.base_crossover import CrossoverFunction

@dataclass
class NPointRandomCrossoverFunction(CrossoverFunction):

    n_points: int = 3

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
            Perform uniform crossover between parent_1 and parent_2 tensors.

            Args:
                parent_1 (torch.Tensor): [C,H,W] First parent tensor
                parent_2 (torch.Tensor): [C,H,W] Second parent tensor
                *args: Additional positional arguments (not used)
                **kwargs (float): Additional keyword arguments with optional 'n_point' parameter

            Returns:
                torch.Tensor: [C,H,W] Child tensor created with n+1 random sized segment mix of parents
            """
        assert parent_1.shape == parent_2.shape, "Parents must have the same size for crossover."

        n_point : int = kwargs.pop("n_point", self.n_points)
        C,H,W = parent_1.shape
        length = H * W

        # Flatten spartial dims, keeping channels separate
        parent_1_flat = parent_1.reshape(C, length)
        parent_2_flat = parent_2.reshape(C, length)
        child_flat = torch.empty_like(parent_1_flat)

        points = torch.sort(torch.randint(low=1, high=length, size=(n_point,)))[0]

        start = 0
        take_from_parent_1 = True
        for point in points:
            if take_from_parent_1:
                child_flat[:, start:point] = parent_1_flat[:, start:point]
            else:
                child_flat[:, start:point] = parent_2_flat[:, start:point]
            start = point
            take_from_parent_1 = not take_from_parent_1

        if take_from_parent_1:
            child_flat[:, start:] = parent_1_flat[:, start:]
        else:
            child_flat[:, start:] = parent_2_flat[:, start:]

        return child_flat.reshape(C, H, W)


@dataclass
class NPointEqualCrossoverFunction(CrossoverFunction):

    n_points: int = 3

    def crossover(self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """

        Args:
        parent_1 (torch.Tensor): [C,H,W] First parent tensor
        parent_2 (torch.Tensor): [C,H,W] Second parent tensor
        *args: Additional positional arguments (not used)
        **kwargs (float): Additional keyword arguments with optional 'n_point' parameter

        Returns: torch.Tensor: [C,H,W] Child tensor created with n+1 equally large segment mix of parents
        """

        n_points: int = kwargs.pop("n_point", self.n_point)
        C, H, W = parent_1.shape
        length = H * W

        # Flatten spartial dims, keeping channels separate
        parent_1_flat = parent_1.reshape(C, length)
        parent_2_flat = parent_2.reshape(C, length)
        child_flat = torch.empty_like(parent_1_flat)

        segment_length = length // n_points

        for i in range(n_points):
            start = i * segment_length

            # last segment takes the remainder, if not exactly divisible
            end = (i + 1) * segment_length if i < n_points - 1 else length

            if i % 2 == 0:
                child_flat[:, start:end] = parent_1_flat[:, start:end]
            else :
                child_flat[:, start:end] = parent_2_flat[:, start:end]

        return child_flat.reshape(C, H, W)