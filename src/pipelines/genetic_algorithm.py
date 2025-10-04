from .pipeline import Pipeline
from ..crossover.base_crossover import CrossoverFunction
from ..evaluators.base_evaluator import Evaluator
from ..factory import NoiseFactory
from src.huggingface_models.base_strategy import GenerativModelStrategy, EmbeddingModelStrategy, \
    CaptionModelStrategy
from ..mutators.base_mutator import MutationFunction
from ..selectors.base_selector import SelectorFunction


class GeneticAlgorithmPipeline(Pipeline):

    def __init__(self,
               generative_model : GenerativModelStrategy,
               embedding_model : EmbeddingModelStrategy,
               crossover_operation : CrossoverFunction,
               noise_factory : NoiseFactory,
               mutator: MutationFunction,
               selector : SelectorFunction,
               evaluator : Evaluator,
               prompt : str,
               num_generations: int,
               population_size: int,
               initial_mutation_rate: float,
               initial_crossover_rate: float,
               elite_size: int = 0,
               caption_model : CaptionModelStrategy = None,
               ):
        pass

    def run(self):
        pass