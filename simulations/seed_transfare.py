import os
import sys
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from functools import partial

from crossover import UniformCrossover
from evaluators.global_max_mean_divergence_evaluator import GlobalMaxMeanDivergenceEvaluator
from evaluators.kernel_density_estimation_evaluator import KernelDensityEstimationEvaluator
from factorys import NoiseFactory
from huggingface_models import ModelLoader
from mutators.uniform_gaussian_mutator import UniformGaussianMutator
from pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from selector_functions.tournament_selector import TournamentSelector
from src.utils.arg_parser import args
from src.utils.loader import TensorLoader

def main(experiment_id: str = "1549660"):

    base_path = Path(os.environ.get("BASE_PATH"))


    prompt = "a cat"
    new_prompt = "a dog"
    nf = NoiseFactory(device="cpu")

    get_generation = partial(
        nf.create_from_files,
        base_path=base_path,
        prompt=prompt.replace(" ", "_"),
        experiment_id=1549660  # Optional: Bei der Verwendung von Positional Arguments (wie in Ihrem Beispiel)
    )

    gens = [get_generation(generation=4*10) for i in range(4)]

    selector = TournamentSelector(tournament_size=2)
    mutator = UniformGaussianMutator(mutation_rate=0.2, mutation_strengh=0.2)
    crossover = UniformCrossover()
    evaluator = GlobalMaxMeanDivergenceEvaluator(prompt)


    ml = ModelLoader(cache_dir=os.environ.get("HF_CACHE", ""))

    sdxl = ml.load_sdxl()
    embed = ml.load_clip_embeddings()

    pipe = GeneticAlgorithmPipeline(
        generative_model=sdxl,
        prompt=new_prompt,
        embedding_model=embed,
        crossover_operation=crossover,
        selector=selector,
        mutator=mutator,
        noise_factory=nf,
        evaluator=evaluator,
        experiment_id=experiment_id,
        num_generations=4,
        population_size=10,
        initial_mutation_rate=0.05,
        initial_crossover_rate=0.8,
        elite_size=0,
        batch_size=3
    )
    pipe.name = f"{prompt.replace(' ', '_')}_to_{new_prompt.replace(' ', '_')}_{experiment_id}"

    for gen in gens:

        pipe.population = gen
        pipe.one_generation()
        pipe.save_generation()
        pipe.save_states()
        gen = pipe.population




if __name__ == "__main__":
    load_dotenv()
    experiment_id = args().experiment_id
    main(experiment_id=experiment_id)
