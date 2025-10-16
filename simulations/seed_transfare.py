import os
import sys
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from functools import partial

from src.crossover import UniformCrossover
from src.evaluators.global_max_mean_divergence_evaluator import GlobalMaxMeanDivergenceEvaluator
from src.evaluators.kernel_density_estimation_evaluator import KernelDensityEstimationEvaluator
from src.factorys import NoiseFactory
from src.huggingface_models import ModelLoader
from src.mutators.uniform_gaussian_mutator import UniformGaussianMutator
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from src.selector_functions.tournament_selector import TournamentSelector
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
        experiment_id=1549660
    )

    gens = [get_generation(generation=i*10) for i in range(5)]

    selector = TournamentSelector(tournament_size=2)
    mutator = UniformGaussianMutator(mutation_rate=0.2, mutation_strengh=0.2)
    crossover = UniformCrossover()
    


    ml = ModelLoader(cache_dir=os.environ.get("HF_CACHE", ""))

    sdxl = ml.load_sdxl()
    embed = ml.load_blip2_embeddings()

    evaluator = GlobalMaxMeanDivergenceEvaluator(prompt, embed.device)

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
        num_generations=5,
        population_size=100,
        initial_mutation_rate=0.0,
        initial_crossover_rate=0.0,
        elite_size=0,
        batch_size=5
    )
    pipe.name = f"{prompt.replace(' ', '_')}_to_{new_prompt.replace(' ', '_')}_{experiment_id}"

    for gen in gens:
        print(f"Generation: {gen[0].end_generation}")
    for gen in gens:
        
        pipe.population = gen
        pipe.one_generation()
        pipe.save_generation()
        pipe.save_states()
        gen = pipe.population






load_dotenv()
experiment_id = args().experiment_id
main(experiment_id=experiment_id)
