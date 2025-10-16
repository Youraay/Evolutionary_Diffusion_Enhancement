import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.arg_parser import args
from src.crossover import UniformCrossover
from src.evaluators.kernel_density_estimation_evaluator import KernelDensityEstimationEvaluator
from src.evaluators.global_max_mean_divergence_evaluator import GlobalMaxMeanDivergenceEvaluator
from src.huggingface_models import ModelLoader
from src.mutators.uniform_gaussian_mutator import UniformGaussianMutator
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from src.selector_functions.tournament_selector import TournamentSelector
from src.factorys import NoiseFactory

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def main(experiment_id):

    prompt = "sorrow"

    base_path = Path(os.environ.get("BASE_PATH"))
    blip_2_path = base_path / Path(os.environ.get("BLIP_2_BASELINE", "")) / prompt.replace(" ", "_")
    clip_path = base_path / Path(os.environ.get("CLIP_BASELINE", "")) / prompt.replace(" ", "_")
    sdxl_path = base_path / Path(os.environ.get("SDXL_BASELINE", "")) / prompt.replace(" ", "_")

    selector = TournamentSelector(tournament_size=2)
    mutator = UniformGaussianMutator(mutation_rate=0.2,mutation_strengh=0.2)
    crossover = UniformCrossover()
    evaluator = GlobalMaxMeanDivergenceEvaluator(prompt, "cuda")

    noise_factory = NoiseFactory()

    ml = ModelLoader(cache_dir=os.environ.get("HF_CACHE", ""))

    sdxl = ml.load_sdxl()
    embed = ml.load_blip2_embeddings()

    pipe = GeneticAlgorithmPipeline(
        generative_model=sdxl,
        prompt=prompt,
        embedding_model=embed,
        crossover_operation=crossover,
        selector=selector,
        mutator=mutator,
        noise_factory=noise_factory,
        evaluator=evaluator,
        experiment_id=experiment_id,
        num_generations=40,
        population_size=100,
        initial_mutation_rate=0.05,
        initial_crossover_rate=1,
        elite_size=0,
        batch_size=5
    )

    pipe.run()


if __name__ == "__main__":
    load_dotenv()
    experiment_id = args().experiment_id
    main(experiment_id= experiment_id)
