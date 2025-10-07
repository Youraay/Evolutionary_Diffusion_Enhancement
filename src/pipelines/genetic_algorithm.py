import copy
import logging
import os
import random
from pathlib import Path
from typing import Generator, Any
import pandas as pd
import torch

from models import Noise
from .pipeline import Pipeline
from ..crossover.base_crossover import CrossoverFunction
from ..evaluators.base_evaluator import Evaluator
from ..factorys import NoiseFactory
from src.huggingface_models.base_strategy import GenerativModelStrategy, EmbeddingModelStrategy, \
    CaptionModelStrategy
from ..mutators.base_mutator import MutationFunction
from ..selector_functions.base_selectorf_unction import SelectorFunction

logger = logging.getLogger(__name__)

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
               experiment_id : str,
               num_generations: int,
               population_size: int,
               initial_mutation_rate: float,
               initial_crossover_rate: float,
               elite_size: int = 0,
               batch_size: int = 1,
               caption_model : CaptionModelStrategy = None,
               ):
        self.population = None
        self.generative_model = generative_model
        self.embedding_model = embedding_model
        self.crossover_operation = crossover_operation
        self.noise_factory = noise_factory
        self.crossover_rate = initial_crossover_rate
        self.elite_size = elite_size
        self.caption_model = caption_model
        self.evaluator = evaluator
        self.initial_mutation_rate = initial_mutation_rate
        self.prompt = prompt
        self.num_generations = num_generations
        self.population_size = population_size
        self.selection_function = selector
        self.mutator = mutator


        self.batch_size = batch_size
        self.generations_done = 0

        base_path = Path(os.environ['BASE_PATH'])
        self.name = f"{self.prompt.replace(' ', '_')}_{experiment_id}"

        self.result_path = base_path / 'results' / self.name
        self.state_path = base_path / 'states' / self.name

    def create_batches(self, embeds : list[Any]) -> Generator[list[Any], None, None]:
        num_samples = len(embeds)
        for i in range(0, num_samples, self.batch_size):
            # Schneidet die Liste der Tensoren in Chunks der Größe batch_size
            yield embeds[i:i + self.batch_size]

    def save_states(self):
        columns = [
            "generation",
            "candidate_id",
            "start_gen",
            "end_gen",
            "fitness",
            "score_name",
            "score_value",
            "caption",
            "file_name"
        ]
        data_rows = [ ]
        for candidate in self.population:
            row = [
                self.generations_done,
                candidate.id,
                candidate.start_generation,
                candidate.end_generation,
                candidate.fitness,
                candidate.evaluation_scores[0]["name"],
                candidate.evaluation_scores[0]["score"],
                getattr(candidate, 'caption', ''),
                candidate.file_name
            ]
        data_rows.append(row)
        new_data_df = pd.DataFrame(data_rows, columns=columns)
        if not self.output_file.exists():
            header_needed = True
        else:
            header_needed = False
        try:
            new_data_df.to_csv(self.state_path,
                                mode='a',
                                header=header_needed,
                                index=False)
        except Exception as e:
            logging.error(f"FEHLER beim Schreiben der CSV-Datei: {e}")

    def one_generation(self):
        pils = []
        for batch in self.create_batches([candidate.initial_noise for candidate in self.population]):
            pil_images = self.generative_model.generate_batch(batch, self.prompt)
            pils.extend(pil_images)

        embeddings = []
        for batch in self.create_batches(pils):
            batch_embeddings = self.embedding_model.embed_batch(batch)
            embeddings.extend(batch_embeddings)

        scores = []
        for batch in self.create_batches(embeddings):
            batch_scores = self.evaluator.evaluate_batch(batch)
            scores.extend(batch_scores)

        captions = []
        if self.caption_model is not None:

            for batch in self.create_batches(pils):
                batch_captions = self.caption_model.caption_batch(batch)
                captions.extend(batch_captions)

        for i, candidate in enumerate(self.population):
            candidate.pil_images = pils[i]
            candidate.blip2_embedding = embeddings[i]
            candidate.evaluation_scores.append(scores[i])
            candidate.calculate_fitness()
            if self.caption_model is not None:
                candidate.caption = captions[i]


    def evolve(self):
        self.generations_done += 1
        new_gen : list[Noise]= [ ]
        self.population = sorted(self.population, key=lambda noise: noise.fitness )
        new_gen.extend(self.population[:self.elite_size])

        while len(new_gen) < self.population_size:

            parent1 = self.selection_function.select(self.population)
            parent2 = self.selection_function.select(self.population)

            if random.random() < self.crossover_rate:
                child = copy.deepcopy(parent1)
                child.end_generation = self.generations_done
                new_gen.append(parent1)

            else:
                child_noise = self.crossover_operation.crossover(parent1.initial_noise, parent2.initial_noise)
                child = self.noise_factory.create_noise_from_noise(child_noise)
                child.end_generation = self.generations_done
                child.start_generation = self.generations_done
                child.crossover = True

                if random.random() < self.initial_mutation_rate:
                    child.initial_noise = self.mutator.mutate(child.initial_noise)
                    child.mutate = True

        self.population = new_gen



    def initial_population(self):
        self.population = self.noise_factory.create_batch(self.population_size)
        self.one_generation()
        for candidate in self.population:
            candidate.end_generation = 0
            candidate.start_generation = 0

    def save_generation(self):
        for candidate in self.population:
            candidate.save_pil_image(self.result_path / "images")
            candidate.save_blip2(self.result_path / "blip2")
            candidate.save_noise(self.result_path / "initial_noise")
            candidate.save_noise_to_rgb(self.result_path / "initial_noise_rgb")

    def run(self):
        self.initial_population()
        self.save_generation()
        self.save_states()
        while self.generations_done < self.num_generations:
            self.evolve()
            self.one_generation()
            self.save_generation()
            self.save_states()

        return self.population


