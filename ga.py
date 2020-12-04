# from configurations import Config
from aux.common import GALogs, GeneratorInputSampler
from aux.ga import GeneratorPopulation
# from aux.models import ActorCache
import os
from tqdm import tqdm


def train(config, logs):
    ga = GeneticAlgorithm(config, logs)
    ga.run()


class GeneticAlgorithm:
    def __init__(self, config, logs):
        self.config = config
        self.logs: GALogs = logs
        if config.mode == 'resume':
            self.population = logs.load_population(population=GeneratorPopulation(self.config, variant='generator'))
            self.config.log_path = self.config.log_path + '-resume'
            self.config.current_generation = self.config.last_gen
            print('Resuming from generation', config.last_gen)
        else:
            self.population = GeneratorPopulation(config, variant='generator')
        self.truncate_threshold_number = int(self.config.population_size * self.config.truncate_fraction)
        print('Truncation threshold', self.truncate_threshold_number)

    def reached_threshold(self, best_ind):
        return best_ind.mean_fitness >= self.config.env_config.reward_threshold and \
               best_ind.std_fitness <= self.config.env_config.reward_std_threshold

    def run(self):
        overall_best_individual = self.population.individuals[0]  # Initializing to a random one, but will update next
        # population_best_individual = self.population.individuals[0]

        # Log directory creation
        os.makedirs(self.config.log_path)

        # for num_gen in trange(self.config.num_of_generations):
        for num_gen in tqdm(range(self.config.num_of_generations)):
            # Evaluation (Fitness of the entire population) # 1st time evaluation

            # Single-core processing
            for individual in self.population.individuals:
                individual.evaluate(self.population.test_env, self.population.generator_input_sampler)

            # print('Afresh first eval')
            # Sort population on fitness
            self.population.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

            # Re-evaluate elite set, and generation best in this generation # 2nd time for elite top 5
            for individual in self.population.individuals[:self.config.elite_per_gen]:
                individual.evaluate(self.population.test_env, self.population.generator_input_sampler)
            self.population.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
            for elite in self.population.individuals[:self.config.elite_per_gen]:
                if elite.fitness >= self.population.population_best_individual.fitness:
                    self.population.population_best_individual = elite
            # Assuming above that the elite performance doesn't tank crazily below threshold just because of more
            # evaluations.
            if self.population.population_best_individual.fitness >= overall_best_individual.fitness:
                overall_best_individual = self.population.population_best_individual

            # Drop under-threshold people from population.
            del self.population.individuals[self.truncate_threshold_number:]
            # At no point outside truncation has the population been reduced further, so we must have half the initial
            # Now we mutate.
            self.population.refill_with_mutations()
            # Logs and plotting - evaluate the best a bit more, make it sweat. # 3rd time
            overall_best_individual.evaluate(self.population.test_env, self.population.generator_input_sampler)

            tqdm.write('[' + str(self.config.current_generation) + '] Best GA ind. Performance - ' + str(
                overall_best_individual.mean_fitness) + ', ' + str(overall_best_individual.std_fitness) + ', ' + str(
                overall_best_individual.fitness) + ', evaluated ' + str(
                overall_best_individual.num_times_evaluated) + ' time(s).')
            self.logs.generator_performance_mean.append(overall_best_individual.mean_fitness)
            self.logs.generator_performance_std.append(overall_best_individual.std_fitness)
            # print('===========================')
            # Save every nth population in entirety, must be able to resume with all statistics
            if num_gen % self.config.population_save_iterations == 0:
                self.logs.save_population(self.population, self.config.current_generation)
            self.config.current_generation = self.config.current_generation + 1

            # End early if threshold is reached
            if self.reached_threshold(overall_best_individual):
                tqdm.write('Reached the threshold, but must stress test a bit further')
                # Evaluate and test 2 more times and if passed, then end that it works.
                for i in range(1):
                    overall_best_individual.evaluate(self.population.test_env, self.population.generator_input_sampler,
                                                     mode='retest')
                if self.reached_threshold(overall_best_individual):
                    self.config.ending_generation = self.config.current_generation
                    self.config.ended_early = True
                    self.config.converged_performance_mean = overall_best_individual.mean_fitness
                    self.config.converged_performance_std = overall_best_individual.std_fitness
                    tqdm.write('Ended run early at generation ' + str(self.config.current_generation))
                    tqdm.write(
                        'Final best GA ind. Performance - ' + str(overall_best_individual.mean_fitness) + ', ' + str(
                            overall_best_individual.std_fitness) + ', ' + str(
                            overall_best_individual.fitness) + ', evaluated ' + str(
                            overall_best_individual.num_times_evaluated) + ' time(s).')
                    break

            num_gen += 1

        self.logs.save(overall_best_individual, self.population)
