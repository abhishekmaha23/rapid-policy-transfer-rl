import torch
import numpy as np
import random
import os
import copy
import gym
from aux.models import Generator, ActorCache
from aux.common import GeneratorInputSampler
from util.utils import test_generator_performance


class GeneratorPopulation:
    def __init__(self, config, variant='generator'):
        self.config = config
        self.individuals = []
        self.test_env = gym.make(config.env_name)

        if config.env_max_timesteps is None:
            config.env_max_timesteps = self.test_env._max_episode_steps
        if config.env_config.action_space_type == 'continuous':
            config.action_space_low = self.test_env.action_space.low
            config.action_space_high = self.test_env.action_space.high
        elif config.env_config.action_space_type == 'discrete':
            config.action_space_low = None
            config.action_space_high = None
        config.observation_space_high = self.test_env.observation_space.high
        config.observation_space_low = self.test_env.observation_space.low

        self.generator_input_sampler = GeneratorInputSampler(config)
        self.random_actor_sampler = ActorCache(config)
        print('Initialized actor cache with ', self.random_actor_sampler.num_actors, config.actor_type, 'actor(s).')
        print('-------')

        # Initialize entire population
        if variant == 'generator':
            while len(self.individuals) < self.config.population_size:
                self.individuals.append(GeneratorIndividual(self.config, random_actor_sampler=self.random_actor_sampler))
            self.population_best_individual = self.individuals[0]
        elif variant == 'resume':
            pass

    def save(self, gen_num_identifier='latest'):
        # Must save config and individuals
        # Assuming here that the population state is frozen at this generation.
        models_dict = {}
        for i in range(len(self.individuals)):
            models_dict[str(i)] = self.individuals[i].model.state_dict()
            models_dict[str(i) + '_fitness'] = str(self.individuals[i].fitness_list)
        torch.save(models_dict, os.path.join(self.config.log_path, 'population-'+gen_num_identifier+'.pt'))

    def refill_with_mutations(self):
        while len(self.individuals) < self.config.population_size:
            # Note ----->>>>> 2-way selection code completely taken from code of GA World Models
            # While the exact logic doesn't match the Wikipedia page, the results can't be argued with
            s1 = random.choice(self.individuals)
            s2 = s1
            while s1 == s2:
                s2 = random.choice(self.individuals)
            if s1.mean_fitness < s2.mean_fitness:  # Lower is better
                selected_solution = s1
            else:
                selected_solution = s2
            if s1 == self.population_best_individual:  # If they are the elite they definitely win
                selected_solution = s1
            elif s2 == self.population_best_individual:
                selected_solution = s2

            mutant_child = selected_solution.copy_model()
            mutant_child.mutate()
            # print('Testing number ', mutant_child.num_times_evaluated)

            self.individuals.append(mutant_child)


class GeneratorIndividual:
    def __init__(self, config, random_actor_sampler=None, model=None, num_times_evaluated=0):
        if model is None:
            self.model = Generator(config)
        else:
            self.model = model
        self.random_actor_sampler = random_actor_sampler
        self.config = config
        self.fitness_list = []  # Contains list of previous calculations of the fitness (specific to weights)
        self.mean_fitness = 0.
        self.std_fitness = 0.
        self.fitness = 0.
        self.num_times_evaluated = num_times_evaluated
    #  Basically we want more and more evaluations to happen of the elites, just so
    #  that the elites aren't flukes that got lucky with their evaluations

    def evaluate(self, test_env, generator_input_sampler, mode='normal'):
        generator_performance = test_generator_performance(self.random_actor_sampler, self.model, test_env,
                                                           self.config, generator_input_sampler, multi=False, mode=mode)
        self.mean_fitness = (generator_performance[0] + (self.mean_fitness * self.num_times_evaluated)) / (self.num_times_evaluated + 1)
        self.std_fitness = (generator_performance[1] + (self.std_fitness * self.num_times_evaluated)) / (self.num_times_evaluated + 1)
        if mode == 'retest':
            self.num_times_evaluated += self.config.retest_generator_testing_loops
        elif mode == 'normal':
            self.num_times_evaluated += self.config.generator_testing_loops

        self.fitness = self.mean_fitness

    def copy_model(self):  # For existing individuals
        return GeneratorIndividual(self.config, random_actor_sampler=self.random_actor_sampler, model=copy.deepcopy(self.model))

    def mutate(self):  # For newly created individuals
        params = self.model.state_dict()
        for key in params:
            # print(key)
            if not key.endswith('num_batches_tracked'):
                params[key] += torch.from_numpy(np.random.normal(0, 1, params[key].size()) * self.config.mutation_power).float()


# Code for testing normal RL with GAs, but abandoned.

# class ActorIndividual:
#     def __init__(self, config, model=None):
#         if model is None:
#             self.model = MLPActor(config.state_size, config.generator_hidden_layer_sizes, config.action_size)
#             self.model.apply(init_weights)
#         else:
#             self.model = model
#         self.config = config
#         self.fitness_list = []  # Contains list of previous calculations of the fitness (specific to weights)
#         self.mean_fitness = 0
#         self.std_fitness = 0
#     #  Basically we want more and more evaluations to happen of the elites, just so
#     #  that the elites aren't flukes that got lucky with their evaluations
#
#     def evaluate(self, test_env, num_evals):
#         for i in range(num_evals):
#             # generator_performance = test_generator_performance(self.model, test_env, self.config,
#             #                                                self.config.generator_expected_actions,
#             #                                                self.config.generator_one_hot_expected_actions)
#             agent_performance = test_agent_performance(self.model, test_env, 'cpu')
#             self.fitness_list.append(agent_performance[0])
#         self.mean_fitness = np.mean(self.fitness_list)
#         self.std_fitness = np.std(self.fitness_list)
#
#     def copy_model(self):  # For existing individuals
#         return ActorIndividual(self.config, copy.deepcopy(self.model))
#
#     def mutate(self):  # For newly created individuals
#         params = self.model.state_dict()
#         for key in params:
#             params[key] += torch.from_numpy(np.random.normal(0, 1, params[key].size()) * self.config.mutation_power).float()
