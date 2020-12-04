import os
import torch
import numpy as np
from util.utils import generate_all_logs, generate_ga_plots, generate_backprop_plots, save_object, save_meta_models
from util.utils import generate_discrete_one_hot_output, save_rl_models, generate_rl_plots


class DummyStateScaler:
    def __init__(self):
        pass

    def scale_down(self, input_array):
        return input_array


class CustomStateScaler:
    def __init__(self, state_dim, return_type='array'):
        # Considered as a row of a state. This should be able to handle multiple rows at the same time, specifically
        # for PPO's actor.
        self.max_state = np.zeros((1, state_dim))
        self.max_state += 1e-4

    def scale_down(self, input_array):
        # Should get in something like np array (n states, state_dim)
        max_in_input_per_dim = np.amax(input_array, axis=0).reshape(1, -1)  # converts (n,) to (1,n)
        self.max_state = np.maximum(self.max_state, max_in_input_per_dim)
        output = np.divide(input_array, self.max_state)
        # print('scaler max', self.max_state.shape, self.max_state)
        # output = np.tanh(input_array)
        # print('scaler', input_array, '-->', output)
        return output


class GeneratorDiscreteExpectedActionSampler:
    # make a sampling type that precreates a proper set of values as input to the generator.
    def __init__(self, config):
        self.sampler_type = config.env_config.action_space_type
        # Replacing this to give better representation of action space
        generator_expected_actions = torch.zeros(
            (config.generator_input_sample_length, config.generator_action_vector_size))
        indices = list(np.linspace(0, config.generator_input_sample_length, num=config.action_dim, endpoint=False, dtype=np.int8))
        indices.pop(0)
        for num in indices:
            generator_expected_actions[num:] += 1
        self.data = generator_expected_actions
        self.target_data = generate_discrete_one_hot_output(config.action_dim, config.generator_input_sample_length)

    def sample(self):
        return self.data, self.target_data


class GeneratorInputSampler:
    def __init__(self, config, num_samples_generated=10000):
        self.num_samples_generated = num_samples_generated
        self.sampler_type = config.env_config.action_space_type
        self.dev = config.dev
        if self.sampler_type == 'discrete':
            # Precreates the discrete input samples that can be used by the generator to work with.
            self.generator_discrete_expected_actions_sampler = GeneratorDiscreteExpectedActionSampler(config)
            # Precreates samples of noise to concatenate with discrete values to get variations in the states generated.
            self.generator_input_noise_3d_vector = torch.randn(
                (num_samples_generated, config.generator_input_sample_length, config.generator_noise_vector_size))
        elif self.sampler_type == 'continuous':
            self.generator_continuous_input_samples = torch.FloatTensor(
                np.random.uniform(config.action_space_low, config.action_space_high, size=(
                num_samples_generated, config.generator_input_sample_length, config.action_dim)))

    def sample(self):
        index = np.random.randint(0, self.num_samples_generated)
        if self.sampler_type == 'discrete':
            expected_actions_sample, target_data = self.generator_discrete_expected_actions_sampler.sample()
            with torch.no_grad():
                generator_input_noise_vector = self.generator_input_noise_3d_vector[index]
                return torch.cat((generator_input_noise_vector, expected_actions_sample), 1).to(self.dev), target_data.to(self.dev)
        elif self.sampler_type == 'continuous':
            return self.generator_continuous_input_samples[index].to(self.dev), self.generator_continuous_input_samples[index].to(self.dev)


class SupervisedLogs:
    def __init__(self, config):
        # Plotting statistics
        self.config = config
        self.actor_losses = []
        self.meta_losses = []  # PPO losses that have backpropagated across to the generator
        self.critic_losses = []
        self.new_actor_performances_mean = []
        self.new_actor_performances_std = []

    def start(self):
        # Log directory creation
        os.makedirs(self.config.log_path, exist_ok=True)

    def save(self, generator, critic):
        # Final plots
        generate_backprop_plots(self.config, self)

        # Saving statistics
        # generate_all_logs(self.config, self)
        generate_all_logs(self.config, self)

        # Saving all the models saved here
        save_meta_models(generator, critic, self.config.log_path)

        # Save config
        save_object(self.config, os.path.join(self.config.log_path, 'config.pkl'))


class RLLogs:
    def __init__(self, config):
        # Plotting statistics
        self.config = config
        self.actor_losses = []
        self.critic_losses = []
        self.actor_performances_mean = []
        self.actor_performances_std = []

    def start(self):
        # Log directory creation
        os.makedirs(self.config.log_path, exist_ok=True)

    def save(self, actor, critic):
        # Final plots
        generate_rl_plots(self.config, self)

        # Saving statistics
        generate_all_logs(self.config, self)

        # Saving all the models saved here
        save_rl_models(actor, critic, self.config.log_path)

        # Save config
        save_object(self.config, os.path.join(self.config.log_path, 'config.pkl'))


class GALogs:
    def __init__(self, config):
        self.config = config
        self.generator_performance_mean = []
        self.generator_performance_std = []

    def start(self):
        # Log directory creation
        os.makedirs(self.config.log_path)

    def load_population(self, population):
        models_dict = torch.load(os.path.join(self.config.log_path, 'population-' + str(self.config.last_gen) + '.pt'))
        for i in range(len(population.individuals)):
            population.individuals[i].model.load_state_dict(models_dict[str(i)])
            # models_dict[str(i)] = self.individuals[i].model.state_dict()
            # models_dict[str(i) + '_fitness'] = str(self.individuals[i].fitness_list) # Useless
        return population

    @staticmethod
    def save_population(population, num_gen):
        population.save(str(num_gen + 1))

    def save(self, overall_best_individual, final_population):
        # Save logs
        generate_all_logs(self.config, self)

        # Plot the graphs with this flow
        generate_ga_plots(self.config, self)

        # Final save population and best model
        torch.save(overall_best_individual.model.state_dict(),
                   os.path.join(self.config.log_path, 'overall_best_generator.pt'))

        # Saving population to the file
        final_population.save()

        # Save config
        save_object(self.config, os.path.join(self.config.log_path, 'config.pkl'))


# TODO
class CustomNormalizer:
    def __init__(self, x_len, return_type='array', num_prev_stored=50):
        # Meant for 1D arrays
        self.previous_vals = [[0 for i in range(num_prev_stored)] for k in range(x_len)]
        # self.mean = np.zeros(x_len)
        # self.num_previous_computes = 0
        self.return_type = return_type

    def normalize(self, input_array):
        # self.mean = (self.mean * self.num_previous_computes + input)
        # self.num_previous_computes += 1

        output = []
        for idx, val in enumerate(input_array):
            self.previous_vals[idx].append(val)
            self.previous_vals[idx].pop(0)
            val_normal = (val - np.mean(self.previous_vals[idx])) / (np.std(self.previous_vals[idx])+ 1e-5)
            output.append(val_normal)
        if self.return_type == 'array':
            return np.array(output)
        elif self.return_type == 'tensor':
            return torch.FloatTensor(output)
        else:
            raise Exception('Normalizer received unexpected return type')


class DummyNormalizer(CustomNormalizer):
    def __init__(self, return_type=None):
        self.return_type = return_type

    def normalize(self, input_array):
        if self.return_type is None:
            return input_array
        elif self.return_type == 'array':
            return np.array(input_array)
        elif self.return_type == 'tensor':
            return torch.FloatTensor(input_array)
        else:
            raise Exception('Normalizer received unexpected return type')
