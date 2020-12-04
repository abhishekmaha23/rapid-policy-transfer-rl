import torch
import os
import numpy as np
import random

from env_config import *


class BasicConfig(object):
    def __init__(self, run_id, env_name, params):
        print('Starting run - ', run_id)
        self.env_name = env_name
        print('Environment', env_name)
        print('Method', params['method'])
        print('Algorithm', params['algorithm'], ' - Family', params['family'])
        print('Memory Type', params['memory_type'])
        if params['algorithm'] is None:
            log_folder = params['method']
        else:
            log_folder = params['method'] + '-' + params['algorithm']
        log_folder = os.path.join(log_folder, env_name)
        # General
        self.run_id = run_id
        self.method = params['method']
        self.algorithm = params['algorithm']
        self.memory_type = params['memory_type']
        self.family = params['family']
        self.comment = 'Integrating phenomenal changes. Crowd going wild. Chants going around - ...Abhishek, Abhishek, Abhishek...'
        self.log_path = os.path.join(os.getcwd(), 'logs', log_folder, str(run_id))

        self.plot_save_iterations = 50  # Only used for intervals of testing agent performance, rest Gaussian smoothed
        self.plot_smoothing_sigma = 2

        self.dev = "cpu"
        # self.dev = "cuda:0"
        self.multiprocessing_cpu_cores = 5  # Unsolved issues with torch.multiprocessing
        self.multi = False
        # Reproducibility efforts
        # self.seed = 1234
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)
        # np.random.seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # random.seed(self.seed)


class SupervisedConfig(BasicConfig):
    def __init__(self, run_id, env_name, params):
        super(SupervisedConfig, self).__init__(run_id, env_name, params)
        self.current_episode = 0


class GeneticConfig(BasicConfig):
    def __init__(self, run_id, env_name, params):
        super(GeneticConfig, self).__init__(run_id, env_name, params)
        self.population_size = 200
        self.num_of_generations = 200

        self.plot_save_iterations = 50  # Only used for intervals of testing agent performance, rest Gaussian smoothed
        self.plot_smoothing_sigma = 2
        self.population_save_iterations = 1

        self.elite_per_gen = 15
        self.truncate_fraction = 0.5

        self.current_generation = 0
        self.ending_generation = 0


class CartpoleGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params):
        super(CartpoleGeneticConfig, self).__init__(run_id, env_name, params)

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.01

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 64

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 4  # Number of random agents tested
        self.actor_testing_loops = 4  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 10
        self.retest_actor_testing_loops = 10

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'CartPole-v0':
            self.env_config = CartPolev0Config
        if env_name == 'CartPole-v1':
            self.env_config = CartPolev1Config

        self.state_dim = 4
        self.action_dim = 2
        self.env_max_timesteps = None


class CartpoleSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(CartpoleSupervisedConfig, self).__init__(run_id, env_name, params)
        self.plot_save_iterations = 10  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 3)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.01
        self.actor_rl_learning_rate = 0.01

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 25

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 1000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 10  # Number of random agents tested
        self.actor_testing_loops = 10  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 10  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 10

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 200
        self.include_entropy_in_ppo = False
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental)
        self.offpol_batch_size = 500  # Setting this value even if it won't be used.

        if env_name == 'CartPole-v0':
            self.env_config = CartPolev0Config
        if env_name == 'CartPole-v1':
            self.env_config = CartPolev1Config

        self.state_dim = 4
        self.action_dim = 2
        self.env_max_timesteps = 200

        self.ended_early = False


class MountainCarSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, method='supervised', algorithm='PPO', memory_type='onpolicy'):
        super(MountainCarSupervisedConfig, self).__init__(run_id, env_name, method, algorithm, memory_type=memory_type)
        self.plot_save_iterations = 30  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_learning_rate = 0.01

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 512

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 2e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 3  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 10  # Number of random agents tested
        self.actor_testing_loops = 10  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 10  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 10

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 200
        self.include_entropy_in_ppo = True
        self.entropy_coefficient = 0.1
        self.num_ppo_steps = 4
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 500

        if env_name == 'MountainCar-v0':
            self.env_config = MountainCarDiscretev0Config

        self.state_dim = 2
        self.action_dim = 3
        self.env_max_timesteps = 250

        self.ended_early = False


class AcrobotSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, method='supervised', algorithm='PPO', memory_type='onpolicy'):
        super(AcrobotSupervisedConfig, self).__init__(run_id, env_name, method, algorithm, memory_type=memory_type)
        self.plot_save_iterations = 20  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.01
        self.actor_rl_learning_rate = 0.01

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 100

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 10000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 10  # Number of random agents tested
        self.actor_testing_loops = 10  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 10  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 10

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 200
        self.include_entropy_in_ppo = False
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental)
        self.offpol_batch_size = 500  # Setting this value even if it won't be used.

        if env_name == 'Acrobot-v1':
            self.env_config = Acrobotv1Config

        self.state_dim = 6
        self.action_dim = 3
        self.env_max_timesteps = 300

        self.ended_early = False


class LunarLanderGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params=None):
        super(LunarLanderGeneticConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 200

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 5  # Number of random agents tested
        self.actor_testing_loops = 2  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 10
        self.retest_actor_testing_loops = 10

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'LunarLander-v2':
            self.env_config = LunarLanderv2Config

        self.state_dim = 8
        self.action_dim = 4
        self.env_max_timesteps = 300


class LunarLanderSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(LunarLanderSupervisedConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 200

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 2e-1
        self.generator_min_lr_scaler = 1e-1

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.1

        self.outer_loop_iterations = 1000  # Generator training steps
        self.inner_loop_iterations = 2  # Actor training steps

        self.generator_testing_loops = 10  # Number of random agents tested
        self.actor_testing_loops = 10  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 10  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 10

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 500
        self.include_entropy_in_ppo = False
        self.entropy_coefficient = 0.1
        self.num_ppo_steps = 4
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 500

        if env_name == 'LunarLander-v2':
            self.env_config = LunarLanderv2Config

        self.state_dim = 8
        self.action_dim = 4
        self.env_max_timesteps = 300

        self.ended_early = False


class PendulumSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(PendulumSupervisedConfig, self).__init__(run_id, env_name, params)
        self.plot_save_iterations = 40  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1
        self.actor_rl_learning_rate = 0.01

        self.generator_action_vector_size = 1
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 50

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 2e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for stability purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 1000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 10  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 10

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 200
        self.include_entropy_in_ppo = True
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 1000
        self.offpol_start_episodes = 75  # Initial steps are kept random in order to fill memory with random values.
        self.offpol_target_update_rate = 0.1
        self.offpol_num_iterations_update = 4
        self.offpol_expl_noise = 0.01  # std for gaussian noise in exploration of actions in actor.

        if env_name == 'Pendulum-v0':
            self.env_config = Pendulumv0Config

        self.state_dim = 3
        self.action_dim = 1
        self.env_max_timesteps = 500  # Setting none, causes it to use the original instead.

        self.ended_early = False


class HalfCheetahSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(HalfCheetahSupervisedConfig, self).__init__(run_id, env_name, params)
        self.plot_save_iterations = 40  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1
        self.actor_rl_learning_rate = 0.1

        self.generator_action_vector_size = 6
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 1000

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 2e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 1000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 3  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 3

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 1000
        self.include_entropy_in_ppo = True
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 500
        self.offpol_start_episodes = 50  # Initial steps are kept random in order to fill memory with random values.
        self.offpol_target_update_rate = 0.1
        self.offpol_num_iterations_update = 5
        self.offpol_expl_noise = 0.1  # std for gaussian noise in exploration of actions in actor.

        if env_name == 'HalfCheetah-v2':
            self.env_config = HalfCheetahConfig

        self.state_dim = 17
        self.action_dim = 6
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class LunarLanderContinuousSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(LunarLanderContinuousSupervisedConfig, self).__init__(run_id, env_name, params)
        self.plot_save_iterations = 40  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1
        self.actor_rl_learning_rate = 0.01

        self.generator_action_vector_size = 2
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 100

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 1000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 5  # Number of random agents tested
        self.actor_testing_loops = 5  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 5  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 5

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 1000
        self.include_entropy_in_ppo = True
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 500
        self.offpol_start_episodes = 150  # Initial steps are kept random in order to fill memory with random values.
        self.offpol_target_update_rate = 0.01
        self.offpol_num_iterations_update = 5
        self.offpol_expl_noise = 0.1  # std for gaussian noise in exploration of actions in actor.

        if env_name == 'LunarLanderContinuous-v2':
            self.env_config = LunarLanderSupervisedv2Config

        self.state_dim = 8
        self.action_dim = 2
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class PendulumGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params=None):
        super(PendulumGeneticConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_action_vector_size = 1
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 200

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 5
        self.retest_actor_testing_loops = 5

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'Pendulum-v0':
            self.env_config = Pendulumv0Config

        self.state_dim = 3
        self.action_dim = 1
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class LunarLanderContinuousGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params=None):
        super(LunarLanderContinuousGeneticConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_action_vector_size = 2
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 200

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 5
        self.retest_actor_testing_loops = 5

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'LunarLanderContinuous-v2':
            self.env_config = LunarLanderSupervisedv2Config

        self.state_dim = 8
        self.action_dim = 2
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class HalfCheetahGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params=None):
        super(HalfCheetahGeneticConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_action_vector_size = 6
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 200

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 10
        self.retest_actor_testing_loops = 10

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'HalfCheetah-v2':
            self.env_config = HalfCheetahConfig

        self.state_dim = 17
        self.action_dim = 6
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class AcrobotGeneticConfig(GeneticConfig):
    def __init__(self, run_id, env_name, params=None):
        super(AcrobotGeneticConfig, self).__init__(run_id, env_name, params)
        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 80)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_init_learning_rate = 0.1

        self.generator_noise_vector_size = 16  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 16
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_input_sample_length = 100

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.inner_loop_iterations = 1  # Actor training steps

        self.ended_early = False

        self.generator_testing_loops = 3  # Number of random agents tested
        self.actor_testing_loops = 3  # Number of tests per agent (reduced due to multiprocessing hiccups)

        self.retest_generator_testing_loops = 10
        self.retest_actor_testing_loops = 10

        self.mutation_power = 0.5  # For generator to mutate into other possibilities

        self.normalize_states = False
        self.scale_states = False

        if env_name == 'Acrobot-v1':
            self.env_config = Acrobotv1Config

        self.state_dim = 6
        self.action_dim = 3
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False


class HopperSupervisedConfig(SupervisedConfig):
    def __init__(self, run_id, env_name, params):
        super(HopperSupervisedConfig, self).__init__(run_id, env_name, params)
        self.plot_save_iterations = 30  # Resetting some specific Cartpole plots
        self.plot_smoothing_sigma = 2

        self.actor_type = 'random'  # 'random' or 'standard'
        self.actor_batch_norm = True
        self.actor_random_layer_range = (1, 2)  # used for random
        self.actor_random_dim_range = (30, 200)  # used for random
        self.actor_standard_dim_num = 64  # used for standard
        self.actor_standard_layer_num = 1  # used for standard
        self.actor_learning_rate = 0.01

        self.generator_action_vector_size = 3
        self.generator_input_size = self.generator_action_vector_size
        self.generator_input_sample_length = 500

        self.generator_batch_norm = True
        self.generator_type = 'simple'  # simple, complex, complexer (unused)
        self.generator_learning_rate = 1e-1
        self.generator_min_lr_scaler = 1e-1

        self.critic_batch_norm = False
        self.critic_type = 'simple'  # simple, complex, complexer (unused)
        self.reward_size = 1
        self.critic_learning_rate = 0.01

        self.clip_grad_norm = True  # Attempting a clipping of the gradients for general purposes.
        self.max_grad_norm = 0.5

        self.outer_loop_iterations = 20000  # Generator training steps
        self.inner_loop_iterations = 1  # Actor training steps

        self.generator_testing_loops = 5  # Number of random agents tested
        self.actor_testing_loops = 5  # Number of tests per agent (reduced due to multiprocessing hiccups)
        self.retest_generator_testing_loops = 5  # If initial check worked out, then do multiple more times.
        self.retest_actor_testing_loops = 5

        self.discount_factor = 0.99

        self.normalize_states = False
        self.scale_states = True
        self.normalize_rewards_and_advantages = True  # In calculation of the

        # Algorithm specific  # A2C and PPO
        # On policy
        self.rollout_update_type = 'trajectory'  # Either 'trajectory' or 'batch'
        self.onpol_batch_size = 200
        self.include_entropy_in_ppo = True
        self.entropy_coefficient = 0.01
        self.num_ppo_steps = 5
        self.ppo_clip = 0.2

        # Off policy  # TD3 and SAC (experimental) (Won't work for discrete spaces)
        self.offpol_batch_size = 5000
        self.offpol_start_episodes = 100  # Initial steps are kept random in order to fill memory with random values.
        self.offpol_target_update_rate = 0.1
        self.offpol_num_iterations_update = 10
        self.offpol_expl_noise = 0.01  # std for gaussian noise in exploration of actions in actor.

        if env_name == 'Hopper-v3':
            self.env_config = HopperConfig

        self.state_dim = 11
        self.action_dim = 3
        self.env_max_timesteps = None  # Setting none, causes it to use the original instead.

        self.ended_early = False
