import torch
import gym
from tqdm import tqdm
from aux.models import ActorCache, Generator, Critic
from util.utils import test_generator_performance, check_convergence_of_generator
import higher
from aux.common import GeneratorInputSampler
from aux.supervised import rollout, MemoryCache
from algorithms import ppo_update, a2c_update, td3_update, ddpg_update
import numpy as np

''' A2C and PPO code adapted from https://github.com/bentrevett/pytorch-rl'''
''' TD3 code adapted from original author'''


def train(config, logs):
    actual_test_env = gym.make(config.env_name)
    rollout_env = gym.make(config.env_name)
    # actual_test_env.seed(config.seed + 20)
    # rollout_env.seed(config.seed)

    if config.env_max_timesteps is None:
        config.env_max_timesteps = actual_test_env._max_episode_steps

    if config.env_config.action_space_type == 'continuous':
        config.action_space_low = actual_test_env.action_space.low
        config.action_space_high = actual_test_env.action_space.high
    elif config.env_config.action_space_type == 'discrete':
        config.action_space_low = None
        config.action_space_high = None

    config.observation_space_high = actual_test_env.observation_space.high
    config.observation_space_low = actual_test_env.observation_space.low
    # print('high', config.observation_space_high)
    # print('low', config.observation_space_low)

    models = dict()
    models['generator'] = Generator(config).to(config.dev)
    models['critic'] = Critic(config)
    if config.algorithm == 'TD3':
        models['critic_2'] = Critic(config)
        models['critic_2'].train()
    models['generator'].train()
    models['critic'].train()

    # Init population of random actors so that initialization doesn't consume all compute
    random_actor_sampler = ActorCache(config)

    if config.family == 'deterministic':
        # Only for Q-learning family
        models['critic_target'] = Critic(config)
        models['critic_target'].load_state_dict(models['critic'].state_dict())
        if config.algorithm == 'TD3':
            models['critic_target_2'] = Critic(config)
            models['critic_target_2'].load_state_dict(models['critic_2'].state_dict())
        # Initializing a random actor only for the beginning. New actors will be adapted from the actor of the last loop
        models['actor_target'] = random_actor_sampler.sample()

    optimizers = dict()
    optimizers['generator_opt'] = torch.optim.Adam(models['generator'].parameters(), lr=config.generator_learning_rate)
    optimizers['critic_opt'] = torch.optim.Adam(models['critic'].parameters(), lr=config.critic_learning_rate)
    # optimizers['generator_opt'] = torch.optim.Adam(models['generator'].parameters())
    # optimizers['critic_opt'] = torch.optim.Adam(models['critic'].parameters())
    if config.algorithm == 'TD3':
        optimizers['critic_opt_2'] = torch.optim.Adam(models['critic_2'].parameters(), lr=config.critic_learning_rate)

    optimizers['generator_opt'].zero_grad()
    optimizers['critic_opt'].zero_grad()

    # Generator LR scheduler - not major impact, but nice to have
    generator_opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['generator_opt'], config.outer_loop_iterations,
                                                                         config.generator_learning_rate * config.generator_min_lr_scaler)

    generator_input_sampler = GeneratorInputSampler(config)

    memory_cache = MemoryCache(memory_type=config.memory_type, batch_size=config.offpol_batch_size)

    print('Initialized actor cache with ', random_actor_sampler.num_actors, config.actor_type, 'actor(s).')
    print('-------')

    rollout_performance = dict()
    rollout_performance['rollout_rewards'] = []
    rollout_performance['rollout_lengths'] = []
    logs.start()
    inner_loop_performance = dict()
    for outer_loop_num in tqdm(range(config.outer_loop_iterations)):
        rollout_length, rollout_reward = inner_loop(config, random_actor_sampler, models, optimizers, rollout_env, logs, generator_input_sampler, memory_cache)
        rollout_performance['rollout_rewards'].append(rollout_reward)
        rollout_performance['rollout_lengths'].append(rollout_length)
        # tqdm.write('INFO - '+str(outer_loop_num) + ' ->' + str(rollout_performance))
        # This has to happen outside. It's not good to have inside. Makes no sense. Just an aesthetic thing.
        optimizers['generator_opt'].step()  # Second-order gradient update -
        generator_opt_scheduler.step()  # Slowly reducing the learning rate of the generator to a minimal value.

        if outer_loop_num % config.plot_save_iterations == 0:
            if config.inner_loop_iterations > 0:
                generator_performance = test_generator_performance(random_actor_sampler, models['generator'], actual_test_env, config, generator_input_sampler, multi=config.multi)
                tqdm.write('Evaluation perf.' + str(generator_performance) + ', Rollout perf.-[' + str(
                    np.mean(rollout_performance['rollout_rewards'])) + ', ' + str(
                    np.mean(rollout_performance['rollout_lengths'])) + '], Memory filled -' + str(len(memory_cache.storage)) + '/' + str(memory_cache.max_size))
                rollout_performance['rollout_rewards'] = []
                rollout_performance['rollout_lengths'] = []
                # tqdm.write('Max_state_reached'+ str(state_scaler.max_state))
                logs.new_actor_performances_mean.append(generator_performance[0])
                logs.new_actor_performances_std.append(generator_performance[1])

                if check_convergence_of_generator(config, random_actor_sampler, generator_performance, models['generator'], actual_test_env, generator_input_sampler):
                    tqdm.write('Ended run early at loop ' + str(outer_loop_num))
                    tqdm.write('Final passing performance ' + str(config.converged_performance_mean) + ', ' + str(config.converged_performance_std))
                    config.ended_iteration = outer_loop_num
                    break

    logs.save(models['generator'], models['critic'])


def inner_loop(config, random_actor_sampler, models, optimizers, rollout_env, logs, generator_input_sampler, memory_cache):
    # Getting a random agent with an architecture that's sampled from a random hidden layer size each iteration.

    optimizers['generator_opt'].zero_grad()

    actor = random_actor_sampler.sample()
    actor.train()
    actor_inner_opt = torch.optim.SGD(actor.parameters(), lr=config.actor_init_learning_rate)
    actor_inner_opt.zero_grad()
    actor_rl_opt = torch.optim.Adam(actor.parameters(), lr=0.001)
    # Idea of initialization training is the same for deterministic algorithms.

    with higher.innerloop_ctx(actor, actor_inner_opt, copy_initial_weights=True) as (fnet_actor, diff_act_opt):
        # Changing differentiable optimizer inside the loop to SGD as follows
        for inner_loop_num in range(config.inner_loop_iterations):
            actor_criterion = torch.nn.MSELoss(reduction='sum')

            generator_input, actor_target_output = generator_input_sampler.sample()
            softmax_actor_predicted_actions = fnet_actor(models['generator'](generator_input), source='generator')
            actor_loss = actor_criterion(softmax_actor_predicted_actions, actor_target_output)

            # Train the actor with a differentiable optimizer.
            diff_act_opt.step(actor_loss)

            logs.actor_losses.append(actor_loss.item())

        # Do rollout and get outputs for loss calculation
        # Here, we use trajectory-based steps for also determining the frequency of deterministic policy updates
        memory_cache, current_rollout_reward, current_rollout_length = rollout(config, fnet_actor, rollout_env, memory_cache)  # Ideal.
        actor_inner_opt.zero_grad()
        optimizers['critic_opt'].zero_grad()

        # Discounted normalized returns and advantages
        if config.algorithm == 'PPO':
            total_policy_loss, total_value_loss = ppo_update(config, fnet_actor, diff_act_opt, models['critic'], optimizers['critic_opt'], memory_cache, update_type='meta')
        elif config.algorithm == 'A2C':
            total_policy_loss, total_value_loss = a2c_update(config, models['critic'], optimizers['critic_opt'], memory_cache, update_type='meta')
        elif config.algorithm == 'DDPG':
            total_policy_loss, total_value_loss = ddpg_update(config, fnet_actor, diff_act_opt, models, optimizers, memory_cache, update_type='meta')
        elif config.algorithm == 'TD3':
            total_policy_loss, total_value_loss = td3_update(config, fnet_actor, diff_act_opt, models, optimizers, memory_cache, update_type='meta')

        total_policy_loss.backward()  # The gradients backpropagate far and generator_opt makes a second-order step.
        # total_value_loss.backward()
        # Optional to clip grad norm
        if config.clip_grad_norm:
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(models['generator'].parameters(), config.max_grad_norm)

        actor_inner_opt.step()   # Not really necessary, since we're restarting the whole thing.
        # optimizers['critic_opt'].step()

        logs.meta_losses.append(total_policy_loss.item())
        logs.critic_losses.append(total_value_loss)

        # del fnet_actor

    actor.to('cpu')
    # del actor_inner_opt

    return current_rollout_reward, current_rollout_length
