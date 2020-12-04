import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
import os
import time
from scipy.ndimage.filters import gaussian_filter1d
from itertools import repeat
import copy
import gym
# import torch.multiprocessing as multiprocessing
import multiprocessing
import pickle
import matplotlib
matplotlib.use("TkAgg")
from collections import defaultdict, Counter


def plot_fig(x, y, std=None, title=None, draw_grid=True,
             xlabel=None, ylabel=None, add_legend=False,
             label=None, display_fig=True,
             save_fig=False, save_name=None, xlim=[None, None], ylim=[None, None], img_size=(10, 6), update_fig=None, smooth_fill=False, smooth_fill_sigma=None):
    # plt.ion()
    plt.figure(figsize=img_size)
    assert type(x) == list, 'X is not a list'
    # assert type(y) == list, 'Y is not a list'
    if update_fig is None:
        fig, = plt.plot(x, y, label=label)
    axes = plt.gca()
    axes.set_autoscale_on(True)  # enable autoscale
    axes.autoscale_view(True, True, True)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if std is not None:
        if type(std) == list:
            lower_list = [y[i] - std[i] for i in range(len(x))]
            upper_list = [y[i] + std[i] for i in range(len(x))]
        else:
            lower_list = [i - std for i in y]
            upper_list = [i + std for i in y]
        if smooth_fill is True:
            if smooth_fill_sigma is None:
                smooth_fill_sigma = (len(x) // 1000) + 1
            # smooth upper and lower parts of the filling
            lower_list = gaussian_filter1d(lower_list, sigma=smooth_fill_sigma)
            upper_list = gaussian_filter1d(upper_list, sigma=smooth_fill_sigma)
        plt.fill_between(x, lower_list, upper_list, color='b', alpha=.1)
    if add_legend:
        plt.legend(fontsize=22)
    plt.xticks(size=22)
    plt.yticks(size=22)
    if draw_grid:
        plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=22)
    if ylabel is not None:
        plt.ylabel(None, fontsize=22)
    if title is not None:
        plt.title(title, fontsize=22)

    if save_fig:
        if save_name is None:
            save_name = 'plot_-'+xlabel+' vs. ' + ylabel + str(datetime.now()) + ".pdf"
        plt.savefig(save_name)
    if display_fig:
        plt.show()
    return fig, axes


def test_agent_performance(agent, env, device, num_tests=10, agent_id=999, mode='supervised'):
    agent.eval()
    rewards_so_far = []
    time_steps_so_far = []
    agent_type = agent.action_space_type
    actions_dict = defaultdict(int)
    # print(agent_id, 'starting testing of agent', agent.dim)
    for test in range(num_tests):
        done = False
        observation = env.reset()
        i = 0
        time_step = 0
        while not done:
            action = agent.get_action(agent(observation), context='test')

            if agent_type == 'discrete':
                actions_dict[action] += 1
            elif agent_type == 'continuous':
                action = action.reshape(agent.dim[-1],)
            observation, reward, done, info = env.step(action)
            # if mode == 'ga':
            #     if i < -200:
            #         done = True
            i += reward
            time_step += 1
        rewards_so_far.append(i)
        time_steps_so_far.append(time_step)
    if mode == 'ga':
        return np.mean(rewards_so_far), np.std(rewards_so_far), np.mean(time_steps_so_far)
    else:
        return np.mean(rewards_so_far), np.std(rewards_so_far), np.mean(time_steps_so_far), actions_dict


def test_generator_performance(random_actor_sampler, generator, actual_test_env, config, generator_input_sampler, multi=True, mode='normal'):
    multi_performances = []
    trained_agents = []
    if mode == 'retest':
        outer_test_loops = config.retest_generator_testing_loops
        inner_test_loops = config.retest_actor_testing_loops
    else:
        outer_test_loops = config.generator_testing_loops
        inner_test_loops = config.actor_testing_loops
    count = Counter(defaultdict(int))
    time_steps = []
    for i in range(outer_test_loops):
        # new_actor = get_random_agent(config.state_dim, config.action_dim, config.env_config.action_space_type, batch_norm=config.batch_norm)
        new_actor = random_actor_sampler.sample()
        new_actor_opt = torch.optim.SGD(new_actor.parameters(), lr=config.actor_init_learning_rate)
        for inner_loop_num in range(config.inner_loop_iterations):
            new_actor_opt.zero_grad()
            actor_criterion = torch.nn.MSELoss(reduction='sum')
            # softmax_actor_predicted_actions = new_actor(generator(get_generator_input()))
            generator_input, actor_target_output = generator_input_sampler.sample()
            softmax_actor_predicted_actions = new_actor(generator(generator_input), source='generator')
            new_actor_loss = actor_criterion(softmax_actor_predicted_actions, actor_target_output)
            new_actor_loss.backward()
            new_actor_opt.step()
        trained_agents.append(new_actor)
    if multi is False:
        performances_mean = []
        performances_std = []
        for agent in trained_agents:
            performance = test_agent_performance(agent, actual_test_env, config.dev, num_tests=inner_test_loops, mode='ga')
            performances_mean.append(performance[0])
            performances_std.append(performance[1])
            time_steps.append(performance[2])
            # count += Counter(performance[3])
    else:
        pool = multiprocessing.Pool(5)
        envs_list = []
        for i in range(len(trained_agents)):
            envs_list.append(copy.deepcopy(actual_test_env))
            ids = [i for i in range(len(trained_agents))]
        multi_performances = pool.starmap(test_agent_performance, zip(trained_agents, envs_list, repeat(config.dev), repeat(inner_test_loops), ids, repeat('ga')))
        performances_mean, performances_std, time_steps = zip(*multi_performances)
        pool.close()

    return np.mean(performances_mean), np.mean(performances_std), np.mean(time_steps), sorted(dict(count).items())


def check_convergence_of_generator(config, random_actor_sampler, current_generator_performance, generator, test_env, generator_input_sampler):
    def reached_threshold(generator_performance):
        return generator_performance[0] >= config.env_config.reward_threshold and generator_performance[1] <= config.env_config.reward_std_threshold

    if reached_threshold(current_generator_performance):
        print('Crossed threshold once, testing again.')
        final_test_performance_mean = [current_generator_performance[0]]
        final_test_performance_std = [current_generator_performance[1]]
        for i in range(1):
            extra_generator_performance = test_generator_performance(random_actor_sampler, generator, test_env, config,
                                                                     generator_input_sampler, multi=config.multi,
                                                                     mode='retest')
            final_test_performance_mean.append(extra_generator_performance[0])
            final_test_performance_std.append(extra_generator_performance[1])
        final_test_performance = (np.mean(final_test_performance_mean), np.mean(final_test_performance_std))
        if reached_threshold(final_test_performance):
            config.ended_early = True
            config.converged_performance_mean = final_test_performance[0]
            config.converged_performance_std = final_test_performance[1]
    return config.ended_early


def check_convergence_of_actor(config, actor, current_actor_performance, test_env):
    def reached_threshold(actor_performance):
        return actor_performance[0] >= config.env_config.reward_threshold and actor_performance[1] <= config.env_config.reward_std_threshold

    if reached_threshold(current_actor_performance):
        print('Crossed threshold once, testing again.')
        final_test_performance_mean = [current_actor_performance[0]]
        final_test_performance_std = [current_actor_performance[1]]
        for i in range(1):
            extra_actor_performance = test_agent_performance(actor, test_env, config.dev)
            final_test_performance_mean.append(extra_actor_performance[0])
            final_test_performance_std.append(extra_actor_performance[1])
        final_test_performance = (np.mean(final_test_performance_mean), np.mean(final_test_performance_std))
        if reached_threshold(final_test_performance):
            config.ended_early = True
            config.converged_performance_mean = final_test_performance[0]
            config.converged_performance_std = final_test_performance[1]
    return config.ended_early


def generate_backprop_plots(config, logs, show_plots=True):

    generator_losses_smoothed = gaussian_filter1d(logs.meta_losses, sigma=config.plot_smoothing_sigma)
    plot_fig([i for i in range(len(generator_losses_smoothed))],
             generator_losses_smoothed,
             title="Meta Losses", draw_grid=True, xlabel="steps", ylabel="generator loss", add_legend=True,
             label="generator loss", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Generator_losses.pdf'))

    critic_losses_smoothed = gaussian_filter1d(logs.critic_losses, sigma=config.plot_smoothing_sigma)
    plot_fig([i for i in range(len(critic_losses_smoothed))],
             critic_losses_smoothed,
             title="Critic Losses", draw_grid=True, xlabel="steps", ylabel="critic loss", add_legend=True,
             label="critic loss", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Critic_losses.pdf'))

    actor_performances_mean_plot_smoothed = gaussian_filter1d(logs.new_actor_performances_mean, sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(actor_performances_mean_plot_smoothed))],
             actor_performances_mean_plot_smoothed, std=logs.new_actor_performances_std,
             title="New actor performance", draw_grid=True, xlabel="steps", ylabel="a2c actor perf", add_legend=True,
             label="New actor performance", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Actor_perf_smoothed.pdf'),
             ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high],
             smooth_fill=False, smooth_fill_sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(logs.new_actor_performances_mean))],
             logs.new_actor_performances_mean,
             std=logs.new_actor_performances_std, title=str(config.algorithm)+" Actor Performance", draw_grid=True, xlabel="steps",
             ylabel="reward", add_legend=True, label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Actor_Perf.pdf'), ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high])


def generate_rl_plots(config, logs, show_plots=True):
    actor_losses_smoothed = gaussian_filter1d(logs.actor_losses, sigma=config.plot_smoothing_sigma)
    plot_fig([i for i in range(len(actor_losses_smoothed))],
             actor_losses_smoothed,
             title="Actor Losses", draw_grid=True, xlabel="steps", ylabel="actor loss", add_legend=True,
             label="actor loss", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Actor_losses.pdf'))

    critic_losses_smoothed = gaussian_filter1d(logs.critic_losses, sigma=config.plot_smoothing_sigma)
    plot_fig([i for i in range(len(critic_losses_smoothed))],
             critic_losses_smoothed,
             title="Critic Losses", draw_grid=True, xlabel="steps", ylabel="critic loss", add_legend=True,
             label="critic loss", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Critic_losses.pdf'))

    actor_performances_mean_plot_smoothed = gaussian_filter1d(logs.actor_performances_mean, sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(actor_performances_mean_plot_smoothed))],
             actor_performances_mean_plot_smoothed, std=logs.actor_performances_std,
             title="Actor performance", draw_grid=True, xlabel="steps", ylabel="reward", add_legend=True,
             label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Actor_perf_smoothed.pdf'),
             ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high],
             smooth_fill=False, smooth_fill_sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(logs.actor_performances_mean))],
             logs.actor_performances_mean,
             std=logs.actor_performances_std, title="Actor Performance", draw_grid=True, xlabel="steps",
             ylabel="reward", add_legend=True, label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'Actor_Perf.pdf'), ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high])


def generate_ga_plots(config, logs, show_plots=True):
    generator_performances_mean_smoothed = gaussian_filter1d(logs.generator_performance_mean,
                                                              sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(generator_performances_mean_smoothed))],
             generator_performances_mean_smoothed, std=logs.generator_performance_std,
             title="Generator-actor performance", draw_grid=True, xlabel="steps", ylabel="actor perf", add_legend=True,
             label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'GA-Generator-actor_perf_smoothed.pdf'), ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high], smooth_fill=True,
             smooth_fill_sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(logs.generator_performance_mean))],
             logs.generator_performance_mean,
             std=logs.generator_performance_std, title="Generator-actor performance", draw_grid=True, xlabel="steps",
             ylabel="Actor perf", add_legend=True, label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'GA-Generator-actor_perf.pdf'), ylim=[config.env_config.plot_performance_low, config.env_config.plot_performance_high])


def generate_all_logs(config, log):
    time_taken = time.time() - config.run_id
    config_file_name = os.path.join(config.log_path, 'config.log')
    with open(config_file_name, 'a+') as f:
        f.write('time_taken--' + str(time_taken) + '\n')
        variables = vars(config)
        for item in variables:
            f.write(str(item) + '--' + str(variables[item]))
            f.write('\n')
    log_file_name = os.path.join(config.log_path, 'data.log')
    with open(log_file_name, 'a+') as f:
        variables = vars(log)
        for item in variables:
            f.write(str(item) + '--' + str(variables[item]))
            f.write('\n')


def save_meta_models(generator, critic, save_path):
    torch.save(generator.state_dict(), os.path.join(save_path, 'generator.pt'))
    torch.save(critic.state_dict(), os.path.join(save_path, 'critic.pt'))


def save_rl_models(actor, critic, save_path):
    torch.save(actor.state_dict(), os.path.join(save_path, 'actor.pt'))
    torch.save(critic.state_dict(), os.path.join(save_path, 'critic.pt'))


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def generate_discrete_one_hot_output(action_space_size, num_generator_samples):
    # Creating expected output for the generator
    # num_generator_samples must be divisible by action_space_size
    with torch.no_grad():
        indices = list(np.linspace(0, num_generator_samples, num=action_space_size, endpoint=False, dtype=np.int8))
        inclusive_indices = list(np.linspace(0, num_generator_samples, num=action_space_size+1, dtype=np.int8))
        generator_one_hot_expected_actions = torch.zeros((num_generator_samples, action_space_size))
        for idx, num in enumerate(indices):
            generator_one_hot_expected_actions[num:inclusive_indices[idx+1], idx] += 1
        return generator_one_hot_expected_actions