import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
import os
import time
from scipy.ndimage.filters import gaussian_filter1d
from attempt_2.v6.models import get_random_agent, init_weights
import matplotlib
from itertools import repeat
import gym
import multiprocessing
import pickle
matplotlib.use("TkAgg")


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


def calculate_advantages(returns, values, normalize=True):
    # New A2C code method
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def test_agent_performance(agent, env, device, num_tests=10):
    agent.eval()
    rewards_so_far = []
    for i in range(num_tests):
        done = False
        observation = env.reset()
        i = 0
        while not done:
            action = agent(torch.Tensor([observation]).to(device))
            # action = np.argmax(action.detach().cpu().numpy())
            # action = np.argmax(action.detach().numpy())
            action = torch.argmax(action).item()
            observation, reward, done, info = env.step(action)
            i += 1
        rewards_so_far.append(i)
    # print(rewards_so_far)
    return np.mean(rewards_so_far), np.std(rewards_so_far)


def multi_help_test(agent, num_tests, env_name, device):
    # Function to just wrap around making function calls for multiprocessing
    return test_agent_performance(agent, gym.make(env_name), device, num_tests)


def test_generator_performance(generator, actual_test_env, config, generator_expected_actions,
                               generator_one_hot_expected_actions, multi=True):

    def get_generator_input():
        # Called per generator loop
        with torch.no_grad():
            generator_input_noise_vector = torch.randn((config.generator_input_sample_length, config.generator_noise_vector_size))
            generator_concatenated_input = torch.cat((generator_input_noise_vector, generator_expected_actions), 1)
        return generator_concatenated_input

    # generator.eval()
    multi_performances = []
    trained_agents = []
    for i in range(config.generator_testing_loops):
        new_actor = get_random_agent(config)
        # new_actor.apply(init_weights)
        new_actor_opt = torch.optim.SGD(new_actor.parameters(), lr=config.actor_inner_learning_rate)
        # new_actor.train()
        for inner_loop_num in range(config.inner_loop_iterations):
            new_actor_opt.zero_grad()
            actor_criterion = torch.nn.MSELoss(reduction='sum')
            softmax_actor_predicted_actions = new_actor(generator(get_generator_input()))
            new_actor_loss = actor_criterion(softmax_actor_predicted_actions, generator_one_hot_expected_actions)
            # print(new_actor_loss)
            new_actor_loss.backward()
            new_actor_opt.step()
        trained_agents.append(new_actor)
    if multi is False:
        performances_mean = []
        performances_std = []
        for agent in trained_agents:
            performance = test_agent_performance(agent, actual_test_env, config.dev, num_tests=config.actor_testing_loops)
            performances_mean.append(performance[0])
            performances_std.append(performance[1])
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 3)
        multi_performances = pool.starmap(multi_help_test, zip(trained_agents, repeat(config.actor_testing_loops), repeat(config.env_name), repeat(config.dev)))
        performances_mean, performances_std = zip(*multi_performances)
        pool.close()
    return np.mean(performances_mean), np.mean(performances_std)


def generate_all_plots(config, logs, show_plots=True):

    generator_losses_smoothed = gaussian_filter1d(logs.meta_losses, sigma=config.plot_smoothing_sigma)
    plot_fig([i for i in range(len(generator_losses_smoothed))],
             generator_losses_smoothed,
             title="A2C Meta Losses", draw_grid=True, xlabel="steps", ylabel="generator loss", add_legend=True,
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
             save_name=os.path.join(config.log_path, 'A2C_actor_perf_smoothed.pdf'), ylim=[0, 501], smooth_fill=True, smooth_fill_sigma=config.plot_smoothing_sigma)
    plot_fig([i * config.plot_save_iterations for i in range(len(logs.new_actor_performances_mean))],
             logs.new_actor_performances_mean,
             std=logs.new_actor_performances_std, title="A2C Actor Performance", draw_grid=True, xlabel="steps",
             ylabel="reward", add_legend=True, label="reward", display_fig=show_plots, save_fig=True,
             save_name=os.path.join(config.log_path, 'A2C_Actor_Perf.pdf'), ylim=[0, 501])


def generate_backprop_logs(config, log):
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


def generate_all_logs(config, logs):
    end_time = time.time()
    time_taken = end_time - config.run_id
    log_file_name = os.path.join(config.log_path, 'tests.log')

    with open(log_file_name, 'a+') as f:
        f.write('---------------------------')
        f.write('\n')
        f.write('Run ID' + str(config.run_id) + '\n')
        f.write('Configuration - CPU version \n')
        f.write('Outer loop iterations- ' + str(config.outer_loop_iterations) + '\n')
        f.write('Inner loop iterations- ' + str(config.inner_loop_iterations) + '\n')
        f.write('Time-taken -' + str(time_taken) + '\n')
        # f.write(
        #     'Testing both outer and inner loops, actor not training except with A2C loss, actor maintained across generations. \n')
        # f.write('Testing configurations of parameters and model formats \n')
        f.write('\n')
        f.write('Critic losses compiled' + str(logs.critic_losses))
        f.write('\n')
        f.write('Generator losses compiled' + str(logs.meta_losses))
        f.write('\n')
        f.write('Actor losses compiled' + str(logs.actor_losses))
        f.write('\n')
        f.write('New actor reward means compiled' + str(logs.new_actor_performances_mean))
        f.write('\n')
        f.write('New actor reward stds compiled' + str(logs.new_actor_performances_std))
        f.write('\n')
        f.write('---------------------------')


def generate_ga_logs(config, logs):
    end_time = time.time()
    time_taken = end_time - config.run_id
    log_file_name = os.path.join(config.log_path, 'tests.log')
    with open(log_file_name, 'a+') as f:
        f.write('---------------------------')
        f.write('\n')
        f.write('Run ID' + str(config.run_id) + '\n')
        # self.population_size = 200
        # self.num_of_generations = 20
        # self.elite_per_gen = 3
        # self.normal_evaluations = 5
        # self.elite_evaluations = 10
        # self.truncate_fraction = 0.5
        # self.mutation_power = 0.01
        # self.config.current_generation
        f.write('Configuration - CPU version \n')
        f.write('Genetic Algorithm implementation \n')
        f.write('num_of_generations- ' + str(config.num_of_generations) + '\n')
        f.write('Inner loop iterations- ' + str(config.inner_loop_iterations) + '\n')
        f.write('population_size- ' + str(config.population_size) + '\n')
        f.write('elite_per_gen- ' + str(config.elite_per_gen) + '\n')
        f.write('normal_evaluations- ' + str(config.normal_evaluations) + '\n')
        f.write('elite_evaluations- ' + str(config.elite_evaluations) + '\n')
        f.write('truncate_fraction- ' + str(config.truncate_fraction) + '\n')
        f.write('mutation_power- ' + str(config.mutation_power) + '\n')
        f.write('generations completed- ' + str(config.current_generation + 1) + '\n')
        f.write('Time-taken -' + str(time_taken) + '\n')
        f.write('Generator means compiled' + str(logs.generator_performance_mean))
        f.write('\n')
        f.write('Generator stds compiled' + str(logs.generator_performance_std))
        f.write('\n')
        f.write('---------------------------')

def save_all_models(generator, actor, critic, save_path):
    torch.save(generator.state_dict(), os.path.join(save_path, 'generator.pt'))
    torch.save(actor.state_dict(), os.path.join(save_path, 'actor.pt'))
    torch.save(critic.state_dict(), os.path.join(save_path, 'critic.pt'))

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
