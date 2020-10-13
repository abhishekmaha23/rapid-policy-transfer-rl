import gym
import higher
from tqdm import tqdm
import torch.distributions as distributions

import random

from attempt_2.v6.utils import *
from attempt_2.v6.models import *
from attempt_2.v6.tests import *

'''A2C code adapted from https://github.com/bentrevett/pytorch-rl'''

'''
Version 6 todo
--- Clean up code, separate functions, isolate components.
--- Save config properly in logs (done)
--- Change layers in MLPActor for the generator to use something different that might work.
--- Implement resume functionality with run_id
--- Implement variations based on changing flow for LSTMs and recurrent flow
--- Implement a loop that identifies good architecture design for the generator
--- Fix GAs to use new functions that may or may not have broken with this functionality. Isolate if possible.
'''


class Config:
    def __init__(self, run_id):
        self.run_id = run_id

        self.env_name = 'CartPole-v1'
        # self.comment = 'Testing code restructuring and verifying logs are still being generated in proper format'
        # self.comment = 'Testing addition of learning rate scheduler to verify that there is learning'
        # self.comment = 'Testing different range of learning rates'
        self.comment = 'Testing on CartPole-v1'
        self.log_path = os.path.join(os.getcwd(), 'logs', 'supervised', str(run_id))

        # Reproducibility efforts // Skipping since this seems to cause issues
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)

        self.plot_save_iterations = 100  # Only used for intervals of testing agent performance, rest Gaussian smoothed
        self.plot_smoothing_sigma = 6

        # GPU efforts
        # if torch.cuda.is_available():
        #     dev = "cuda:0"
        # else:
        #     dev = "cpu"
        self.dev = "cpu"

        self.outer_loop_iterations = 30001
        self.inner_loop_iterations = 1

        self.state_size = 4
        self.action_size = 2
        self.actor_hidden_layer_sizes = 64

        self.generator_noise_vector_size = 32  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size =  32
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_hidden_layer_sizes = 128
        self.generator_input_sample_length = 512

        self.critic_hidden_layer_sizes = 128

        self.reward_size = 1
        self.discount_factor = 0.99
        self.number_of_a2c_samples = 30

        self.batch_norm = True

        self.critic_learning_rate = 0.01

        self.generator_learning_rate = 2e-1
        self.generator_min_lr_scaler = 1e-1

        self.actor_learning_rate = 1e-1
        self.actor_inner_learning_rate = 1e-1

        self.generator_testing_loops = 10  # Number of random agents tested
        self.actor_testing_loops = 10  # Number of tests per agent


class Logs:
    def __init__(self, config):
        # Plotting statistics
        self.config = config
        self.actor_losses = []
        self.meta_losses = []  # A2C losses that have backpropagated across to the generator
        self.critic_losses = []
        self.new_actor_performances_mean = []
        self.new_actor_performances_std = []

    def start(self):
        # Log directory creation
        os.makedirs(self.config.log_path, exist_ok=True)

    def save(self, generator, actor, critic):
        # Final plots
        generate_all_plots(self.config, self)

        # Saving statistics
        # generate_all_logs(self.config, self)
        generate_backprop_logs(self.config, self)

        # Saving all the models saved here
        save_all_models(generator, actor, critic, self.config.log_path)

        # Save config
        save_object(self.config, os.path.join(self.config.log_path, 'config.pkl'))


def main():
    # if resume is True:
    #
    # else:
    config = Config(run_id)
    logs = Logs(config)

    actual_test_env = gym.make(config.env_name)
    rollout_env = gym.make(config.env_name)

    generator = MLP(config.generator_input_size, config.generator_hidden_layer_sizes, config.state_size,
                    config.batch_norm)
    critic = MLP(config.state_size, config.critic_hidden_layer_sizes, 1)
    generator_opt = torch.optim.RMSprop(generator.parameters(), lr=config.generator_learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    generator_opt.zero_grad()
    critic_opt.zero_grad()

    # Testing scheduler
    generator_opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_opt, config.outer_loop_iterations, config.generator_learning_rate * config.generator_min_lr_scaler)

    # Creating expected output for the generator
    with torch.no_grad():
        # Declaring expected output, a matrix of 0s and 1s
        generator_expected_actions = torch.zeros(
            (config.generator_input_sample_length, config.generator_action_vector_size))
        generator_expected_actions[config.generator_input_sample_length // 2:] += 1

        # Creating a one-hot encoded vector
        generator_expected_actions_column = torch.zeros((config.generator_input_sample_length, 1))
        generator_expected_actions_column[config.generator_input_sample_length // 2:] += 1
        generator_flipped_actions = generator_expected_actions_column.flip(0)
        generator_one_hot_expected_actions = torch.cat((generator_flipped_actions, generator_expected_actions_column), 1)

    logs.start()

    for outer_loop_num in tqdm(range(config.outer_loop_iterations)):
        # Getting a random agent with an architecture that's sampled from a random hidden layer size each iteration.
        actor = get_random_agent(config, batch_norm=config.batch_norm)
        actor_opt = torch.optim.SGD(actor.parameters(), lr=config.actor_learning_rate)
        actor_opt.zero_grad()

        generator.train()
        actor.train()
        critic.train()

        inner_loop(outer_loop_num, config, actor, actor_opt, generator, generator_opt, critic, critic_opt,
                   actual_test_env, rollout_env, logs,
                   generator_expected_actions, generator_one_hot_expected_actions)
        generator_opt.step()  # Second-order gradient update
        generator_opt_scheduler.step() # Slowly reducing the learning rate of the generator to a minimal value.
        if outer_loop_num % config.plot_save_iterations == 0:
            if config.inner_loop_iterations > 0:
                generator_performance = test_generator_performance(generator, actual_test_env, config,
                                                                   generator_expected_actions,
                                                                   generator_one_hot_expected_actions, multi=True)
                tqdm.write('Performance' + str(generator_performance))
                logs.new_actor_performances_mean.append(generator_performance[0])
                logs.new_actor_performances_std.append(generator_performance[1])
                # Trying some more code
                if generator_performance[0] > 200:
                    # tqdm.write('Reached a decent level, testing more')
                    extra_generator_performance = test_generator_performance(generator, actual_test_env, config,
                                                                       generator_expected_actions,
                                                                       generator_one_hot_expected_actions, multi=True)
                    tqdm.write('Retest performance for high levels' + str(extra_generator_performance))

            else:
                actor_performance = test_agent_performance(actor, actual_test_env, config.dev)
                logs.new_actor_performances_mean.append(actor_performance[0])
                logs.new_actor_performances_std.append(actor_performance[1])
            # generate_all_plots(config, logs, show_plots=False)

    logs.save(generator, actor, critic)


def rollout(fnet_actor, critic, rollout_env):
    # Rollout out here does not focus on getting a certain amount of samples - it instead just completes one entire
    # trajectory. This approach towards A2C worked perfectly and caused training within 200 steps.
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    state = rollout_env.reset()

    n = 1
    fnet_actor.eval()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_prob = fnet_actor(state)  # Using this here, instead of the normal actor allows higher-order gradients
        value_pred = critic(state)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        state, reward, done, _ = rollout_env.step(action.item())
        log_prob_actions.append(log_prob_action)

        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

        n += 1

    # This causes a backflow of gradients even when the batch-normalized model is called in eval mode, which is
    # weirdly interesting. This shouldn't happen since running_mean isn't calculated excellently.
    # Mild cause for concern. But only in the sense that the scaling isn't perfect, the generator model still
    # get a reasonable gradient, and there is demonstratable learning.
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    return log_prob_actions, rewards, values


def get_generator_input(config, generator_expected_actions):
    with torch.no_grad():
        generator_input_noise_vector = torch.randn(
            (config.generator_input_sample_length, config.generator_noise_vector_size))
        generator_concatenated_input = torch.cat((generator_input_noise_vector, generator_expected_actions), 1)
    return generator_concatenated_input


def inner_loop(outer_loop_num, config, actor, actor_opt, generator, generator_opt, critic, critic_opt, actual_test_env,
               rollout_env, logs, generator_expected_actions, generator_one_hot_expected_actions):
    actor_inner_opt = torch.optim.SGD(actor.parameters(), lr=config.actor_inner_learning_rate)
    actor_inner_opt.zero_grad()
    with higher.innerloop_ctx(actor, actor_inner_opt, copy_initial_weights=False) \
            as (fnet_actor, diff_act_opt):
        # Changing differentiable optimizer inside the loop to SGD as follows
        for inner_loop_num in range(config.inner_loop_iterations):
            actor_criterion = torch.nn.MSELoss(reduction='sum')

            softmax_actor_predicted_actions = fnet_actor(
                generator(get_generator_input(config, generator_expected_actions)))
            actor_loss = actor_criterion(softmax_actor_predicted_actions, generator_one_hot_expected_actions)

            logs.actor_losses.append(actor_loss.item())

            # Train the actor with a differentiable optimizer.
            diff_act_opt.step(actor_loss)

        # Do rollout and get outputs for loss calculation
        log_prob_actions, rewards, values = rollout(fnet_actor, critic, rollout_env)

        # Discounted normalized returns and advantages
        returns = calculate_returns(rewards, config.discount_factor)
        advantages = calculate_advantages(returns, values)

        advantages = advantages.detach()
        returns = returns.detach()

        policy_loss = - (advantages * log_prob_actions).sum()
        value_loss = F.smooth_l1_loss(returns, values).sum()

        actor_opt.zero_grad()
        generator_opt.zero_grad()
        critic_opt.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        actor_opt.step()  # The gradients backpropagate to the generator and the generator_opt makes a step there.
        critic_opt.step()

        logs.critic_losses.append(value_loss.item())
        logs.meta_losses.append(policy_loss.item())


if __name__ == '__main__':
    run_id = time.time()
    # resume = True
    # resume_id = '1602458398.3501544'
    main()
