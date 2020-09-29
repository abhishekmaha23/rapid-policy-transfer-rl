import gym
import higher
from tqdm import tqdm
import torch.distributions as distributions

from .utils import *
from .models import *

'''A2C code adapted from https://github.com/bentrevett/pytorch-rl'''

'''
Version 5 todo
--- Clean up code, separate functions, isolate components and include tests within tests.
--- Save all models within a single file, and also save the model structure in those files.
--- Reuse old log files if A2C loss calculation is the part having errors.
'''


class Config:
    def __init__(self, run_id):
        self.run_id = run_id
        self.env_name = 'CartPole-v0'

        # log_path = os.path.join('/home/TUE/20181091/v4/logs', str(run_id)) # Cluster running
        self.log_path = os.path.join(os.getcwd(), str(run_id))

        # Reproducibility efforts // Skipping since this seems to cause issues
        # seed = 0
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)

        self.plot_save_iterations = 10 # Only used for intervals of testing agent performance, rest Gaussian smoothed
        self.plot_smoothing_sigma = 100

        # GPU efforts
        # if torch.cuda.is_available():
        #     dev = "cuda:0"
        # else:
        #     dev = "cpu"
        self.dev = "cpu"

        self.outer_loop_iterations = 3000
        self.inner_loop_iterations = 1

        self.state_size = 4
        self.action_size = 2
        self.actor_hidden_layer_sizes = 64

        self.generator_noise_vector_size = 8  # Per action sample comes with a vector of length 8 atm
        self.generator_action_vector_size = 1
        self.generator_input_size = self.generator_noise_vector_size + self.generator_action_vector_size
        self.generator_hidden_layer_sizes = 64

        self.critic_hidden_layer_sizes = 64

        self.reward_size = 1
        self.discount_factor = 0.99
        self.number_of_a2c_samples = 30

        self.generator_input_sample_length = 512
        # self.generator_input_sample_length = 20


class Logs:
    def __init__(self):
        # Plotting statistics
        # Dirty code
        self.actor_losses = []
        self.generator_losses = []
        self.critic_losses = []
        self.new_actor_performances_mean = []
        self.new_actor_performances_std = []


def main():

    config = Config(run_id)
    logs = Logs()

    actual_test_env = gym.make(config.env_name)
    rollout_env = gym.make(config.env_name)

    generator = MLP(config.generator_input_size, config.generator_hidden_layer_sizes, config.state_size)
    # actor = MLPActor(config.state_size, config.actor_hidden_layer_sizes, config.action_size)
    critic = MLP(config.state_size, config.critic_hidden_layer_sizes, 1)
    generator.apply(init_weights)
    # actor.apply(init_weights)
    critic.apply(init_weights)
    learning_rate = 0.01
    generator_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
    # actor_opt = torch.optim.SGD(actor.parameters(), lr=learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    current_env_state = rollout_env.reset()

    # actor_opt.zero_grad()
    generator_opt.zero_grad()
    critic_opt.zero_grad()

    # creating expected output for the generator
    with torch.no_grad():
        # Declaring expected output, a single vector of 0s and 1s
        generator_expected_actions = torch.zeros((config.generator_input_sample_length), requires_grad=False)
        generator_expected_actions[config.generator_input_sample_length // 2:] += 1
        generator_expected_actions = generator_expected_actions.view(-1, 1)

        # Creating a one-hot encoded vector
        generator_flipped_actions = generator_expected_actions.flip(0).view(-1, 1)
        generator_one_hot_expected_actions = torch.cat((generator_flipped_actions, generator_expected_actions), 1)

    for outer_loop_num in tqdm(range(config.outer_loop_iterations)):
        actor = getRandomAgent(config) # Getting a random agent with an architecture that's sampled from a random hidden layer size each iteration.
        actor.apply(init_weights)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        actor_opt.zero_grad()

        generator.train()
        actor.train()
        critic.train()

        inner_loop(outer_loop_num, config, actor, actor_opt, generator, generator_opt, critic, critic_opt,
                                        actual_test_env, rollout_env, logs,
                                        generator_expected_actions, generator_one_hot_expected_actions)
        # Tested to only modify generator parameters (as expected)
        generator_opt.step()

        if outer_loop_num % config.plot_save_iterations == 0:
            if config.inner_loop_iterations > 0:
                generator_performance = test_generator_performance(generator, actual_test_env, config,
                                                                   generator_expected_actions,
                                                                   generator_one_hot_expected_actions)
                logs.new_actor_performances_mean.append(generator_performance[0])
                logs.new_actor_performances_std.append(generator_performance[1])

    # Log directory creation
    os.mkdir(config.log_path)

    # Preliminary plots
    generate_all_plots(config, logs)

    # Saving statistics
    generate_all_logs(config, logs)

    # Saving all the models saved here
    save_all_models(generator, actor, critic, config.log_path)


def get_generator_input(config, generator_expected_actions):
    with torch.no_grad():
        generator_input_noise_vector = torch.randn((config.generator_input_sample_length, config.generator_noise_vector_size),
                                                   requires_grad=False)
        generator_concatenated_input = torch.cat((generator_input_noise_vector, generator_expected_actions), 1)
    return generator_concatenated_input


def inner_loop(outer_loop_num, config, actor, actor_opt, generator, generator_opt, critic, critic_opt, actual_test_env,
               rollout_env, logs, generator_expected_actions, generator_one_hot_expected_actions):
    actor_inner_opt = torch.optim.SGD(actor.parameters(), lr=0.1)
    with higher.innerloop_ctx(actor, actor_inner_opt, copy_initial_weights=False) \
            as (fnet_actor, diff_act_opt):
        # Changing differentiable optimizer inside the loop to SGD as follows
        for inner_loop_num in range(config.inner_loop_iterations):
            actor_criterion = torch.nn.MSELoss(reduction='sum')

            softmax_actor_predicted_actions = fnet_actor(generator(get_generator_input(config, generator_expected_actions)))
            actor_loss = actor_criterion(softmax_actor_predicted_actions, generator_one_hot_expected_actions)

            logs.actor_losses.append(actor_loss.item())

            # Train the actor with a differentiable optimizer.
            diff_act_opt.step(actor_loss)

        # Replacing logic here with a different implementation of A2C
        log_prob_actions = []
        values = []
        rewards = []
        done = False
        episode_reward = 0

        state = rollout_env.reset()
        n = 0
        while not done:
            n += 1
            state = torch.FloatTensor(state).unsqueeze(0)
            action_prob = fnet_actor(state)
            value_pred = critic(state)
            # action_prob = F.softmax(action_pred, dim=-1)

            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            state, reward, done, _ = rollout_env.step(action.item())

            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)

            episode_reward += reward

        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)

        # Discounted normalized returns and advantages
        returns = calculate_returns(rewards, config.discount_factor)
        advantages = calculate_advantages(returns, values)

        advantages = advantages.detach()
        returns = returns.detach()

        policy_loss = - (advantages * log_prob_actions).sum()
        value_loss = F.smooth_l1_loss(returns, values).sum()

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        actor_opt.step()  # The gradients should backpropagate to the outside and the generator_opt should step there.
        critic_opt.step()

        logs.critic_losses.append(value_loss.item())
        logs.generator_losses.append(policy_loss.item())

        # if outer_loop_num % config.plot_save_iterations == 0:
        #     actor_performance = test_agent_performance(actor, actual_test_env, config.dev)
        #     logs.actor_performances_mean_plot.append(actor_performance[0])
        #     logs.actor_performances_std_plot.append(actor_performance[1])

if __name__ == '__main__':
    run_id = time.time()
    main()