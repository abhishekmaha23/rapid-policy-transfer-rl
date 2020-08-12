# sys.stdout = open('/home/pycharmuser/shared/logs/log_2.txt', 'w')
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from generators import SyntheticDataTestEnv

STATE_DIM = 128
ACTION_DIM = 2048
# STEP = 1
# SAMPLE_NUMS = 2
STEP = 20000
SAMPLE_NUMS = 50


class ActorNetwork(nn.Module):
    # Used by direct A2C implementation
    # Creating Actor Network of size (128, 64, 64, 2048)
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        print('Creating Actor Network of size', input_size, hidden_size, hidden_size, action_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        # out = F.log_softmax(self.fc3(out))
        return out


class ValueNetwork(nn.Module):
    # Used by direct A2C implementation
    # Creating Value Network of size (2048, 64, 64, 1)
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        print('Creating Value Network of size', input_size, hidden_size, hidden_size, output_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out


def roll_out(actor_network, env, sample_nums, value_network, init_state, step):
    # Creates samples per step, a number of (generated_data, reward) values in lists.

    #env.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)

        action = actor_network(torch.Tensor(state))
        next_state, reward, done, _ = env.step(action)
        actions.append(action.detach())
        rewards.append(reward)
        final_state = next_state
        state = next_state
        state = env.reset()
        # if done:
        #     is_done = True
        #     state = env.reset()
        #     break
    state = env.reset()
    return states, actions, rewards, final_r, state


def direct_a2c_main():
    env = SyntheticDataTestEnv()
    init_state = env.reset()

    value_network = ValueNetwork(input_size=ACTION_DIM, hidden_size=64, output_size=1)
    value_network_optim = torch.optim.SGD(value_network.parameters(), lr=0.1)

    actor_network = ActorNetwork(STATE_DIM, 64, ACTION_DIM)
    actor_network_optim = torch.optim.SGD(actor_network.parameters(), lr=0.01)

    steps = []
    task_episodes = []
    test_results = []

    for step in range(STEP):
        states, actions, rewards, final_r, current_state = roll_out(actor_network, env, SAMPLE_NUMS, value_network,
                                                                    init_state, step)
        init_state = current_state
        actions_var = torch.stack(actions)

        states_var = np.stack(states)

        # train actor network
        actor_network_optim.zero_grad()

        output_data_from_actor = actor_network(torch.tensor(states_var).float())
        # output_data_from_actor[:, 0] = 4.8 * output_data_from_actor[:, 0]
        # output_data_from_actor[:, 1] = 10 * output_data_from_actor[:, 1]
        # output_data_from_actor[:, 2] = 24 * output_data_from_actor[:, 2]
        # output_data_from_actor[:, 3] = 10 * output_data_from_actor[:, 3]
        
        vs = value_network(actions_var).detach()
        
        qs = torch.Tensor(rewards).reshape(-1, 1)

        advantages = qs - vs
        actor_network_loss = - torch.mean(torch.sum(output_data_from_actor * actions_var, 1) * advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 0.5)
        actor_network_optim.step()

        # train value network
        value_network_optim.zero_grad()
        target_values = qs

        values = value_network(actions_var)

        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_network.parameters(), 0.5)
        value_network_optim.step()

        if (step + 1) % 50 == 0:

            result = 0
            test_task = SyntheticDataTestEnv()
            for test_epi in range(10):
                state = test_task.reset()
                for test_step in range(200):
                    action = actor_network(torch.tensor(state).float())
                    next_state, reward, done, _ = test_task.step(action)
                    result += reward
                    state = next_state
                    if done:
                        break
            print("step:", step + 1, "test result:", result / 10.0)
            steps.append(step + 1)
            test_results.append(result / 10)
    print('All test results in all steps', test_results)


def stable_baselines_main():
    synthetic_data_test_env = SyntheticDataTestEnv()

    print('=====')
    check_env(synthetic_data_test_env)
    print('=====')
    # model = PPO2(MlpPolicy, synthetic_data_test_env, verbose=1, tensorboard_log="/home/pycharmuser/shared/tensorboard_logs/synthetic_test_env_cartpole_5")
    model = A2C(MlpPolicy, synthetic_data_test_env, verbose=1, max_grad_norm=1, n_steps=10, learning_rate=0.1, tensorboard_log="/home/pycharmuser/shared/tensorboard_logs/synthetic_test_env_cartpole_5")
    model.learn(total_timesteps=40000)
    model.save("a2c-synthetic-env_5")


if __name__ == '__main__':
    # Possibly incorrect usage, since the synthetic data environment has space as noise. The critic module needs to
    # predict an output independent of state, but noise doesn't help
    # stable_baselines_main()

    # Manual implementation, using A2C code obtained from the internet. The modification done is to use the generated
    # data to predict the output, instead of noise. However, this doesn't work very well either.
    direct_a2c_main()
