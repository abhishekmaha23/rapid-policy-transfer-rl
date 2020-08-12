import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import gym
from gym import spaces


class SyntheticDataAgent(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = self.fc1(x)
        # out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = torch.sigmoid(out) - 0.5
        # out = nn.Softmax(dim=1)(out)
        out = nn.LogSoftmax(dim=1)(out)
        return out


class SyntheticDataTestEnv(gym.Env):
    '''
    Custom Environment that follows gym interface
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SyntheticDataTestEnv, self).__init__()    # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.agent = SyntheticDataAgent(4, 40, 2)
        self.agent_initial_parameters = self.agent.state_dict()
        a = np.ones((512, 4))
        a = np.transpose(a.reshape(-1, 1))[0]
        # Action space is the space of observations
        self.action_space = spaces.Box(low=-a, high=a, shape=a.shape, dtype=np.float64)

        # Observation space is the noise given to the generator. It's going to be different each time, but for simple
        # purposes, let us assume that there is a trajectory of 1 steps.
        # Since both upper and lower 2D bounds are provided, noise is uniformly distributed, which should lead to
        # properly homogenous results.
        self.observation_space = spaces.Box(low=-np.ones(128), high=np.ones(128), dtype=np.float64)

        a = np.zeros((256, 1))
        b = np.ones((256, 1))
        self.expected_actions = torch.tensor(np.vstack((a, b))).float()
        self.steps_crossed_in_trajectory = 0
        self.done = False

        self.obtained_rewards = []

    def compute_loss(self, data):
        tensor_data = torch.tensor(data)

        model_copy = type(self.agent)(4, 40, 2)  # get a new instance
        agent_optimizer = optim.SGD(model_copy.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        number_of_training_steps = 10
        # Train the agent on this data.
        model_copy.train()

        for train_step in range(number_of_training_steps):
            # The agent must learn from the training data, and do one backprop step to complete character arc
            # The agent must also reset itself before the start of this loop
            agent_optimizer.zero_grad()  # zero the gradient buffers
            y_predict = model_copy(tensor_data.float())
            y_predicted_actions = (torch.argmax(torch.exp(y_predict), axis=1)).view(-1, 1)
            y_predicted_values = (torch.max(torch.exp(y_predict), dim=1)).values.view(-1, 1)
            loss = criterion(y_predicted_values, self.expected_actions)
            loss.backward()
            agent_optimizer.step()

        temp_env = gym.make("CartPole-v0")
        init_state = temp_env.reset()
        state = init_state

        model_copy.eval()
        # print('Agent trained on', number_of_training_steps, 'step(s) of generated data.')
        # print('Attempting to run simulation with the trained agent.')
        length_of_simulation = 0
        for i in range(250):
            # print('Step in trained agent ', i)

            state = torch.from_numpy(state).float()

            action = model_copy(state.view(1, 4))
            action = (torch.argmax(torch.exp(action), axis=1))[0].numpy()
            observation, reward, done, _ = temp_env.step(action)
            state = observation
            length_of_simulation += 1
            if done:
                self.obtained_rewards.append(i+1)
                break
        return np.mean(self.obtained_rewards)

    def step(self, action):
        # Done happens every step, so...
        # action refers to the (2048,) (actually (512, 4)) data matrix that's supposed to come out of the PPO agent
        # outside, and this needs to go to the supervised learning agent and train it.

        action = action.detach().reshape(4, 512)
        action = np.transpose(action)
        action[:, 0] = 4.8 * action[:, 0]
        action[:, 1] = 10 * action[:, 1]
        action[:, 2] = 24 * action[:, 2]
        action[:, 3] = 10 * action[:, 3]

        reward = self.compute_loss(action)

        # Execute one time step within the environment
        self.steps_crossed_in_trajectory += 1

        self.done = True
        # if self.steps_crossed_in_trajectory % 10 == 0:
        #     self.done = True

        return self.observation_space.sample(), reward, self.done, {}

    def reset(self):
        # Nothing to reset, really. Done happens every step.
        # Changing done to happen every 10 steps.
        # print('Resetting environment after', self.steps_crossed_in_trajectory, ' (usually 10) steps')

        if self.steps_crossed_in_trajectory != 0:
            # print('Average over last trajectory was ', np.mean(self.obtained_rewards))
            pass
        self.obtained_rewards = []
        self.steps_crossed_in_trajectory = 0
        self.done = False
        # Reset the state of the environment to an initial state
        return self.observation_space.sample()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # Nothing to render
        pass