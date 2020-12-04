import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import torch

from aux.common import DummyNormalizer, DummyStateScaler, CustomStateScaler

#  Making a cache of random actors so that the code is a bit streamlined and cleaner at this point. Changing other
#  types of models to dynamic counterparts as well. Having a cache will streamline things since they will already
#  exist in memory, and it'll be a bit faster, since we can just initialize as required.


class ActorCache:
    def __init__(self, config, num_actors=10000):
        if config.actor_type == 'standard':
            self.num_actors = 1
        elif config.actor_type == 'random':
            self.num_actors = num_actors
        self.device = config.dev
        self.state_normalizer = DummyNormalizer()
        if config.scale_states:
            self.state_scaler = CustomStateScaler(config.state_dim)
        else:
            self.state_scaler = DummyStateScaler()
        # The state scaler and state normalizer should be global, and be common across all actors.
        # TODO - verify this Python formulation later on.
        self.actors_list = [SequentialActor(config, self.state_scaler, self.state_normalizer) for _ in range(self.num_actors)]

    def sample(self):
        index = np.random.randint(0, self.num_actors)
        return self.actors_list[index].reset_weights()


class SequentialActor(nn.Module):
    # Returning to a dynamic actor generator, because Multiprocessing bugs aren't working in this situation atm anyway
    def __init__(self, config, global_state_scaler, global_state_normalizer, activation_type=F.leaky_relu, dim=None, num_layers=None):
        super().__init__()
        self.batch_norm = config.actor_batch_norm
        self.action_space_type = config.env_config.action_space_type
        self.batch_norm_momentum = 0.1
        self.state_scaler = global_state_scaler
        self.state_normalizer = global_state_normalizer
        self.family = config.family
        self.dev = config.dev

        if self.action_space_type == 'continuous':
            self.action_lower_limit = config.action_space_low[0]
            self.action_upper_limit = config.action_space_high[0]
            if self.family == 'stochastic':
                self.action_std_dev = 0.5  # TODO - Add as general param
                self.action_var_vector = torch.full((config.action_dim,), self.action_std_dev * self.action_std_dev).to(config.dev)
                self.covariance_matrix = torch.diag_embed(self.action_var_vector).to(config.dev)
            elif self.family == 'deterministic':
                # Nothing required, the value will just pass through as is.
                # Q-learning family of algorithms go here.
                pass

        if dim is None and num_layers is None:
            self.dim = [config.state_dim]
            if config.actor_type == 'random':
                self.num_layers = np.random.randint(config.actor_random_layer_range[0], config.actor_random_layer_range[1])
                for i in range(self.num_layers):
                    self.dim.append(np.random.randint(config.actor_random_dim_range[0], config.actor_random_dim_range[1]))
            elif config.actor_type == 'standard':
                self.num_layers = config.actor_standard_layer_num
                self.dim = self.dim + [config.actor_standard_dim_num for _ in range(self.num_layers)]
            self.dim += [config.action_dim]
        else:
            self.dim = dim
            self.num_layers = num_layers

        linear_layers = []
        for input_dim in range(len(self.dim) - 1):
            linear_layers.append(nn.Linear(self.dim[input_dim], self.dim[input_dim+1]))
        self.linear_layers = nn.ModuleList(linear_layers)

        if self.batch_norm:
            batch_norm_layers = []
            for input_dim in range(1, len(self.dim) - 1):
                batch_norm_layers.append(nn.BatchNorm1d(self.dim[input_dim], momentum=self.batch_norm_momentum))
            self.batch_norm_layers = nn.ModuleList(batch_norm_layers)
        self.activation_layers = [activation_type for _ in range(len(self.linear_layers)-1)]

    def prepare_state_for_actor(self, state, source):
        # 4 sources for inputs
        # 1 - test, which gives np.array(state_dim,) (needs modification)
        # 2 - rollout, which also gives np.array(state_dim,) (needs modification)
        # 3 - generator, which gives tensors of (sample_num, state_dim) (assumed proper)
        # 4 - ppo, which also gives tensor (sample_num, state_dim) (needs modification)
        if type(state) == np.ndarray:
            # In this scenario, we ensure that the output data has to be scaled and normalized, always env.
            out = state
            out = out.reshape(-1, self.dim[0])
            normalized_state = self.state_normalizer.normalize(out)
            scaled_state = self.state_scaler.scale_down(normalized_state)
            processed_state = torch.FloatTensor(scaled_state)
        else:
            # Assuming that the generator gives a normalized scaled state, so not bothering. But has to be detached,
            if source == 'env':  # In PPO, this is needed, since the states are tensors now, but the value is original.

                state = state.detach().numpy()
                state = self.state_normalizer.normalize(state)
                state = self.state_scaler.scale_down(state)
                processed_state = torch.FloatTensor(state).to(self.dev)
            else:
                processed_state = state.to(self.dev)
        return processed_state

    def forward(self, state, source='env'):
        out = self.prepare_state_for_actor(state, source)
        for layer in range(len(self.linear_layers) - 1):
            out = self.linear_layers[layer](out)
            out = self.activation_layers[layer](out)
            if self.batch_norm:
                out = self.batch_norm_layers[layer](out)
        out = self.linear_layers[-1](out)
        if self.action_space_type == 'discrete':
            out = F.softmax(out, dim=-1)
        elif self.action_space_type == 'continuous':
            out = torch.tanh(out)
        return out

    def get_action(self, action_prob, context='train'):
        if self.action_space_type == 'discrete':
            if context == 'train':
                dist = distributions.Categorical(action_prob)
                action = dist.sample()
                log_prob_action = dist.log_prob(action)
                return action, log_prob_action  # Both have to be tensors, or the processing gets weird later on.
            else:
                action = torch.argmax(action_prob).item()
                return action
        elif self.action_space_type == 'continuous':
            if self.family == 'stochastic':
                action_mean_vector = action_prob * self.action_upper_limit  # Scaling up to ensure that the value works.
                # Covariance matrix is defined in init. Does not change.
                dist = distributions.MultivariateNormal(action_mean_vector, self.covariance_matrix)
                action = dist.sample()
                # Clip action to ensure no erroneous action taken by accident.
                action = torch.clamp(action, min=self.action_lower_limit, max=self.action_upper_limit)
                log_prob_action = dist.log_prob(action)
                if context == 'train':
                    return action, log_prob_action
                else:
                    return action.detach().numpy()
            elif self.family == 'deterministic':
                # Action Prob here is actually just action itself. No probability.
                # Logic adapted from TD3 repo actors. Scaled up to max_action_possible, since tanh before.
                action = action_prob * self.action_upper_limit
                if context == 'train':
                    return action, None
                else:
                    return action.detach().numpy().reshape(self.dim[-1],)

    def reset_weights(self):
        self.apply(init_actor_weights)
        return self


class Generator(nn.Module):
    # If simple, 1 layer with 128 hidden units.
    # If complex, 2 layers with 256 each.
    # If complexer, 3 layers with 512 each, but not yet implemented
    def __init__(self, config, activation_type=F.leaky_relu):
        super().__init__()
        self.batch_norm = config.generator_batch_norm
        self.batch_norm_momentum = 0.1
        self.gen_type = config.generator_type
        self.observation_space_high = torch.FloatTensor(config.observation_space_high).to(config.dev)
        self.observation_space_low = torch.FloatTensor(config.observation_space_low).to(config.dev)

        if self.gen_type == 'simple':
            self.dims = [config.generator_input_size, 64, config.state_dim]
        elif self.gen_type == 'complex':
            self.dims = [config.generator_input_size, 128, 128, config.state_dim]
        elif self.gen_type == 'complexer':
            self.dims = [config.generator_input_size, 256, 256, 256, config.state_dim]

        linear_layers = []
        for layer in range(len(self.dims) - 1):
            linear_layers.append(nn.Linear(self.dims[layer], self.dims[layer+1]))
        self.linear_layers = nn.ModuleList(linear_layers)
        # if self.batch_norm:
        batch_norm_layers = []
        for layer in range(1, len(self.dims)):
            batch_norm_layers.append(nn.BatchNorm1d(self.dims[layer], momentum=self.batch_norm_momentum))
        self.batch_norm_layers = nn.ModuleList(batch_norm_layers)

        self.activation_layers = [activation_type for _ in range(len(self.batch_norm_layers))]
        self.apply(init_generator_weights)


    def forward(self, x):
        out = x
        for layer in range(len(self.linear_layers)):
            out = self.linear_layers[layer](out)
            out = self.activation_layers[layer](out)
            if self.batch_norm:
                out = self.batch_norm_layers[layer](out)

        # Clamping for safety.
        # out = torch.max(torch.min(out, self.observation_space_high), self.observation_space_low)
        # Assuming that the data learns to predict the best possible scaling by itself.
        return out


class Critic(nn.Module):
    # If simple, 1 layer with 64 hidden units.
    # If complex, 2 layers with 128 each.
    # If complexer, 3 layers with 256 each, but not yet implemented
    def __init__(self, config, activation_func=F.leaky_relu):
        super().__init__()
        self.batch_norm = config.critic_batch_norm
        self.batch_norm_momentum: float = 0.0
        self.critic_type = config.critic_type
        self.family = config.family
        # Default
        if self.family == 'deterministic':
            self.input_dim = config.state_dim + config.action_dim
        elif self.family == 'stochastic':
            self.input_dim = config.state_dim
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.dev = config.dev
        if self.critic_type == 'simple':
            self.fc_1 = nn.Linear(self.input_dim, 64)
            self.fc_2 = nn.Linear(64, config.reward_size)
            if self.batch_norm:
                self.bn_1 = nn.BatchNorm1d(64, momentum=self.batch_norm_momentum)
                self.bn_2 = nn.BatchNorm1d(config.reward_size, momentum=self.batch_norm_momentum)
                # Normalizing output of the generator may or may not give good results
        elif self.critic_type == 'complex':
            self.fc_3 = nn.Linear(self.input_dim, 128)
            self.fc_4 = nn.Linear(128, 128)
            self.fc_5 = nn.Linear(128, config.reward_size)
            if self.batch_norm:
                self.bn_3 = nn.BatchNorm1d(128, momentum=self.batch_norm_momentum)
                self.bn_4 = nn.BatchNorm1d(128, momentum=self.batch_norm_momentum)
                self.bn_5 = nn.BatchNorm1d(config.reward_size, momentum=self.batch_norm_momentum)
        self.apply(init_critic_weights)

    def forward(self, state, action=None):
        # Not doing any scaling in the critic model. This sees what the environment actually is.
        if type(state) == np.ndarray:
            tensor_state = torch.FloatTensor(state).squeeze(0).view(-1, self.state_dim).to(self.dev)
        elif type(state) == torch.Tensor:
            tensor_state = state.squeeze(0).view(-1, self.state_dim).to(self.dev)
        else:
            raise Exception('Unknown state type to critic')

        if self.family == 'deterministic':
            # Action has to be managed here as well.
            if type(action) == np.ndarray:
                tensor_action = torch.FloatTensor(action).squeeze(0).view(-1, self.action_dim).to(self.dev)
            elif type(action) == torch.Tensor:
                tensor_action = action.squeeze(0).view(-1, self.action_dim).to(self.dev)
            else:
                raise Exception('Unknown action type to critic')
            tensor_input = torch.cat([tensor_state, tensor_action], 1)
        elif self.family == 'stochastic':
            tensor_input = tensor_state
        else:
            raise Exception('Unknown family type in critic')

        out = tensor_input
        # print('input to critic', out.shape, out)
        if self.critic_type == 'simple':
            out = self.fc_1(out)
            out = F.relu(out)
            # out = torch.tanh(out)
            if self.batch_norm:
                out = self.bn_1(out)
            out = self.fc_2(out)


        elif self.critic_type == 'complex':
            out = self.fc_3(out)
            out = F.relu(out)
            # out = torch.tanh(out)
            if self.batch_norm:
                out = self.bn_3(out)
            out = self.fc_4(out)
            out = F.relu(out)
            # out = torch.tanh(out)
            if self.batch_norm:
                out = self.bn_4(out)
            out = self.fc_5(out)

        return out


def init_actor_weights(self):
    for m in self.modules():
        if isinstance(m, nn.BatchNorm1d):
            # print('Init BN Actor Layer', m)
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # print('Init Linear Actor Layer', m)
            nn.init.orthogonal_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.orthogonal_(m.weight, gain=1.41)
            # nn.init.kaiming_normal_(m.weight)
            # nn.init.sparse_(m.weight, sparsity=0.5)
            m.bias.data.fill_(0)


def init_generator_weights(self):
    for m in self.modules():
        if isinstance(m, nn.BatchNorm1d):
            # print('Init BN Generator Layer')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # print('Init Linear Generator Layer')
            nn.init.orthogonal_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)


def init_critic_weights(self):
    for m in self.modules():
        if isinstance(m, nn.BatchNorm1d):
            # print('Init BN Critic Layer')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # print('Init BN Critic Layer')
            nn.init.orthogonal_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)
