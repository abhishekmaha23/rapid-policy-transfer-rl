import torch
import numpy as np

# Memory code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# And code-base of TD3. Part of the code is not mine. I really do stand upon the shoulders of giants.
# Seriously, not gonna be making any money off of this, so meh.


# Expects tuples of (state, next_state, action, reward, done)
# Changing to tuples of (state, next_state, action, reward, done, log_prob_actions)
class MemoryCache(object):
    def __init__(self, memory_type='onpolicy', max_size=5e4, batch_size=None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.memory_type = memory_type
        self.batch_size = batch_size

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self):
        states, next_states, actions, rewards, dones, log_prob_actions = [], [], [], [], [], []
        # Weird speedup, but taken from the official python docs as a loop speed up hack
        states_append = states.append  # state
        nextstates_append = next_states.append  # next state  - not majorly used in the policy-gradient family.
        actions_append = actions.append  # action
        rewards_append = rewards.append  # reward
        dones_append = dones.append  # done
        log_prob_actions_append = log_prob_actions.append  # log_prob_actions - not used here in the Q-learning family,
        # just keeping for a bit of symmetry.
        make_np_array = np.array
        # This isn't exactly perfect, since there's an offpolicy stochastic policy also (SAC), but here there's only
        # offpolicy deterministic
        if self.memory_type == 'offpolicy':
            # Randomly sampling from the memory cache
            indices = np.random.randint(0, len(self.storage), size=self.batch_size)
            for index in indices:
                X, Y, U, R, D, L = self.storage[index]  # Unpacked tuples. SS'ARD-I mean, SARSDL was literally there xD
                states_append(make_np_array(X, copy=False))
                nextstates_append(make_np_array(Y, copy=False))
                actions_append(make_np_array(U, copy=False))
                rewards_append(R)
                dones_append(D)
                log_prob_actions_append(None)
            # log_prob_actions = make_np_array([None for _ in indices])
        elif self.memory_type == 'onpolicy':
            # Get all memory, perhaps in order, but not mandatory.
            for ind in self.storage:
                X, Y, U, R, D, L = ind  # Unpacked tuples. SS'ARDL  # I mean, SARSDL was literally there xD
                states_append(make_np_array(X, copy=False))
                nextstates_append(make_np_array(Y, copy=False))
                actions_append(make_np_array(U, copy=False))
                rewards_append(R)
                dones_append(D)
                log_prob_actions_append(L)  # log_prob_actions is a tensor here.
            del self.storage[:]

        # All should be lists here.
        return states, next_states, actions, rewards, dones, log_prob_actions


def get_rollout_action(config, state, actor, env):
    if config.family == 'deterministic':
        log_prob = None
        # Q-learning type
        if config.current_episode < config.offpol_start_episodes:
            action = torch.FloatTensor(env.action_space.sample())
        else:
            action, _ = actor.get_action(actor(state))  # just to get the action itself. Gradients not needed here.
            action = action.flatten()
            noise = torch.empty(action.shape).normal_(mean=0, std=config.offpol_expl_noise)
            action += noise
            action = torch.clamp(action, min=actor.action_lower_limit, max=actor.action_upper_limit)
            # action = (action + np.random.normal(0, config.offpol_expl_noise, size=env.action_space.shape[0])).clip(
            #     env.action_space.low, env.action_space.high)

    elif config.family == 'stochastic':
        action, log_prob = actor.get_action(actor(state))
    else:
        raise Exception("Can't select rollout action")
    return action, log_prob


def rollout(config, actor, rollout_env, memory_cache):
    current_state = rollout_env.reset()
    trajectory_rewards = []
    trajectory_lengths = []
    done = False
    action_space_type = config.env_config.action_space_type
    num_samples_collected = 0
    current_trajectory_timestep = 0
    current_trajectory_reward = 0
    actor.eval()  # Functional actor in meta, normal in rl, either way, batch normalization, so eval.
    while True:
        # State processing moved to inside the actor. Separating of env is cleaner this way.
        # Choose action, and perform.
        # action, log_prob_action = actor.get_action(actor(current_state))
        action, log_prob_action = get_rollout_action(config, current_state, actor, rollout_env)
        if action_space_type == 'discrete':

            valid_action = action.item()
        elif action_space_type == 'continuous':
            valid_action = action.detach().numpy()
            if config.family == 'deterministic':
                action = valid_action
            valid_action = valid_action.reshape(config.action_dim,)

        new_state, step_reward, done, _ = rollout_env.step(valid_action)
        # Store transition data and update current_state
        memory_cache.add((current_state, new_state, action, step_reward, done, log_prob_action))
        current_trajectory_reward += step_reward
        num_samples_collected += 1
        current_trajectory_timestep += 1
        current_state = new_state
        if config.rollout_update_type == 'trajectory' and (done or current_trajectory_timestep > config.env_max_timesteps):
            trajectory_rewards.append(current_trajectory_reward)
            trajectory_lengths.append(current_trajectory_timestep)
            break  # No more rollout steps required, collected stuff is enuf.
        elif config.rollout_update_type == 'batch' and num_samples_collected >= config.onpol_batch_size:
            break  # Same, because batch is filled.
        elif config.rollout_update_type == 'batch' and (done or current_trajectory_timestep > config.env_max_timesteps) and num_samples_collected < config.onpol_batch_size:
            current_state = rollout_env.reset()
            trajectory_lengths.append(current_trajectory_timestep)
            current_trajectory_timestep = 0
            trajectory_rewards.append(current_trajectory_reward)
            current_trajectory_reward = 0
            done = False
        # NOTE - Storing only step rewards here.
        # Discount factor in off-policy is shuffled, so only works for the each data point.
        # Discount factor in on-policy requires the data from each point, so we have to either do the cumulative stuff
        # here, or do it at the algorithm step.
        # Thus choosing to do it at the algorithm step.
    config.current_episode += 1
    return memory_cache, np.mean(trajectory_rewards), np.mean(trajectory_lengths)


