import torch
import torch.distributions as distributions
import torch.nn.functional as F
from aux.models import SequentialActor
import higher

''' A2C and PPO code adapted from https://github.com/bentrevett/pytorch-rl'''
''' TD3, DDPG code adapted from original author of TD3, https://github.com/sfujim/TD3'''


def get_shaped_memory_sample(config, memory_cache):
    states, next_states, actions, rewards, dones, log_prob_actions = memory_cache.sample()
    # states should be a tensor of shape (sample_size, config.state_dim)
    # next_states should be a tensor of shape (sample_size, config.state_dim)
    # actions should be (sample_size, config.action_dim)
    # rewards should be (sample_size, 1)
    # log_prob_actions_init should be (sample_size, 1)
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    # If actions are discrete, then they're single values.
    # If continuous, then they're a tensor, of at least one value.
    # Either way, not squeezing.
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards).squeeze(-1)
    dones = torch.FloatTensor(dones).squeeze(-1)
    if config.family == 'deterministic':
        log_prob_actions = None
    elif config.family == 'stochastic':
        log_prob_actions = torch.cat(log_prob_actions).view(-1, 1)
    return states, next_states, actions, rewards, dones, log_prob_actions


def td3_update(config, fnet_actor, actor_rl_opt, models, optimizers, memory_cache, update_type='meta'):
    summed_policy_loss = torch.zeros(1)
    summed_value_loss = torch.zeros(1)
    optimizers['critic_opt'].zero_grad()
    diff_crit_opt = higher.get_diff_optim(optimizers['critic_opt'], models['critic'].parameters(), track_higher_grads=False)
    diff_crit_opt_2 = higher.get_diff_optim(optimizers['critic_opt_2'], models['critic_2'].parameters(),
                                          track_higher_grads=False)
    diff_act_opt_internal = higher.create_diff_optim(torch.optim.SGD, fmodel=fnet_actor, track_higher_grads=True, opt_kwargs={'lr': config.actor_rl_learning_rate, 'momentum': 0.01})

    # Initially attempted alternate structure with functional critics, abandoned as found to be unnecessary.
    # with higher.innerloop_ctx(models['critic'], optimizers['critic_opt'], copy_initial_weights=False) as (fnet_critic, diff_crit_opt):
    #     with higher.innerloop_ctx(models['critic_2'], optimizers['critic_opt_2'], copy_initial_weights=False) as (fnet_critic_2, diff_crit_opt_2):

    for it in range(config.offpol_num_iterations_update):
        states, next_states, actions_init, rewards, dones, _ = get_shaped_memory_sample(config, memory_cache)
        inverted_dones = 1 - dones

        rewards = rewards.view(-1, 1)
        inverted_dones = inverted_dones.view(-1, 1)

        noise = torch.FloatTensor(actions_init).data.normal_(0, 0.2)
        noise = noise.clamp(-0.5, 0.5)
        next_action = (models['actor_target'](next_states) + noise).clamp(-config.action_space_high[0], config.action_space_high[0])

        target_Q1 = models['critic_target'](next_states, next_action)
        target_Q2 = models['critic_target_2'](next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (inverted_dones * config.discount_factor * target_Q).detach()
        current_Q1 = models['critic'](states, actions_init)
        current_Q2 = models['critic_2'](states, actions_init)

        torch.autograd.set_detect_anomaly(True)

        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)
        diff_crit_opt.step(critic_loss_1,  models['critic'].parameters())
        diff_crit_opt_2.step(critic_loss_2,  models['critic_2'].parameters())

        summed_value_loss += critic_loss_1 + critic_loss_2

        if it % 2 == 0:
            actor_loss = - models['critic'](states, fnet_actor(states)).mean()

            # Optimize the actor # Functionally
            diff_act_opt_internal.step(actor_loss)
            summed_policy_loss += actor_loss

            # Update the frozen target models
            # Critic is as before.
            for param, target_param in zip( models['critic'].parameters(), models['critic_target'].parameters()):
                target_param.data.copy_(config.offpol_target_update_rate * param.data + (
                            1 - config.offpol_target_update_rate) * target_param.data)
            for param, target_param in zip( models['critic_2'].parameters(), models['critic_target_2'].parameters()):
                target_param.data.copy_(config.offpol_target_update_rate * param.data + (
                            1 - config.offpol_target_update_rate) * target_param.data)

            if not (fnet_actor.dim == models['actor_target'].dim):
                # print('Changing target actor to resemble fnet_actor')
                models['actor_target'] = SequentialActor(config, fnet_actor.state_scaler,
                                                         fnet_actor.state_normalizer,
                                                         dim=fnet_actor.dim,
                                                         num_layers=fnet_actor.num_layers).reset_weights()
            # For actor, performing a bit of a unwieldier approach.
            for param, target_param in zip(fnet_actor.parameters(), models['actor_target'].parameters()):
                target_param.data.copy_(config.offpol_target_update_rate * param.data +
                                        (1 - config.offpol_target_update_rate) * target_param.data)

    return summed_policy_loss, summed_value_loss.item()


def ddpg_update(config, fnet_actor, diff_act_opt, models, optimizers, memory_cache, update_type='meta'):
    summed_policy_loss = torch.zeros(1)
    summed_value_loss = torch.zeros(1)
    diff_crit_opt = higher.get_diff_optim(optimizers['critic_opt'], models['critic'].parameters(),
                                          track_higher_grads=False)
    diff_act_opt_internal = higher.create_diff_optim(torch.optim.SGD, fmodel=fnet_actor, track_higher_grads=True,
                                                     opt_kwargs={'lr': config.actor_rl_learning_rate, 'momentum': 0.01})
    for it in range(config.offpol_num_iterations_update):
        states, next_states, actions_init, rewards, dones, _ = get_shaped_memory_sample(config, memory_cache)
        inverted_dones = 1 - dones
        rewards = rewards.view(-1, 1)
        inverted_dones = inverted_dones.view(-1, 1)
        target_Q = models['critic_target'](next_states, models['actor_target'](next_states))
        target_Q = rewards + (inverted_dones * config.discount_factor * target_Q).detach()

        current_Q = models['critic'](states, actions_init)

        torch.autograd.set_detect_anomaly(True)
        critic_loss = F.mse_loss(current_Q, target_Q)

        diff_crit_opt.step(critic_loss, models['critic'].parameters())
        summed_value_loss += critic_loss

        actor_loss = -models['critic'](states, fnet_actor(states)).mean()
        # Optimize the actor # Functionally

        diff_act_opt.step(actor_loss)
        summed_policy_loss += actor_loss

        # Update the frozen target models
        # Critic is as before.
        for param, target_param in zip(models['critic'].parameters(), models['critic_target'].parameters()):
            target_param.data.copy_(config.offpol_target_update_rate * param.data + (1 - config.offpol_target_update_rate) * target_param.data)

        if not (fnet_actor.dim == models['actor_target'].dim):
            # print('Changing target actor to resemble fnet_actor')
            models['actor_target'] = SequentialActor(config, fnet_actor.state_scaler, fnet_actor.state_normalizer,
                                                     dim=fnet_actor.dim,
                                                     num_layers=fnet_actor.num_layers).reset_weights()
        # For actor, performing a bit of a unwieldier approach.
        for param, target_param in zip(fnet_actor.parameters(), models['actor_target'].parameters()):
            target_param.data.copy_(config.offpol_target_update_rate * param.data +
                                    (1 - config.offpol_target_update_rate) * target_param.data)

    return summed_policy_loss, summed_value_loss.item()


# Update policies and critics for all involved algorithms
def ppo_update(config, f_actor, diff_actor_opt, critic, critic_opt, memory_cache, update_type='meta'):
    # Actor is functional in meta, and normal in rl.
    summed_policy_loss = torch.zeros(1)
    summed_value_loss = torch.zeros(1)

    states, next_states, actions_init, rewards, dones, log_prob_actions_init = get_shaped_memory_sample(config, memory_cache)
    # Using critic to predict last reward. Just as a placeholder in case the trajectory is incomplete in the batch-mode.
    final_predicted_reward = 0.
    if dones[-1] == 0.:  # Then last step is not done. Last value has to be predicted.
        final_state = next_states[-1]
        with torch.no_grad():
            final_predicted_reward = critic(final_state).detach().item()
    returns = calculate_returns(config, rewards, dones, predicted_end_reward=final_predicted_reward) #Returns(samples,1)
    # At this point, they should always be tensors and output a tensor based solution.
    values_init = critic(states)
    advantages = returns - values_init
    if config.normalize_rewards_and_advantages:
        advantages = (advantages - advantages.mean()) / advantages.std()
    advantages = advantages.detach()  # Necessary to keep the advantages from have a connection to the value model.
    # Now the actor makes steps and recalculates actions and log_probs based on the current values for k epochs.

    for ppo_step in range(config.num_ppo_steps):
        action_prob = f_actor(states)
        # print('action_prob', type(action_prob), action_prob.shape, action_prob)
        values_pred = critic(states)
        if config.env_config.action_space_type == 'discrete':
            dist = distributions.Categorical(action_prob) ## Stupido
            actions_init = actions_init.squeeze(-1)
            new_log_prob_actions = dist.log_prob(actions_init)
            new_log_prob_actions = new_log_prob_actions.view(-1, 1)
        elif config.env_config.action_space_type == 'continuous':
            action_mean_vector = action_prob * f_actor.action_upper_limit  # Direct code from actor get_action, refer there
            dist = distributions.MultivariateNormal(action_mean_vector, f_actor.covariance_matrix)
            actions_init = actions_init.view(-1, config.action_dim)
            new_log_prob_actions = dist.log_prob(actions_init)
            new_log_prob_actions = new_log_prob_actions.view(-1, 1)

        policy_ratio = (new_log_prob_actions - log_prob_actions_init).exp()
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - config.ppo_clip, max=1.0 + config.ppo_clip) * advantages
        if config.include_entropy_in_ppo:
            inner_policy_loss = (
                        -torch.min(policy_loss_1, policy_loss_2) - config.entropy_coefficient * dist.entropy()).sum()
        else:
            inner_policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
        if update_type == 'meta':
            diff_actor_opt.step(inner_policy_loss)
        else:
            # In this case, it's normal RL, and so there is no updating that happens outside in the main function.
            diff_actor_opt.zero_grad()
            inner_policy_loss.backward()
            diff_actor_opt.step()
        inner_value_loss = F.smooth_l1_loss(values_pred, returns).sum()
        inner_value_loss.backward()
        critic_opt.step()
        summed_policy_loss += inner_policy_loss
        summed_value_loss += inner_value_loss
    return summed_policy_loss, summed_value_loss.item()


def a2c_update(config, critic, critic_opt, memory_cache, update_type='meta'):
    # Preprocessing using memory in on-policy algorithm to calculate advantages and stuff.
    # This is needed to separate on-policy and off-policy logic.
    # The next_states array isn't required in an on-policy updating algorithm. Only the last state is useful.
    # states, next_states, actions, rewards, dones, log_prob_actions = memory_cache.sample()
    states, next_states, actions, rewards, dones, log_prob_actions = get_shaped_memory_sample(config, memory_cache)
    # Using critic to predict last reward. Just as a placeholder in case the trajectory is incomplete in the batch-mode.
    final_predicted_reward = 0.
    if dones[-1] == 0.:  # Then last step is not done. Last value has to be predicted.
        final_state = next_states[-1]
        with torch.no_grad():
            final_predicted_reward = critic(final_state).detach().item()
    returns = calculate_returns(config, rewards, dones,
                                predicted_end_reward=final_predicted_reward)  # Returns(samples,1)

    values = critic(states)
    advantages = returns - values

    if config.normalize_rewards_and_advantages:
        advantages = (advantages - advantages.mean()) / advantages.std()
    advantages = advantages.detach()  # Necessary to keep the advantages from have a connection to the value model.

    # Loss calculation based on equations
    policy_loss = - (advantages * log_prob_actions).sum()
    value_loss = F.smooth_l1_loss(values, returns).sum()

    value_loss.backward()
    critic_opt.step()

    return policy_loss, value_loss.item()


def calculate_returns(config, rewards, dones, predicted_end_reward=0.):
    returns = []
    discounted_reward = predicted_end_reward  # The critic calls this the potential final reward of the last trajectory
    for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (config.discount_factor * discounted_reward)
        returns.insert(0, discounted_reward)
    returns = torch.Tensor(returns)
    # if config.normalize_rewards_and_advantages:
    #     returns = (returns - returns.mean()) / returns.std()
    returns = returns.view(-1, 1)
    return returns