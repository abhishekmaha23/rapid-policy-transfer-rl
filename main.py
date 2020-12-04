from configurations import CartpoleSupervisedConfig, LunarLanderSupervisedConfig, PendulumSupervisedConfig, \
    CartpoleGeneticConfig, LunarLanderGeneticConfig
from configurations import MountainCarSupervisedConfig, AcrobotSupervisedConfig, HalfCheetahSupervisedConfig, \
    HopperSupervisedConfig, LunarLanderContinuousSupervisedConfig
from configurations import HalfCheetahGeneticConfig, LunarLanderContinuousGeneticConfig, PendulumGeneticConfig, \
    AcrobotGeneticConfig
from aux.common import SupervisedLogs, GALogs
import supervised_meta
# import supervised_agent
import ga
import time


if __name__ == '__main__':
    run_id = time.time()
    params = dict()

    # params['method'] = 'rl' # Not currently included, since algorithms must include a mode to do a normal grad update
    # params['method'] = 'supervised'
    params['method'] = 'ga'

    # Algorithms are A2C, PPO, DDPG, TD3
    # Memory_types are onpolicy and offpolicy. For now these are connected.
    # Families are Stochastic, Stochastic2, Deterministic (Relating to Continuous action spaces only).
    # Notes on family
    # -----
    # Stochastic - Standard Deviation is kept constant, based on values found in working examples.
    # Since generator creates the actions directly, it's hard to use the standard deviation as an output of the policy.
    # Deterministic - Output of policy is the direct action.

    # For both discrete and continuous envs
    # params['algorithm'], params['memory_type'], params['family'] = 'A2C', 'onpolicy', 'stochastic'
    # params['algorithm'], params['memory_type'], params['family'] = 'PPO', 'onpolicy', 'stochastic'

    # For only continuous
    # params['algorithm'], params['memory_type'], params['family'] = 'DDPG', 'offpolicy', 'deterministic'
    # params['algorithm'], params['memory_type'], params['family'] = 'TD3', 'offpolicy', 'deterministic'

    # Genetic algorithms
    params['algorithm'], params['memory_type'], params['family'] = None, None, None
    # params['algorithm'], params['memory_type'], params['family'] = None, None, 'deterministic'

    # Discrete
    # config = CartpoleSupervisedConfig(run_id, env_name='CartPole-v0', params=params)
    # config = MountainCarSupervisedConfig(run_id, env_name='MountainCar-v0', params=params)
    # config = AcrobotSupervisedConfig(run_id, env_name='Acrobot-v1', params=params)
    # config = LunarLanderSupervisedConfig(run_id, env_name='LunarLander-v2', params=params)

    # Continuous
    # config = PendulumSupervisedConfig(run_id, env_name='Pendulum-v0', params=params)
    # config = HalfCheetahSupervisedConfig(run_id, env_name='HalfCheetah-v2', params=params)
    # config = HopperSupervisedConfig(run_id, env_name='Hopper-v3', params=params)
    # config = LunarLanderContinuousSupervisedConfig(run_id, env_name='LunarLanderContinuous-v2', params=params)

    # run_id = '1605488373.2129288'

    # Genetic
    # config = CartpoleGeneticConfig(run_id, env_name='CartPole-v1', params=params)
    # config = LunarLanderGeneticConfig(run_id, env_name='LunarLander-v2', params=params)
    # config = HalfCheetahGeneticConfig(run_id, env_name='HalfCheetah-v2', params=params)
    # config = LunarLanderContinuousGeneticConfig(run_id, env_name='LunarLanderContinuous-v2', params=params)
    # config = PendulumGeneticConfig(run_id, env_name='Pendulum-v0', params=params)
    config = AcrobotGeneticConfig(run_id, env_name='Acrobot-v1', params=params)
    config.mode = 'start'
    # config.mode = 'resume'
    # config.last_gen = 102


    if params['method'] == 'supervised':
        logs = SupervisedLogs(config)
        supervised_meta.train(config, logs)
    elif params['method'] == 'ga':
        logs = GALogs(config)
        ga.train(config, logs)
    # elif params['method'] == 'rl':
    #     logs = RLLogs(config)
    #     supervised_agent.train(config, logs)
