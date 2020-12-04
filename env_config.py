# Discrete action space environments
class CartPolev0Config:
    reward_threshold = 190.
    reward_std_threshold = 19.
    action_space_type = 'discrete'
    plot_performance_low = 0
    plot_performance_high = 201


class CartPolev1Config:
    reward_threshold = 475.
    reward_std_threshold = 30
    action_space_type = 'discrete'
    plot_performance_low = 0
    plot_performance_high = 501


class LunarLanderv2Config:
    reward_threshold = 200.
    reward_std_threshold = 20
    action_space_type = 'discrete'
    plot_performance_low = -1000
    plot_performance_high = 250


class MountainCarDiscretev0Config:
    reward_threshold = -110.0
    reward_std_threshold = 100
    action_space_type = 'discrete'
    plot_performance_low = -5000
    plot_performance_high = 5000


class Acrobotv1Config:
    reward_threshold = -100.0
    reward_std_threshold = 100
    action_space_type = 'discrete'
    plot_performance_low = -500
    plot_performance_high = 0



class Pendulumv0Config:
    reward_threshold = -10.
    reward_std_threshold = 100
    action_space_type = 'continuous'
    plot_performance_low = -1500
    plot_performance_high = 0


class LunarLanderSupervisedv2Config:
    reward_threshold = 200.
    reward_std_threshold = 20
    action_space_type = 'continuous'
    plot_performance_low = -1000
    plot_performance_high = 250


class HalfCheetahConfig:
    reward_threshold = 4800.
    reward_std_threshold = 480
    action_space_type = 'continuous'
    plot_performance_low = -2000
    plot_performance_high = 10000


class HopperConfig:
    reward_threshold = 3800.
    reward_std_threshold = 380.
    action_space_type = 'continuous'


class Walker2DConfig:
    # Unclear - must verify - no ending threshold
    reward_threshold = 5000.
    reward_std_threshold = 500.
    action_space_type = 'continuous'


class AntConfig:
    reward_threshold = 6000.
    reward_std_threshold = 600.
    action_space_type = 'continuous'


class ReacherConfig:
    reward_threshold = 3.75
    reward_std_threshold = 0.3
    action_space_type = 'continuous'


class InvertedPendulumConfig:
    reward_threshold = 950.
    reward_std_threshold = 95.
    action_space_type = 'continuous'
