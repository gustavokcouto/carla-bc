import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym
import json

from expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from configs.data_collection_config import reward_configs, terminal_configs_eval, obs_configs
from eval_agent import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


def eval_bc(policy, device, env):
    ckpt_dir = Path('ckpt')

    ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
    saved_variables = th.load(ckpt_path, map_location='cuda')

    policy.load_state_dict(saved_variables['policy_state_dict'])

    video_path = Path('video')
    video_path.mkdir(parents=True, exist_ok=True)

    eval_video_path = (video_path / f'bc_eval.mp4').as_posix()
    avg_ep_stat, avg_route_completion, ep_events = evaluate_policy(env, policy, eval_video_path)

    print('--- avg_ep_stat ---')
    for k, v in avg_ep_stat.items():
        print(f'{k}: {v}')
    print('--- avg_route_completion ---')
    for k, v in avg_route_completion.items():
        print(f'{k}: {v}')

    env.reset()


def env_maker():
    cfg = json.load(open("config.json", "r"))
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs_eval, host='localhost', port=cfg['port'],
                    seed=2021, no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env)
    return env

if __name__ == '__main__':
    env = SubprocVecEnv([env_maker])

    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(12, 192, 192), dtype=np.uint8)
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Dict(**observation_space)

    action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

    # network
    policy_kwargs = {
        'observation_space': observation_space,
        'action_space': action_space,
        'policy_head_arch': [256, 256],
        'features_extractor_entry_point': 'torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'distributions:BetaDistribution',
    }

    device = 'cuda'

    policy = AgentPolicy(**policy_kwargs)
    policy.to(device)

    batch_size = 24

    eval_bc(policy, device, env)