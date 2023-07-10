import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from agent_policy import AgentPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
import torch as th
from pathlib import Path
import cv2

def evaluate_policy(env, policy):
    policy = policy.eval()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()

    ep_events = {}
    for i in range(env.num_envs):
        ep_events[f'venv_{i}'] = []

    n_step = 0
    while True:
        actions, log_probs, mu, sigma, _, net_state1, net_state2 = policy.forward(obs, deterministic=True, clip_action=True)
        obs, reward, done, info = env.step(actions)

        n_step += 1
        birdview = obs['birdview'][0]
        for i_mask in range(1):
            birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
            birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
           
        central_rgb = obs['central_rgb'][0]
        central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)

        left_rgb = obs['left_rgb'][0]
        left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)

        right_rgb = obs['right_rgb'][0]
        right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
        
        left_rgb = cv2.resize(left_rgb,(512, 288))
        right_rgb = cv2.resize(right_rgb,(512, 288))
        central_rgb = cv2.resize(central_rgb,(512, 288))
        birdview_mask = cv2.resize(birdview_mask,(384,384))

        net_state1_channels = []
        for i in range(16):
            image = net_state1[0, i:i+1, :, :].cpu().numpy() *255
            image = np.transpose(image, [1, 2, 0]).astype(np.uint8)
            net_state1_channels.append(image)

        image_width = net_state1_channels[i].shape[0]
        image_height = net_state1_channels[i].shape[1]
        image_render = np.zeros([4*image_width, 4*image_height, 3],dtype=np.uint8)

        for i in range(16):
            i_column = int(i / 4)
            i_row = int(i % 4)
            image_render[i_column*image_width:(i_column+1)*image_width,i_row*image_height:(i_row+1)*image_height, :] = net_state1_channels[i]
        image_render1 = cv2.resize(image_render,(480, 480))

        net_state2_channels = []
        for i in range(64):
            image = net_state2[0, i:i+1, :, :].cpu().numpy() *255
            image = np.transpose(image, [1, 2, 0]).astype(np.uint8)
            net_state2_channels.append(image)

        image_width = net_state2_channels[i].shape[0]
        image_height = net_state2_channels[i].shape[1]
        image_render = np.zeros([8*image_width, 8*image_height, 3],dtype=np.uint8)

        for i in range(64):
            i_column = int(i / 8)
            i_row = int(i % 8)
            image_render[i_column*image_width:(i_column+1)*image_width,i_row*image_height:(i_row+1)*image_height, :] = net_state2_channels[i]
        image_render2 = cv2.resize(image_render,(480, 480))
        
        cv2.imshow('rgb', cv2.cvtColor(np.hstack((left_rgb, central_rgb, right_rgb)), cv2.COLOR_BGR2RGB))
        cv2.imshow('network_state1', cv2.cvtColor(image_render1, cv2.COLOR_BGR2RGB))
        cv2.imshow('network_state2', cv2.cvtColor(image_render2, cv2.COLOR_BGR2RGB))
        cv2.imshow('birdview', cv2.cvtColor(birdview_mask, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)



env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

def env_maker():
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2000,
                    seed=2021, no_rendering=False, **env_configs)
    env = RlBirdviewWrapper(env)
    return env

if __name__ == '__main__':
    env = SubprocVecEnv([env_maker])

    resume_last_train = False

    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Dict(**observation_space)

    action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)

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

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
    saved_variables = th.load(ckpt_path, map_location='cuda')

    policy.load_state_dict(saved_variables['policy_state_dict'])

    evaluate_policy(env, policy)
