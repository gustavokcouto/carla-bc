# Import CARLA from the egg
import sys
import glob
import carla

import numpy as np
import pandas as pd
import tqdm
import os

from PIL import Image
from pathlib import Path

from carla_gym.envs import LeaderboardEnv
from carla_gym.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
from expert.lbc_roaming_agent import LbcRoamingAgent

from configs.data_collection_config import reward_configs, terminal_configs, env_configs, obs_configs
from carla_gym.utils.expert_noiser import ExpertNoiser


def create_folders(base: str, route_id: int, ep_id: int) -> str:
    episode_dir = base / 'route_{}'.format(route_id) / ('ep_{}'.format(ep_id))
    for name in ['birdview_rendered', 'birdview_masks', 'central_rgb', 'left_rgb','right_rgb' ]:
        path = episode_dir / name
        if not os.path.exists(path):
            (episode_dir / name).mkdir(parents=True)
            
    return episode_dir
  

def store_camera_img(obs, camera_name: str, episode_dir: str, ep_step: int):
    camera = obs[camera_name]['data']
    camera = camera.astype(np.uint8)
    Image.fromarray(camera).save(episode_dir / '{}'.format(camera_name) / '{}.png'.format(ep_step))


def collect_data(dir_name, with_breaking=False, noise_lon=False, noise_lat=False):
    ''' Creates expert dataset. Stores all 15 channels!
    '''
    noise_kwargs = {
        'noise_lon': noise_lon, 
        'lan_intensity': 10,
        'noise_lat': noise_lat,
        'lat_intensity': 4
    }
    # collect data from predefined routes of leaderboard
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                         terminal_configs=terminal_configs, host="localhost", port=2002,
                         seed=2021, no_rendering=False, **env_configs)
    # env = RlBirdviewWrapper(env, with_breaking=with_breaking, **noise_kwargs)
    longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=noise_kwargs['lan_intensity'], min_noise_time_amount=2.0)
    lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=noise_kwargs['lat_intensity'], min_noise_time_amount=0.5)
    expert_file_dir = Path(dir_name)
    expert_file_dir.mkdir(parents=True, exist_ok=True)
    
    for route_id in tqdm.tqdm(range(8, 10)):
        env.set_task_idx(route_id)
        ibc_roaming_agent = LbcRoamingAgent()
        for ep_id in range(2):
            episode_dir = create_folders(expert_file_dir, route_id, ep_id)

            ibc_roaming_agent.reset()
            obs = env.reset()
            timestamp = env.timestamp
            
            ep_dict = {}
            ep_dict['done'] = []
            ep_dict['actions'] = []
            ep_dict['cmd'] = []
            ep_dict['state'] = []
            ep_dict['vehicle_hazard'] = []
            ep_dict['pedestrian_ahead'] = []
            ep_dict['redlight_ahead'] = []

            i_step = 0
            route_completed = False
            episode_done = False

            while not (route_completed or episode_done):

                ep_dict['done'].append(route_completed)
                obs = obs['hero']
                action = ibc_roaming_agent.run_step(obs, timestamp)
                if action.throttle > 0:
                    acc = action.throttle
                else:
                    acc = -action.brake
                steer = action.steer
                ep_dict['actions'].append([acc, steer]) # acc and steering only brake is infered from acc
                birdview = obs['birdview']['masks']
                
                # if I would want to store all 15 channels
                #for i_mask in range(5):
                for i_mask in range(4): #its storing 5 images now, if only 1 then range(1) # only for images if history only 3
                    birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
                    birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{}_{}.png'.format(i_step, i_mask))
                
                birdview = obs['birdview']['rendered']
                Image.fromarray(birdview).save(episode_dir / 'birdview_rendered' / '{}.png'.format(i_step))

                
                camaras = ['central_rgb', 'left_rgb', 'right_rgb']
                for camera_name in camaras:
                    store_camera_img(obs, camera_name, episode_dir, i_step)


                hazard_vehicle_loc = lbc_hazard_vehicle(obs['surrounding_vehicles'], obs['speed']['speed_xy'][0])
                vehicle_hazard = hazard_vehicle_loc is not None
                ep_dict['vehicle_hazard'].append(vehicle_hazard)
                hazard_ped_loc = lbc_hazard_walker(obs['surrounding_pedestrians'])
                pedestrian_ahead = hazard_ped_loc is not None
                ep_dict['pedestrian_ahead'].append(pedestrian_ahead)
                redlight_ahead = obs['traffic_light']['at_red_light'] == 1
                ep_dict['redlight_ahead'].append(redlight_ahead)

                state_list = []
                state_list.append(obs['control']['throttle'])
                state_list.append(obs['control']['steer'])
                state_list.append(obs['control']['brake'])
                state_list.append(obs['control']['gear']/5.0)
                state_list.append(obs['velocity']['vel_xy'])
                state = np.concatenate(state_list)
                ep_dict['state'].append(state)
                # VOID = -1
                # LEFT = 1
                # RIGHT = 2
                # STRAIGHT = 3
                # LANEFOLLOW = 4
                # CHANGELANELEFT = 5
                # CHANGELANERIGHT = 6
                command = obs['gnss']['command'][0]
                if command < 0:
                    command = 4
                command -= 1
                cmd_one_hot = [0] * 6
                cmd_one_hot[command] = 1
                cmd_array = np.array(cmd_one_hot)
                ep_dict['cmd'].append(cmd_array)

                if noise_kwargs['noise_lon']:
                    action, _, _ = longitudinal_noiser.compute_noise(action, obs['speed']['forward_speed'][0] * 3.6)
                if noise_kwargs['noise_lat']:
                    action, _, _ = lateral_noiser.compute_noise(action, obs['speed']['forward_speed'][0] * 3.6)
                action = {'hero': action}
                obs, reward, done, info = env.step(action)
                route_completed = info['hero']['route_completion']['is_route_completed']
                episode_done = done['hero']

                i_step += 1
                timestamp = env.timestamp
                
            ep_df = pd.DataFrame(ep_dict)
            ep_df.to_json(episode_dir / 'episode.json')
                
if __name__ == '__main__':
    # with breaking sets the actions space of acc to -1,1
    # instead of just 0,1. If we want our agent to break this
    # needs to be included
    dir_name = 'experts_traffic_light_noisy'
    collect_data(dir_name=dir_name, with_breaking=True, noise_lon=True, noise_lat=True)
