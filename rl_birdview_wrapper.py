# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/utils/rl_birdview_wrapper.py

import gym
import numpy as np
import cv2
import carla
import pandas as pd
from pathlib import Path
from PIL import Image

import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
from carla_gym.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
from carla_gym.utils.expert_noiser import ExpertNoiser


eval_num_zombie_vehicles = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 150,
    'Town05': 120,
    'Town06': 120
}
eval_num_zombie_walkers = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 80,
    'Town05': 120,
    'Town06': 80
}

class RlBirdviewWrapper(gym.Wrapper):
    def __init__(self, env, env_id=0, with_breaking=False, 
                 noise_lon=False, noise_lat=False, lan_intensity=10, lat_intensity=4):
        self._ev_id = list(env._obs_configs.keys())[0]
        self._render_dict = {}
        self._env_id = env_id
        
        self.longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=lan_intensity, min_noise_time_amount=2.0)
        self.lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=lat_intensity, min_noise_time_amount=0.5)
        self.noise_lon=noise_lon
        self.noise_lat=noise_lat
        
        self.previous_speed = 0


        observation_space = {}
        observation_space['birdview'] = env.observation_space[self._ev_id]['birdview']['masks']
        observation_space['birdview_rendered'] = env.observation_space[self._ev_id]['birdview']['rendered']
        observation_space['central_rgb'] = env.observation_space[self._ev_id]['central_rgb']['data']
        observation_space['left_rgb'] = env.observation_space[self._ev_id]['left_rgb']['data']
        observation_space['right_rgb'] = env.observation_space[self._ev_id]['right_rgb']['data']
        observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
        observation_space['cmd'] = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        observation_space['redlight_ahead'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        observation_space['pedestrian_ahead'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        observation_space['vehicle_hazard'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        observation_space['actor_location'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(3,), dtype=np.float32)

        env.observation_space = gym.spaces.Dict(**observation_space)
        low_acc = 0 if not with_breaking else -1
        env.action_space = gym.spaces.Box(low=np.array([low_acc, -1]), high=np.array([1, 1]), dtype=np.float32)

        super(RlBirdviewWrapper, self).__init__(env)

    def reset(self):
        obs_ma = self.env.reset()
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=True, gear=1)}
        obs_ma, _, _, _ = self.env.step(action_ma)
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=False)}
        obs_ma, _, _, _ = self.env.step(action_ma)

        snap_shot = self.env._world.get_snapshot()
        self.env._timestamp = {
            'step': 0,
            'frame': 0,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        obs = self.process_obs(obs_ma[self._ev_id])

        self._render_dict['prev_obs'] = obs
        self._render_dict['prev_im_render'] = obs_ma[self._ev_id]['birdview']['rendered']
        return obs

    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action)}

        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)

        obs = self.process_obs(obs_ma[self._ev_id])
                
        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]

        self._render_dict = {
            'timestamp': self.env.timestamp,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }
        
        return obs, reward, done, info

    def render(self, mode='human'):
        '''
        train render: used in train_rl.py
        '''
        #self._render_dict['action_logits'] = self.action_logits
        #self._render_dict['action_mu'] = self.action_mu
        #self._render_dict['action_sigma'] = self.action_sigma
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        #mu_str = np.array2string(render_dict['action_mu'], precision=2, separator=',', suppress_small=True)
        #sigma_str = np.array2string(render_dict['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        #txt_3 = f'a{mu_str} b{sigma_str}'
        #im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im


    def process_obs(self, obs):
        obs_dict = {}

        hazard_vehicle_loc = lbc_hazard_vehicle(obs['surrounding_vehicles'], obs['speed']['speed_xy'][0])
        vehicle_hazard = hazard_vehicle_loc is not None
        obs_dict['vehicle_hazard'] = vehicle_hazard
        hazard_ped_loc = lbc_hazard_walker(obs['surrounding_pedestrians'])
        pedestrian_ahead = hazard_ped_loc is not None
        obs_dict['pedestrian_ahead'] = pedestrian_ahead
        redlight_ahead = obs['traffic_light']['at_red_light'] == 1
        obs_dict['redlight_ahead'] = redlight_ahead
        obs_dict['actor_location'] = obs['ego_vehicle']['actor_location']

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
        obs_dict['cmd'] = cmd_array

        state_list = []
        state_list.append(obs['control']['throttle'])
        state_list.append(obs['control']['steer'])
        state_list.append(obs['control']['brake'])
        state_list.append(obs['control']['gear']/5.0)
        state_list.append(obs['velocity']['vel_xy'])
        obs_dict['state'] = np.concatenate(state_list)
        
        self.previous_speed = obs['speed']['forward_speed'][0]

        birdview = obs['birdview']['masks']
        birdview_rendered = obs['birdview']['rendered']
        obs_dict.update({
            'birdview': birdview,
            'birdview_rendered': birdview_rendered
        })

        central_rgb = obs['central_rgb']['data']
        central_rgb = np.transpose(central_rgb, [2, 0, 1])

        left_rgb = obs['left_rgb']['data']
        left_rgb = np.transpose(left_rgb, [2, 0, 1])

        right_rgb = obs['right_rgb']['data']
        right_rgb = np.transpose(right_rgb, [2, 0, 1])

        obs_dict.update({
            'central_rgb': central_rgb,
            'left_rgb': left_rgb,
            'right_rgb': right_rgb
        })

        return obs_dict


    def process_act(self, action):
        acc, steer = action.astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        
        if self.noise_lon:
            control, _, _ = self.longitudinal_noiser.compute_noise(control, self.previous_speed * 3.6)
        if self.noise_lat:
            control, _, _ = self.lateral_noiser.compute_noise(control, self.previous_speed * 3.6)
        
        return control