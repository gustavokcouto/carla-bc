import carla
import numpy as np
import pandas as pd
import tqdm #barra de progresso 
import json

from PIL import Image
from pathlib import Path

from carla_gym.envs import LeaderboardEnv
from carla_gym.core.task_actor.scenario_actor.agents.constant_speed_agent import ConstantSpeedAgent
from carla_gym.utils.expert_noiser import ExpertNoiser
from rl_birdview_wrapper import RlBirdviewWrapper


reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction',
        'kwargs': {}
    }
}

terminal_configs = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {}
    }
}

env_configs = {
    'carla_map': 'Town01', #cidade utilizada
    'weather_group': 'dynamic_1.0',
    'routes_group': 'train'
}

obs_configs = { #"sentidos" do carro: velocidade, controle, bdv, plano de rotas, GNSS e 3 câmeras
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'control': {
            'module': 'actor_state.control'
        },
        'velocity': {
            'module': 'actor_state.velocity'
        },
        'birdview': {
            'module': 'birdview.chauffeurnet',
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1],
            'scale_bbox': True,
            'scale_mask_col': 1.0
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        },
        'gnss': {
            'module': 'navigation.gnss'
        },    
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, -0.25, 1.3],
            'rotation': [0.0, 0.0, -45.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.25, 1.3],
            'rotation': [0.0, 0.0, 45.0]
        }
    }
}

if __name__ == '__main__':
    cfg = json.load(open("config.json", "r"))
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                         terminal_configs=terminal_configs, host="localhost", port=cfg['port'],
                         seed=2021, no_rendering=False, **env_configs)
    env = RlBirdviewWrapper(env)
    expert_file_dir = Path('gail_experts')
    expert_file_dir.mkdir(parents=True, exist_ok=True)
    # obs_metrics = ['control', 'vel_xy', 'linear_speed', 'vec', 'traj', 'cmd', 'command', 'state']
    for route_id in tqdm.tqdm(range(10)): #laço de repetição da execução das rotas
        env.set_task_idx(route_id)
        n_episodes = 1  # change to more if there is noisy actions, representa quantas vezes essa mesma rota será executada
        for ep_id in range(n_episodes):
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'birdview_masks').mkdir(parents=True)
            (episode_dir / 'central_rgb').mkdir(parents=True)
            (episode_dir / 'left_rgb').mkdir(parents=True)
            (episode_dir / 'right_rgb').mkdir(parents=True)

            longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0) #ruído frontal: não aplicado
            lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5) #ruído lateral: não aplicado

            obs = env.reset() #reposiciona o carro no início da rota e retorna a 1ª observação
            basic_agent = ConstantSpeedAgent(env.env._ev_handler.ego_vehicles['hero'], None, 6.0) #especialista PID que vai manter 6m/s e seguir a rota
            ep_dict = {} #Cada passo da simulação acrescenta um item
            ep_dict['done'] = []
            ep_dict['actions'] = []
            ep_dict['state'] = []
            ep_dict['pose'] = [] #lista nova
            ep_dict['command'] = [] #lista nova
            actions_ep = []
            i_step = 0
            c_route = False
            while not c_route:
                state_list = []
                ep_dict['done'].append(c_route)
                action = basic_agent.get_action() #ação dada pelo especialista
                ep_dict['actions'].append([action[0], action[1]])

                # Pose global do carro no instante atual: [x, y, yaw_em_radianos]. 
                # Como convenção usamos radianos igual o notebook 02
                ev_transform = env.env._ev_handler.ego_vehicles['hero'].vehicle.get_transform()
                ep_dict['pose'].append([
                    ev_transform.location.x,
                    ev_transform.location.y,
                    np.deg2rad(ev_transform.rotation.yaw)
                ])

                # Comando de alta ordem (RoadOption) do próximo ponto da rota:
                # -1 VOID, 1 LEFT, 2 RIGHT, 3 STRAIGHT, 4 LANEFOLLOW, 5/6 troca de faixa.
                route_plan = env.env._ev_handler.ego_vehicles['hero'].route_plan
                command = route_plan[0][1].value if len(route_plan) > 0 else 4
                ep_dict['command'].append([command])
                birdview = obs['birdview']
                for i_mask in range(1):
                    birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
                    birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))

                central_rgb = obs['central_rgb']
                central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(central_rgb).save(episode_dir / 'central_rgb' / '{:0>4d}.png'.format(i_step))

                left_rgb = obs['left_rgb']
                left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(left_rgb).save(episode_dir / 'left_rgb' / '{:0>4d}.png'.format(i_step))

                right_rgb = obs['right_rgb']
                right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(right_rgb).save(episode_dir / 'right_rgb' / '{:0>4d}.png'.format(i_step))

                ep_dict['state'].append(obs['state'])

                obs, reward, done, info = env.step(action) #executa a ação e avança a simulação
                c_route = info['route_completion']['is_route_completed']

                i_step += 1


            ep_df = pd.DataFrame(ep_dict)
            ep_df.to_json(episode_dir / 'episode.json')