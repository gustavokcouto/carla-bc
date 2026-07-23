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

# in evaluation we set eval_mode to True because then 
# timeout results in env done! which is good for evaluation
terminal_configs_eval = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {
            'eval_mode' : True,
        }
    }
}

# for collecting data from leaderboard env
env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': 'train'
}

# for training we use endlessEnv and thus need to change configs
env_configs_eval = {
    # town needs to be specified
    'num_zombie_vehicles': 70,
    'num_zombie_walkers': 70,
    'weather_group': 'dynamic_1.0'
}

obs_configs = {
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
            'history_idx': [-16, -9, -1], #[-16, -11, -6, -1], #configs for ObsManager, receives lights information from -1,-6,-11,-16 steps ago
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
        },
        'surrounding_vehicles': {
            'module': 'object_finder.vehicle',
            'distance_threshold': 20,
            'max_detection_number': 20
        },
        'surrounding_pedestrians':{
            'module': 'object_finder.pedestrian',
            'distance_threshold': 15,
            'max_detection_number': 20
        },
        'stop_sign': {
            'module': 'object_finder.stop_sign',
            'distance_threshold': 10
        },
        'traffic_light': {
            'module': 'object_finder.traffic_light_new'
        },
        'ego_vehicle': {
            'module': 'object_finder.ego'
        }
    }
}