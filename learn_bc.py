import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb
import gym

from expert_dataset import ExpertDataset
from agent_policy import AgentPolicy
from carla_gym.envs import EndlessEnv
from rl_birdview_wrapper import RlBirdviewWrapper
from data_collect import reward_configs, terminal_configs, obs_configs
from eval_agent import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}


def learn_bc(policy, device, expert_loader, eval_loader, env, resume_last_train):
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_last_train:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        saved_variables = th.load(ckpt_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']
        start_ep = train_kwargs['start_ep']
        i_steps = train_kwargs['i_steps']

        policy.load_state_dict(saved_variables['policy_state_dict'])
        wandb.init(project='gail-carla2', id=wandb_run_id, resume='must')
    else:
        run = wandb.init(project='gail-carla2', reinit=True)
        with open(last_checkpoint_path, 'w') as log_file:
            log_file.write(wandb.run.path)
        start_ep = 0
        i_steps = 0

    video_path = Path('video')
    video_path.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    episodes = 200
    ent_weight = 0.01
    min_eval_loss = np.inf
    eval_step = int(1e5)
    steps_last_eval = 0

    for i_episode in tqdm.tqdm(range(start_ep, episodes)):
        total_loss = 0
        i_batch = 0
        policy = policy.train()
        # Expert dataset
        for expert_batch in expert_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            expert_action = expert_action.to(device)

            # Get BC loss
            alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
            bcloss = -alogprobs.mean()

            loss = bcloss + ent_weight * entropy_loss
            total_loss += loss
            i_batch += 1
            i_steps += expert_obs_dict['state'].shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        for expert_batch in eval_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            expert_action = expert_action.to(device)

            # Get BC loss
            with th.no_grad():
                alogprobs, entropy_loss = policy.evaluate_actions(obs_tensor_dict, expert_action)
            bcloss = -alogprobs.mean()

            eval_loss = bcloss + ent_weight * entropy_loss
            total_eval_loss += eval_loss
            i_eval_batch += 1
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        wandb.log({
            'loss': loss,
            'eval_loss': eval_loss,
        }, step=i_steps)

        if i_steps - steps_last_eval > eval_step:
            eval_video_path = (video_path / f'bc_eval_{i_steps}.mp4').as_posix()
            avg_ep_stat, avg_route_completion, ep_events = evaluate_policy(env, policy, eval_video_path)
            env.reset()
            wandb.log(avg_ep_stat, step=i_steps)
            wandb.log(avg_route_completion, step=i_steps)
            steps_last_eval = i_steps

        if min_eval_loss > eval_loss:
            ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
            th.save(
                {'policy_state_dict': policy.state_dict()},
               ckpt_path
            )
            min_eval_loss = eval_loss

        train_init_kwargs = {
            'start_ep': i_episode,
            'i_steps': i_steps
        } 
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        th.save({'policy_state_dict': policy.state_dict(),
                 'train_init_kwargs': train_init_kwargs},
                ckpt_path)
        wandb.save(f'./{ckpt_path}')
    run = run.finish()


def env_maker():
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2000,
                    seed=2021, no_rendering=True, **env_configs)
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

    batch_size = 24

    gail_train_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=8,
            n_eps=1,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    
    gail_val_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=2,
            n_eps=1,
            route_start=8
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    learn_bc(policy, device, gail_train_loader, gail_val_loader, env, resume_last_train)