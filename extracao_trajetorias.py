"""
Adaptação de 02_dataset_extracao_trajetorias.ipynb ao projeto carla-bc real.

Principais diferenças em relação à proposta de resolução original:

1. `poses` e `commands` não existiam no dataset coletado pelo projeto original.
   Foi preciso editar `data_collect.py` para gravar, a cada passo, a pose global
   do carro ([x, y, yaw_rad]) e o comando de alta ordem da rota (RoadOption).
   É necessário rodar a coleta de novo (./start_expert.sh) para que
   `episode.json` passe a ter essas duas colunas.

2. As imagens não seguem o padrão `frame_%04d.png`: o projeto grava
   `central_rgb/%04d.png`, `left_rgb/%04d.png`, `right_rgb/%04d.png` e
   `birdview_masks/%04d_00.png`. `build_dataset_from_carla_episode` aqui aceita
   escolher a fonte (`source='birdview'` ou `source='central_rgb'`) e já monta
   o nome de arquivo correto para cada uma.

3. `extract_future_trajectory` original incluía o próprio instante t como
   primeiro ponto da trajetória futura, o que faz o primeiro ponto ser sempre
   (0, 0) no referencial local (perde-se 1 dos H pontos pedidos). Aqui o
   horizonte começa em t + stride, igual ao notebook de referência.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


def global_to_local(points_global, vehicle_xy, vehicle_yaw):
    """Converte pontos [N, 2] do referencial global para o referencial local do veículo."""
    points_global = np.asarray(points_global)
    vehicle_xy = np.asarray(vehicle_xy)
    translated = points_global - vehicle_xy
    c, s = np.cos(vehicle_yaw), np.sin(vehicle_yaw)
    R = np.array([[c, s],
                  [-s, c]])
    return translated @ R.T


def extract_future_trajectory(poses, t, H=20, stride=1):
    """
    Extrai H pontos futuros (t+stride, t+2*stride, ..., t+H*stride) no
    referencial local do veículo no instante t.

    `poses` deve ser um array (T, 3) com colunas [x, y, yaw_rad].
    Retorna None se o episódio não tiver horizonte futuro suficiente.
    """
    poses = np.asarray(poses)
    idx = t + stride * np.arange(1, H + 1)
    if idx[-1] >= len(poses):
        return None

    vehicle_xy = poses[t, :2]
    vehicle_yaw = poses[t, 2]
    future_xy = poses[idx, :2]
    return global_to_local(future_xy, vehicle_xy, vehicle_yaw)


def load_episode(episode_dir):
    """
    Lê episode.json de uma pasta route_XX/ep_00 e devolve um dicionário com
    arrays numpy: done, actions, state, pose, command.

    Lança um erro claro se o episódio foi coletado antes do data_collect.py
    ser atualizado (sem as colunas pose/command).
    """
    episode_dir = Path(episode_dir)
    episode_json = episode_dir / 'episode.json'
    df = pd.read_json(episode_json)

    missing = [c for c in ('pose', 'command') if c not in df.columns]
    if missing:
        raise ValueError(
            f"{episode_json} não tem a(s) coluna(s) {missing}. "
            "Esse episódio foi coletado com a versão antiga do data_collect.py "
            "(sem pose/command). Rode ./start_expert.sh de novo com o "
            "data_collect.py atualizado antes de montar o dataset."
        )

    return {
        'done': df['done'].to_numpy(),
        'actions': np.stack(df['actions'].to_numpy()),
        'state': np.stack(df['state'].to_numpy()),
        'pose': np.stack(df['pose'].to_numpy()),
        'command': df['command'].to_numpy(),
    }


def _image_path(episode_dir, source, t):
    episode_dir = Path(episode_dir)
    if source == 'birdview':
        # único canal de history_idx salvo pelo data_collect.py é o i_mask=0
        return episode_dir / 'birdview_masks' / f'{t:04d}_00.png'
    elif source in ('central_rgb', 'left_rgb', 'right_rgb'):
        return episode_dir / source / f'{t:04d}.png'
    else:
        raise ValueError(f"source inválido: {source!r}")


def build_dataset_from_carla_episode(episode_dir, H=20, stride=1, img_size=192,
                                      source='birdview', route_id=None):
    """
    Monta a lista de amostras de UM episódio (route_XX/ep_00), lendo os dados
    reais gravados pelo carla-bc.

    Parâmetros
    - episode_dir: caminho para a pasta do episódio, ex. 'gail_experts/route_00/ep_00'.
    - H, stride: horizonte temporal futuro, iguais aos usados no notebook.
    - img_size: tamanho para redimensionar a imagem (use 192 para bater com o
      birdview original da rede, ou um valor menor só para inspeção).
    - source: 'birdview' (3 canais via/rota/faixa, o que a rede realmente usa)
      ou 'central_rgb'/'left_rgb'/'right_rgb' (câmeras, não usadas no treino
      atual, mas disponíveis caso você queira experimentar com elas).
    - route_id: número da rota (ex. 0 para route_00), só para rotular a
      amostra depois (não influencia o cálculo). Se None, tenta descobrir
      a partir do nome da pasta em episode_dir.
    """
    episode_dir = Path(episode_dir)
    ep = load_episode(episode_dir)
    poses = ep['pose']
    commands = ep['command']

    # NOVO: se route_id não foi passado, tenta ler do nome da pasta
    # (gail_experts/route_00/ep_00 -> route_id=0), só para rotular as
    # amostras depois — não é usado em nenhum cálculo.
    if route_id is None:
        route_dir_name = episode_dir.parent.name  # ex: 'route_00'
        try:
            route_id = int(route_dir_name.split('_')[-1])
        except ValueError:
            route_id = route_dir_name  # nome não numérico: guarda como texto mesmo

    samples = []
    discarded_count = 0

    for t in range(len(poses)):
        tau = extract_future_trajectory(poses, t, H=H, stride=stride)
        if tau is None:
            discarded_count += 1
            continue

        img_path = _image_path(episode_dir, source, t)
        if not img_path.exists():
            discarded_count += 1
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_size is not None:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        # 'command' pode vir como número solto (coleta antiga) ou como lista
        # de 1 elemento (coleta atual, que corrige a armadilha de
        # th.Tensor(int) em expert_dataset.py) — aceita os dois formatos.
        command_t = np.asarray(commands[t]).reshape(-1)[0]

        samples.append({
            'bev': img,
            'command': int(command_t),
            'pose': poses[t].astype(np.float32),
            'action': ep['actions'][t].astype(np.float32),
            'future_trajectory': tau.astype(np.float32),
            # NOVO: de onde essa amostra veio, para rotular gráficos depois.
            'route_id': route_id,
            't': t,
            'image_path': str(img_path),
        })

    return samples, discarded_count


def build_dataset_from_routes(gail_experts_dir, route_ids=None, ep_id=0, **kwargs):
    """
    Percorre várias rotas de uma vez (gail_experts/route_XX/ep_00) e concatena
    as amostras. route_ids=None usa todas as pastas route_* encontradas.
    """
    gail_experts_dir = Path(gail_experts_dir)
    if route_ids is None:
        route_dirs = sorted(gail_experts_dir.glob('route_*'))
    else:
        route_dirs = [gail_experts_dir / f'route_{r:02d}' for r in route_ids]

    all_samples = []
    total_discarded = 0
    for route_dir in route_dirs:
        episode_dir = route_dir / f'ep_{ep_id:02d}'
        if not episode_dir.exists():
            print(f'aviso: {episode_dir} não existe, pulando')
            continue
        samples, discarded = build_dataset_from_carla_episode(episode_dir, **kwargs)
        print(f'{route_dir.name}: {len(samples)} amostras válidas, {discarded} descartadas')
        all_samples.extend(samples)
        total_discarded += discarded

    return all_samples, total_discarded