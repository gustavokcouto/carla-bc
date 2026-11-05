import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Nomes legíveis para os valores de RoadOption usados como 'command'
# (ver carla_gym/core/task_actor/common/navigation/map_utils.py)
COMMAND_NAMES = {
    -1: 'VOID',
    1: 'LEFT',
    2: 'RIGHT',
    3: 'STRAIGHT',
    4: 'LANEFOLLOW',
    5: 'CHANGELANELEFT',
    6: 'CHANGELANERIGHT',
}


def fit_kmeans(tau, K, seed=0):
    """
    Agrupa trajetórias futuras com k-means, igual ao notebook:
    achata (N, H, 2) -> (N, H*2), roda o KMeans, e desfaz o achatamento dos
    centros dos clusters (os 'protótipos') de volta para (K, H, 2).
    """
    N, H, _ = tau.shape
    X = tau.reshape(N, -1) #achatamento da matriz, para virar um ponto no espaço de 40 dimensões
    K_eff = min(K, N)  # não dá pra pedir mais clusters do que amostras
    kmeans = KMeans(n_clusters=K_eff, random_state=seed, n_init=10) #roda o algoritmo 10 vezes e fica com o melhor
    labels = kmeans.fit_predict(X)
    prototypes = kmeans.cluster_centers_.reshape(K_eff, H, 2)
    return prototypes, labels, kmeans.inertia_


def plot_prototypes(prototypes, title, save_path):
    plt.figure(figsize=(6, 6))
    for k in range(len(prototypes)):
        plt.plot(prototypes[k, :, 0], prototypes[k, :, 1], marker='o', markersize=2, linewidth=1.5)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('x local (frente, m)')
    plt.ylabel('y local (lateral, m)')
    plt.savefig(save_path)
    plt.close()
    print(f'  figura salva em {save_path}')


def atividade_1(tau, out_dir, seed):
    """K=32 e K=64, além do K=16 original, para comparar."""
    print('\n=== k-means com K=32 e K=64 ===')
    resultados = {}
    for K in (16, 32, 64):
        prototypes, labels, inertia = fit_kmeans(tau, K, seed=seed)
        resultados[K] = (prototypes, labels, inertia)
        print(f'K={K:>2d}: {len(prototypes)} protótipos, inércia={inertia:.2f}')
        plot_prototypes(
            prototypes,
            f'{len(prototypes)} trajetórias prototípicas (dados reais)',
            out_dir / f'prototipos_K{K}.png',
        )
    return resultados


def atividade_2(tau, command, out_dir, seed, K_por_comando=8): #alterar K_por_comando para alterar o número de k-means por comando
    """
    Um k-means PARA CADA valor de 'command' encontrado no dataset real,
    para ver se os protótipos realmente mudam de formato conforme o
    comando de navegação.
    """
    print('\n=== Protótipos separados por comando de navegação ===')
    comandos_presentes = sorted(np.unique(command).tolist())
    n_cmds = len(comandos_presentes)

    fig, axes = plt.subplots(1, n_cmds, figsize=(5 * n_cmds, 5), squeeze=False)
    axes = axes[0]

    for ax, cmd in zip(axes, comandos_presentes):
        mask = command == cmd
        tau_cmd = tau[mask]
        nome = COMMAND_NAMES.get(int(cmd), str(cmd))

        if len(tau_cmd) < 2:
            ax.set_title(f'command={cmd} ({nome})\n(poucas amostras: {len(tau_cmd)})')
            continue

        prototypes, _, inertia = fit_kmeans(tau_cmd, K_por_comando, seed=seed)
        print(f'command={cmd} ({nome}): {len(tau_cmd)} amostras, '
              f'{len(prototypes)} protótipos, inércia={inertia:.2f}')

        for k in range(len(prototypes)):
            ax.plot(prototypes[k, :, 0], prototypes[k, :, 1], marker='o', markersize=2, linewidth=1.5)
        ax.axis('equal')
        ax.set_title(f'command={cmd} ({nome})\n{len(tau_cmd)} amostras')
        ax.set_xlabel('x local (frente, m)')
        ax.set_ylabel('y local (lateral, m)')

    fig.suptitle('Protótipos de trajetória por comando de navegação')
    fig.tight_layout()
    save_path = out_dir / 'prototipos_por_comando.png'
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  figura salva em {save_path}')


def atividade_3(tau, out_dir, seed, K=16, n_prototipos=10, n_vizinhos=5):
    """
    Pega os K=16 protótipos já calculados, escolhe os 'n_prototipos' com
    MAIS trajetórias associadas (os mais representativos), e para cada um
    encontra as 'n_vizinhos' trajetórias reais mais próximas dele (menor
    distância euclidiana no espaço achatado (H*2,), a mesma distância que
    o k-means usa para formar os clusters).
    """
    print(f'\n=== {n_prototipos} protótipos e suas '
          f'{n_vizinhos} trajetórias reais mais próximas ===')

    N, H, _ = tau.shape
    X = tau.reshape(N, -1)
    prototypes, labels, _ = fit_kmeans(tau, K, seed=seed)

    # os clusters com mais membros = protótipos mais representativos
    tamanhos = np.bincount(labels, minlength=len(prototypes))
    escolhidos = np.argsort(-tamanhos)[:n_prototipos]

    n_linhas = int(np.ceil(len(escolhidos) / 5))
    fig, axes = plt.subplots(n_linhas, 5, figsize=(20, 4 * n_linhas), squeeze=False)
    axes_flat = axes.flatten()

    for ax, k in zip(axes_flat, escolhidos):
        proto_flat = prototypes[k].reshape(-1)
        distancias = np.linalg.norm(X - proto_flat, axis=1)
        vizinhos_idx = np.argsort(distancias)[:n_vizinhos]

        for idx in vizinhos_idx:
            ax.plot(tau[idx, :, 0], tau[idx, :, 1], alpha=0.4, linewidth=1, color='tab:blue')
        ax.plot(prototypes[k, :, 0], prototypes[k, :, 1], marker='o', markersize=3,
                linewidth=2.5, color='tab:red', label='protótipo')
        ax.axis('equal')
        ax.set_title(f'protótipo {k} ({tamanhos[k]} membros)')
        ax.legend(fontsize=7)

        print(f'protótipo {k} ({tamanhos[k]} membros): '
              f'5 vizinhos mais próximos = índices {vizinhos_idx.tolist()}')

    for ax in axes_flat[len(escolhidos):]:
        ax.axis('off')

    fig.suptitle(f'{len(escolhidos)} protótipos mais representativos e suas '
                 f'{n_vizinhos} trajetórias reais mais próximas')
    fig.tight_layout()
    save_path = out_dir / 'prototipos_com_vizinhos.png'
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  figura salva em {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='real_trajectory_dataset.npz')
    parser.add_argument('--out-dir', default='figures/kmeans_prototipos')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(
            f'{dataset_path} não existe. Rode antes:\n'
            f'    python figuras_trajetorias_futuras.py\n'
            f'para gerar esse arquivo a partir de gail_experts/.'
        )

    d = np.load(dataset_path)
    tau = d['tau']          # (N, H, 2): trajetória futura, referencial do carro
    command = d['command']  # (N,): comando de navegação de cada amostra

    print(f'Dataset carregado: {dataset_path} -> tau {tau.shape}, '
          f'{len(np.unique(command))} comandos distintos: {sorted(np.unique(command).tolist())}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atividade_1(tau, out_dir, args.seed)
    atividade_2(tau, command, out_dir, args.seed)
    atividade_3(tau, out_dir, args.seed)


if __name__ == '__main__':
    main()