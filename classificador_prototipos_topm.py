"""
Resolve as atividades de 05_classificador_prototipos_topM.ipynb com os dados REAIS
do carla-bc (real_trajectory_dataset.npz, gerado por figuras_trajetorias_futuras.py).

Ideia do notebook: em vez de regredir UMA trajetória contínua (notebook 04), a rede
decide QUAL protótipo de trajetória (obtido com k-means, notebook 03) é o mais
provável, e devolve uma distribuição de probabilidade sobre os K protótipos.
Aqui a entrada é BEV + comando, como no 04.

Atividade 1: avaliar top-1, top-3, top-5 e top-10 no dataset real.
Atividade 2: ver casos em que o protótipo correto está no top-M mas NÃO no top-1.
Atividade 3: comparar os protótipos escolhidos com a trajetória da regressão direta (04).

Pré-requisitos (na mesma pasta): baseline_regressao_direta.py e
figuras_trajetorias_futuras.py. Para a Atividade 3, rode antes o 04:
    python baseline_regressao_direta.py

Uso:
    python classificador_prototipos_topM.py
    python classificador_prototipos_topM.py --K 32 --M 5 --epochs 30 --val-routes 8 9
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Peças reaproveitadas do notebook 04 (mesmos dados, mesma divisão, mesmo backbone)
from baseline_regressao_direta import (
    load_dataset, split_indices, commands_to_idx, bev_to_uint8, ade_fde,
    predict as predict_regression, BEVTrajectoryRegressor, pick_examples, nome_cmd,
    COMMAND_TO_IDX,
)
# Conversão metros -> pixels do BEV (para desenhar trajetórias sobre a imagem)
from figuras_trajetorias_futuras import local_to_bev_pixels

KS_ACURACIA = (1, 3, 5, 10)   # os k pedidos no enunciado


# --------------------------------------------------------------------------
# Protótipos (k-means) e rótulos
# --------------------------------------------------------------------------
def fit_prototypes(tau_train, K, seed=0):
    """
    k-means SOMENTE nas trajetórias de treino. Os protótipos (centros) são
    as K "formas típicas" de trajetória. Devolve o objeto KMeans (para rotular
    a validação depois), os protótipos (K, H, 2) e os rótulos do treino.
    """
    N, H, _ = tau_train.shape
    K_eff = min(K, N)
    kmeans = KMeans(n_clusters=K_eff, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(tau_train.reshape(N, -1))
    prototypes = kmeans.cluster_centers_.reshape(K_eff, H, 2).astype(np.float32)
    return kmeans, prototypes, labels.astype(np.int64)


def assign_prototypes(kmeans, tau):
    """Rótulo de cada trajetória = índice do protótipo mais próximo (distância euclidiana)."""
    return kmeans.predict(tau.reshape(len(tau), -1)).astype(np.int64)


# --------------------------------------------------------------------------
# Métricas (só numpy)
# --------------------------------------------------------------------------
def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_np(logits, labels):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float((lse - logits[np.arange(len(labels)), labels]).mean())


def true_label_rank(logits, labels):
    """Posição (0 = 1º lugar) que o protótipo CORRETO ocupa no ranking de probabilidade."""
    true_logit = logits[np.arange(len(labels)), labels]
    return (logits > true_logit[:, None]).sum(axis=1)


def topk_accuracy(logits, labels, ks):
    """Fração de amostras cujo protótipo correto está entre os k mais prováveis."""
    rank = true_label_rank(logits, labels)
    return {int(k): float((rank < k).mean()) for k in ks}


def candidate_errors(real, prototypes):
    """
    Erro geométrico de CADA protótipo contra cada trajetória real.
    real (N,H,2), prototypes (K,H,2) -> ADE (N,K) e FDE (N,K), em metros.
    """
    dist = np.linalg.norm(real[:, None, :, :] - prototypes[None, :, :, :], axis=-1)  # (N,K,H)
    return dist.mean(axis=2), dist[:, :, -1]


def min_error_at_M(ade_all, fde_all, logits, Ms):
    """
    minADE@M / minFDE@M: para cada amostra, olha só os M protótipos que a rede
    considera mais prováveis e fica com o MELHOR deles (o mais próximo da
    trajetória real). Mede "se um planejador escolhesse bem entre os M
    candidatos, quão perto da realidade ele chegaria?".
    """
    order = np.argsort(-logits, axis=1)
    resultado = {}
    for M in Ms:
        top = order[:, :M]
        resultado[int(M)] = {
            'minADE': float(np.take_along_axis(ade_all, top, axis=1).min(axis=1).mean()),
            'minFDE': float(np.take_along_axis(fde_all, top, axis=1).min(axis=1).mean()),
        }
    return resultado, order


def pick_hard_cases(rank, M, rng, max_cases):
    """Índices em que o protótipo correto está no top-M, mas NÃO é o 1º colocado."""
    candidatos = np.where((rank >= 1) & (rank < M))[0]
    if len(candidatos) > max_cases:
        candidatos = np.sort(rng.choice(candidatos, size=max_cases, replace=False))
    return candidatos.tolist()


# --------------------------------------------------------------------------
# Modelo: mesmo backbone do notebook 04, só muda a "cabeça" (saída)
# --------------------------------------------------------------------------
class BEVPrototypeClassifier(BEVTrajectoryRegressor):
    """
    CNN no BEV + embedding do comando (idêntico ao regressor do 04), mas a última
    camada devolve K LOGITS (um por protótipo) em vez de H*2 coordenadas.
    O softmax dos logits é a distribuição de probabilidade sobre os protótipos.
    """

    def __init__(self, K, emb_dim=16):
        super().__init__(H=1, emb_dim=emb_dim)  # H=1 só para construir; a última camada é trocada abaixo
        self.head[-1] = nn.Linear(256, K)


@torch.no_grad()
def predict_logits(model, bev_u8, cmd_idx, device, batch_size=256):
    model.eval()
    saidas = []
    for i in range(0, len(bev_u8), batch_size):
        xb = bev_u8[i:i + batch_size].to(device).float() / 255.0
        cb = cmd_idx[i:i + batch_size].to(device)
        saidas.append(model(xb, cb).cpu())
    return torch.cat(saidas).numpy()


def load_regression_predictions(ckpt_path, bev_val, cmd_val, H, device, val_routes):
    """
    Carrega o regressor treinado pelo notebook 04 e prediz a trajetória das
    amostras de validação (em metros). Devolve None se o checkpoint não existir
    ou não for compatível (o resto do script funciona sem ele).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f'\n(aviso) {ckpt_path} não existe: a comparação com a regressão direta '
              f'será pulada. Rode antes:  python baseline_regressao_direta.py')
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    if int(ckpt['H']) != H:
        print(f"\n(aviso) o regressor foi treinado com H={ckpt['H']}, mas o dataset tem H={H}: "
              f'comparação pulada.')
        return None

    # Comparação só é justa se a regressão NÃO treinou nas rotas que usamos para validar aqui.
    metricas_04 = ckpt_path.parent / 'metricas.json'
    if metricas_04.exists():
        val_04 = json.load(open(metricas_04)).get('val_routes')
        if val_04 != val_routes:
            print(f'\n(aviso) a regressão (04) validou nas rotas {val_04}, mas aqui a validação é '
                  f'{val_routes}. Rotas que a regressão viu no treino deixam a comparação injusta. '
                  f'Use --val-routes igual nos dois scripts.')

    reg = BEVTrajectoryRegressor(H).to(device)
    reg.load_state_dict(ckpt['state_dict'])
    return predict_regression(reg, bev_val, cmd_val, float(ckpt['y_scale']), H, device)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ep = np.arange(1, len(history['train_loss']) + 1)
    ax1.plot(ep, history['train_loss'], label='treino')
    ax1.plot(ep, history['val_loss'], label='validação')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Entropia cruzada')
    ax1.set_title('Perda do classificador de protótipos')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(ep, history['val_top1'], label='top-1')
    ax2.plot(ep, history['val_top5'], label='top-5')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia de validação')
    ax2.set_ylim(0, 1.02)
    ax2.set_title('Acurácia top-k na validação')
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_summary(acc_val, acc_train, minerr, piso, melhor_possivel, reg_ade, save_path):
    """Esquerda: acurácia top-k x k. Direita: minADE@M x M com linhas de referência."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ks = sorted(acc_val)
    ax1.plot(ks, [acc_val[k] for k in ks], marker='o', label='validação')
    ax1.plot(ks, [acc_train[k] for k in ks], marker='o', linestyle='--', label='treino')
    ax1.set_xlabel('k')
    ax1.set_ylabel('acurácia top-k (protótipo correto entre os k mais prováveis)')
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(ks)
    ax1.set_title('Acurácia top-k', fontsize=10)
    ax1.legend()
    ax1.grid(True)

    Ms = sorted(minerr)
    ax2.plot(Ms, [minerr[M]['minADE'] for M in Ms], marker='o', label='minADE@M (melhor dos M candidatos)')
    ax2.axhline(piso, color='tab:green', linestyle='--', label='piso: protótipo correto (erro de quantização)')
    ax2.axhline(melhor_possivel, color='tab:gray', linestyle=':', label='melhor dos K protótipos (oráculo)')
    if reg_ade is not None:
        ax2.axhline(reg_ade, color='tab:orange', linestyle='-.', label='regressão direta (04)')
    ax2.set_xlabel('M (nº de candidatos)')
    ax2.set_ylabel('ADE (m)')
    ax2.set_xticks(Ms)
    ax2.set_title('Erro geométrico: melhor candidato do top-M', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cases(cases, M, save_path):
    """
    Uma linha por caso. Esquerda: BEV com a trajetória real (ciano) e o top-1
    (vermelho) por cima. Direita: trajetória real, os M candidatos (com
    probabilidade), o protótipo correto e, se existir, a regressão direta.
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 4.0 * n), squeeze=False)
    cores = plt.cm.viridis(np.linspace(0.15, 0.85, max(M - 1, 1)))
    for row, c in enumerate(cases):
        ax_img, ax = axes[row]
        h_img, w_img = c['bev'].shape[:2]
        ax_img.imshow(c['bev'])
        col, rw = local_to_bev_pixels(c['real'], w_img, h_img)
        col1, rw1 = local_to_bev_pixels(c['protos'][0], w_img, h_img)
        ev_c, ev_r = local_to_bev_pixels(np.zeros((1, 2)), w_img, h_img)
        ax_img.plot(col, rw, color='cyan', linewidth=1.8, label='real')
        ax_img.plot(col1, rw1, color='red', linewidth=1.5, linestyle='--', label='top-1')
        ax_img.scatter(ev_c, ev_r, marker='s', s=40, color='white', edgecolors='black', zorder=5)
        ax_img.set_xlim(-0.5, w_img - 0.5)
        ax_img.set_ylim(h_img - 0.5, -0.5)
        ax_img.legend(loc='lower left', fontsize=6, framealpha=0.7)
        ax_img.set_title(c['title'], fontsize=8)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax.plot(c['real'][:, 0], c['real'][:, 1], color='black', marker='o', markersize=3,
                linewidth=2.5, label='trajetória real', zorder=4)
        if c['reg'] is not None:
            ade_r, _ = ade_fde(c['reg'][None], c['real'][None])
            ax.plot(c['reg'][:, 0], c['reg'][:, 1], color='tab:orange', marker='x', markersize=4,
                    linewidth=1.5, label=f'regressão direta (ADE {ade_r:.2f} m)', zorder=3)
        for r, (k, p, proto) in enumerate(zip(c['proto_ids'], c['probs'], c['protos'])):
            correto = (k == c['true_id'])
            cor = 'red' if r == 0 else cores[min(r - 1, len(cores) - 1)]
            ade_p, _ = ade_fde(proto[None], c['real'][None])
            ax.plot(proto[:, 0], proto[:, 1], color=cor, linewidth=3 if correto else 1.3,
                    linestyle='-' if correto else '--', alpha=0.95 if correto else 0.8,
                    label=f"top {r + 1}: protótipo {k} (p={p:.2f}, ADE {ade_p:.2f})"
                          + (' ← correto' if correto else ''))
        if c['true_id'] not in list(c['proto_ids']):
            ax.plot(c['true_proto'][:, 0], c['true_proto'][:, 1], color='tab:green', linewidth=2.5,
                    linestyle=':', label=f"correto: protótipo {c['true_id']} (fora do top-{M})")
        ax.scatter([0], [0], marker='s', s=50, color='k', zorder=5)
        ax.axis('equal')
        ax.set_xlabel('x local (frente, m)')
        ax.set_ylabel('y local (lateral, m)\n← esquerda (−)    direita (+) →', fontsize=8)
        ax.legend(fontsize=6, loc='best')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Montagem dos casos
# --------------------------------------------------------------------------
def build_case(i, bev_hwc, tau_val, command_val, z_val, logits_val, probs_val, order, prototypes,
               M, reg_pred, origem, rank):
    top = order[i, :M]
    return {
        'bev': bev_hwc,
        'real': tau_val[i],
        'protos': prototypes[top],
        'proto_ids': top,
        'probs': probs_val[i, top],
        'true_id': int(z_val[i]),
        'true_proto': prototypes[z_val[i]],
        'reg': None if reg_pred is None else reg_pred[i],
        'title': f'{nome_cmd(command_val[i])} | {origem} | correto no top-{int(rank[i]) + 1}',
    }


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='real_trajectory_dataset.npz')
    parser.add_argument('--out-dir', default='figures/classificador_prototipos')
    parser.add_argument('--K', type=int, default=32, help='nº de protótipos (k-means)')
    parser.add_argument('--M', type=int, default=5, help='nº de candidatos nas figuras top-M')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-routes', type=int, nargs='*', default=None)
    parser.add_argument('--regressor-ckpt', default='figures/baseline_regressao/baseline_regressao_bev.pth')
    parser.add_argument('--max-cases', type=int, default=6)
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f'{dataset_path} não existe. Rode antes:\n'
                         f'    python figuras_trajetorias_futuras.py')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dados e divisão (a mesma do notebook 04) ----
    bev, tau, command, route_id, t_arr = load_dataset(dataset_path)
    N, H, _ = tau.shape
    train_idx, val_idx, val_routes = split_indices(N, route_id, args.val_routes)
    print(f'Dataset: {N} amostras, H={H}, dispositivo={device}')
    print(f'Treino: {len(train_idx)} | Validação: {len(val_idx)} | rotas de validação: {val_routes}')
    tau_train, tau_val = tau[train_idx], tau[val_idx]
    command_train, command_val = command[train_idx], command[val_idx]

    # ---- protótipos: k-means só no treino; rótulo da validação = protótipo mais próximo ----
    kmeans, prototypes, z_train = fit_prototypes(tau_train, args.K, seed=args.seed)
    K = len(prototypes)
    z_val = assign_prototypes(kmeans, tau_val)
    cont_train = np.bincount(z_train, minlength=K)
    print(f'K={K} protótipos | membros no treino por protótipo: '
          f'min={cont_train.min()}, mediana={int(np.median(cont_train))}, max={cont_train.max()}')
    print(f'Protótipos com menos de 5 exemplos de treino: {int((cont_train < 5).sum())}')
    np.savez(out_dir / 'prototipos.npz', prototypes=prototypes, counts_train=cont_train)

    # ---- tensores ----
    cmd_idx = torch.from_numpy(commands_to_idx(command))
    bev_u8 = bev_to_uint8(bev)
    bev_train, cmd_train = bev_u8[train_idx], cmd_idx[train_idx]
    bev_val, cmd_val = bev_u8[val_idx], cmd_idx[val_idx]
    batch_size = min(args.batch_size, len(train_idx))
    loader = DataLoader(TensorDataset(bev_train, cmd_train, torch.from_numpy(z_train)),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- modelo e treino ----
    model = BEVPrototypeClassifier(K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f'Modelo: {sum(p.numel() for p in model.parameters()):,} parâmetros\n')

    history = {'train_loss': [], 'val_loss': [], 'val_top1': [], 'val_top5': []}
    best_loss, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        soma, n_lotes = 0.0, 0
        for xb, cb, zb in loader:
            xb = xb.to(device).float() / 255.0
            cb, zb = cb.to(device), zb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, cb), zb)
            loss.backward()
            opt.step()
            soma += loss.item()
            n_lotes += 1
        logits_v = predict_logits(model, bev_val, cmd_val, device)
        val_loss = cross_entropy_np(logits_v, z_val)
        acc = topk_accuracy(logits_v, z_val, (1, min(5, K)))
        history['train_loss'].append(soma / n_lotes)
        history['val_loss'].append(val_loss)
        history['val_top1'].append(acc[1])
        history['val_top5'].append(acc[min(5, K)])
        print(f'época {epoch:>3d}/{args.epochs} | CE treino {soma / n_lotes:6.3f} | CE val {val_loss:6.3f} | '
              f'top-1 {acc[1]:.3f} | top-5 {acc[min(5, K)]:.3f}')
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f'\nMelhor época (menor CE de validação): {best_epoch}')

    # ---- Atividade 1: top-1/3/5/10 ----
    logits_val = predict_logits(model, bev_val, cmd_val, device)
    logits_train = predict_logits(model, bev_train, cmd_train, device)
    ks = [k for k in KS_ACURACIA if k <= K]
    ks_curva = list(range(1, min(10, K) + 1))
    acc_val = topk_accuracy(logits_val, z_val, ks_curva)
    acc_train = topk_accuracy(logits_train, z_train, ks_curva)
    print('\n=== Atividade 1: acurácia top-k (protótipo correto entre os k mais prováveis) ===')
    print(f"{'k':>4}{'validação':>12}{'treino':>10}")
    for k in ks:
        print(f'{k:>4}{acc_val[k]:>12.3f}{acc_train[k]:>10.3f}')
    print(f'(acaso puro: top-1 = {1 / K:.3f}; top-k = {min(1.0, k / K):.3f} para o maior k acima)')

    print('\nPor comando (validação):')
    print(f"{'comando':<20}{'n':>6}{'top-1':>8}{'top-5':>8}")
    rank_val = true_label_rank(logits_val, z_val)
    por_comando = {}
    for c in sorted(np.unique(command_val).tolist()):
        m = command_val == c
        a1, a5 = float((rank_val[m] < 1).mean()), float((rank_val[m] < min(5, K)).mean())
        por_comando[nome_cmd(c)] = {'n': int(m.sum()), 'top1': a1, 'top5': a5}
        print(f'{nome_cmd(c):<20}{int(m.sum()):>6}{a1:>8.3f}{a5:>8.3f}')

    # ---- erro geométrico: minADE@M, piso de quantização e regressão ----
    ade_all, fde_all = candidate_errors(tau_val, prototypes)
    minerr, order = min_error_at_M(ade_all, fde_all, logits_val, ks_curva)
    piso = float(ade_all[np.arange(len(z_val)), z_val].mean())
    melhor_possivel = float(ade_all.min(axis=1).mean())
    reg_pred = load_regression_predictions(args.regressor_ckpt, bev_val, cmd_val, H, device, val_routes)
    reg_ade = reg_fde = None
    if reg_pred is not None:
        reg_ade, reg_fde = ade_fde(reg_pred, tau_val)

    print('\n=== Erro geométrico na validação (metros) ===')
    print(f"{'método':<46}{'ADE':>8}{'FDE':>8}")
    if reg_ade is not None:
        print(f"{'regressão direta (04)':<46}{reg_ade:>8.3f}{reg_fde:>8.3f}")
    for M in ks:
        print(f"{f'classificador, melhor dos top-{M} (minADE/minFDE@{M})':<46}"
              f"{minerr[M]['minADE']:>8.3f}{minerr[M]['minFDE']:>8.3f}")
    fde_piso = float(fde_all[np.arange(len(z_val)), z_val].mean())
    print(f"{'piso: usar o protótipo CORRETO (quantização)':<46}{piso:>8.3f}{fde_piso:>8.3f}")
    print(f"{'oráculo: melhor dos K protótipos':<46}{melhor_possivel:>8.3f}{float(fde_all.min(axis=1).mean()):>8.3f}")

    # ---- figuras ----
    plot_curves(history, out_dir / 'curvas_de_treino.png')
    plot_summary(acc_val, acc_train, minerr, piso, melhor_possivel, reg_ade, out_dir / 'metricas_topk.png')

    probs_val = softmax_np(logits_val)
    M = min(args.M, K)

    def origem(i):
        j = val_idx[i]
        return (f'route_{int(route_id[j]):02d}, t={int(t_arr[j]):04d}'
                if route_id is not None and t_arr is not None else f'amostra {int(j)}')

    def montar(indices):
        return [build_case(i, np.transpose(bev[val_idx[i]], (1, 2, 0)), tau_val, command_val, z_val,
                           logits_val, probs_val, order, prototypes, M, reg_pred, origem(i), rank_val)
                for i in indices]

    # Atividade 2: protótipo correto no top-M, mas não no top-1
    n_dificeis = int(((rank_val >= 1) & (rank_val < M)).sum())
    print(f'\n=== Atividade 2: correto no top-{M} mas não no top-1 ===')
    print(f'{n_dificeis} de {len(rank_val)} amostras de validação ({n_dificeis / len(rank_val):.1%}) '
          f'= top-{M} ({acc_val.get(M, float((rank_val < M).mean())):.3f}) − top-1 ({acc_val[1]:.3f})')
    dificeis = pick_hard_cases(rank_val, M, rng, args.max_cases)
    if dificeis:
        plot_cases(montar(dificeis), M, out_dir / 'casos_topM_nao_top1.png')
        print(f'  {len(dificeis)} casos desenhados em {out_dir / "casos_topM_nao_top1.png"}')
    else:
        print('  nenhum caso desse tipo na validação (o classificador acerta no top-1 ou erra além do top-M).')

    # Atividade 3: protótipos escolhidos x regressão direta, em exemplos variados por comando
    estratificados = pick_examples(command_val, rng, per_cmd=2, max_total=6)
    plot_cases(montar(estratificados), M, out_dir / 'comparacao_regressao_prototipos.png')
    print(f'\nAtividade 3: comparação desenhada em {out_dir / "comparacao_regressao_prototipos.png"}'
          + ('' if reg_pred is not None else ' (sem a curva da regressão: checkpoint do 04 ausente)'))

    # ---- salvar ----
    torch.save({'state_dict': best_state, 'K': K, 'H': H, 'prototypes': prototypes,
                'command_to_idx': COMMAND_TO_IDX},
               out_dir / 'classificador_prototipos.pth')
    metricas = {
        'K': K, 'M': M, 'melhor_epoca': best_epoch, 'val_routes': val_routes,
        'topk_val': {str(k): v for k, v in acc_val.items()},
        'topk_treino': {str(k): v for k, v in acc_train.items()},
        'por_comando': por_comando,
        'min_erro_em_M': {str(k): v for k, v in minerr.items()},
        'piso_quantizacao_ade': piso, 'oraculo_melhor_prototipo_ade': melhor_possivel,
        'regressao_direta': None if reg_ade is None else {'ade': reg_ade, 'fde': reg_fde},
        'casos_topM_nao_top1': {'n': n_dificeis, 'total_val': int(len(rank_val))},
        'historico': history,
    }
    with open(out_dir / 'metricas.json', 'w') as f:
        json.dump(metricas, f, indent=2)
    print(f'\nArquivos em {out_dir}/: curvas_de_treino.png, metricas_topk.png, casos_topM_nao_top1.png, '
          f'comparacao_regressao_prototipos.png, metricas.json, prototipos.npz, classificador_prototipos.pth')


if __name__ == '__main__':
    main()