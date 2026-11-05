"""
Resolve as atividades de 04_baseline_regressao_direta.ipynb com os dados REAIS
do carla-bc (real_trajectory_dataset.npz, gerado por figuras_trajetorias_futuras.py).

No notebook, a entrada era sintética (5 números: comando one-hot, velocidade e
curvatura). Aqui a entrada é a imagem BEV + o comando de navegação, e a saída
continua sendo a trajetória futura contínua (H pontos x,y no referencial do carro).

Atividade 1: adaptar a entrada para BEV + comando no dataset real.
             -> classe BEVTrajectoryRegressor (CNN no BEV + embedding do comando).
Atividade 2: salvar curvas de perda e exemplos de predição.
             -> curvas_de_perda.png, exemplos_predicao.png (+ metricas.json, .pth).
Atividade 3: comparar erros por tipo de comando (esquerda, frente, direita).
             -> tabela no terminal, erro_por_comando.png e metricas.json.

Uso:
    python baseline_regressao_direta.py
    python baseline_regressao_direta.py --epochs 40 --val-routes 8 9
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Valores de RoadOption gravados em 'command' por data_collect.py
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
# Mapeamento fixo comando -> índice 0..6 (para a camada de embedding)
COMMAND_TO_IDX = {c: i for i, c in enumerate(sorted(COMMAND_NAMES))}
N_COMMANDS = len(COMMAND_TO_IDX)

# Os três grupos do enunciado. "frente" junta STRAIGHT (3) e LANEFOLLOW (4).
GRUPOS_NOTEBOOK = {'esquerda': [1], 'frente': [3, 4], 'direita': [2]}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def load_dataset(path):
    d = np.load(path)
    bev = d['bev']                                   # (N, 3, S, S) float em [0, 1]
    tau = d['tau'].astype(np.float32)                # (N, H, 2) trajetória futura em metros
    command = d['command'].astype(np.int64)         # (N,) valores de RoadOption
    route_id = d['route_id'] if 'route_id' in d.files else None
    t = d['t'] if 't' in d.files else None
    return bev, tau, command, route_id, t


def commands_to_idx(command):
    desconhecidos = sorted(set(command.tolist()) - set(COMMAND_TO_IDX))
    if desconhecidos:
        raise ValueError(f'comandos fora do esperado: {desconhecidos}')
    return np.array([COMMAND_TO_IDX[c] for c in command.tolist()], dtype=np.int64)


def split_indices(n, route_id, val_routes=None, val_frac=0.2):
    """
    Separa treino e validação POR ROTA (como o learn_bc.py faz: rotas inteiras
    para validação). Quadros consecutivos de uma mesma rota são quase
    idênticos; se caíssem nos dois conjuntos, a validação pareceria melhor
    do que realmente é.

    Se o .npz não tiver route_id (gerado por uma versão antiga do script),
    usa um corte em bloco: os últimos val_frac das amostras viram validação
    (como as amostras são empilhadas rota a rota, isso equivale, na prática,
    às últimas rotas).
    """
    if route_id is None:
        corte = int(n * (1 - val_frac))
        return np.arange(corte), np.arange(corte, n), None

    rotas = sorted(np.unique(route_id).tolist())
    if val_routes is None:
        n_val = max(1, int(round(len(rotas) * val_frac)))
        val_routes = rotas[-n_val:]
    ausentes = [r for r in val_routes if r not in rotas]
    if ausentes:
        raise ValueError(f'rotas {ausentes} não existem no dataset (existem: {rotas})')
    val_mask = np.isin(route_id, val_routes)
    if val_mask.all():
        raise ValueError('todas as rotas foram marcadas como validação; sobrou nada para treino')
    return np.where(~val_mask)[0], np.where(val_mask)[0], list(val_routes)


def bev_to_uint8(bev, chunk=500):
    """
    float32 em [0,1] -> tensor uint8. Guarda o dataset em 1 byte por pixel
    (4x menos RAM) e só converte de volta para float, em lotes, na hora do treino.
    """
    out = torch.empty(bev.shape, dtype=torch.uint8)
    for i in range(0, len(bev), chunk):
        bloco = np.clip(np.rint(bev[i:i + chunk] * 255.0), 0, 255).astype(np.uint8)
        out[i:i + chunk] = torch.from_numpy(bloco)
    return out


# --------------------------------------------------------------------------
# Métricas (só numpy)
# --------------------------------------------------------------------------
def ade_fde(pred, target):
    """
    ADE: erro médio (em metros) ao longo de todos os H pontos da trajetória.
    FDE: erro (em metros) só no ÚLTIMO ponto. pred/target: (N, H, 2).
    Mesma definição do notebook.
    """
    dist = np.linalg.norm(pred - target, axis=-1)   # (N, H)
    return float(dist.mean()), float(dist[:, -1].mean())


def errors_by_group(pred, target, command, grupos):
    """grupos: {nome: [valores de command]} -> {nome: {n, ade, fde}} (n=0 se vazio)."""
    resultado = {}
    for nome, valores in grupos.items():
        mask = np.isin(command, valores)
        if not mask.any():
            resultado[nome] = {'n': 0, 'ade': None, 'fde': None}
            continue
        ade, fde = ade_fde(pred[mask], target[mask])
        resultado[nome] = {'n': int(mask.sum()), 'ade': ade, 'fde': fde}
    return resultado


def mean_trajectory_baseline(tau_train, cmd_train, cmd_alvo):
    """
    Baseline "burra" de referência: para cada amostra, prediz a trajetória
    MÉDIA do treino para aquele comando (e a média geral se o comando nunca
    apareceu no treino). Serve para responder: a rede aprendeu algo além do
    que o comando sozinho já explica?
    """
    media_geral = tau_train.mean(axis=0)
    medias = {int(c): tau_train[cmd_train == c].mean(axis=0) for c in np.unique(cmd_train)}
    return np.stack([medias.get(int(c), media_geral) for c in cmd_alvo])


# --------------------------------------------------------------------------
# Modelo (Atividade 1: entrada = BEV + comando)
# --------------------------------------------------------------------------
class BEVTrajectoryRegressor(nn.Module):
    """
    CNN sobre o BEV (3 canais: via, rota, faixas) + embedding do comando,
    seguidos de um MLP que devolve H*2 números (a trajetória futura achatada).

    O AdaptiveAvgPool2d((4, 4)) final reduz o mapa de características para uma
    grade 4x4 FIXA (qualquer que seja o tamanho do BEV) e, ao contrário de um
    pooling global, preserva ONDE as coisas estão na imagem (onde a rota
    curva, por exemplo), que é justamente o que importa aqui.
    """

    def __init__(self, H, n_commands=N_COMMANDS, emb_dim=16):
        super().__init__()

        def bloco(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )

        self.cnn = nn.Sequential(
            bloco(3, 16), bloco(16, 32), bloco(32, 64), bloco(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                      # 128 * 4 * 4 = 2048
        )
        self.cmd_emb = nn.Embedding(n_commands, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2048 + emb_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, H * 2),
        )

    def forward(self, bev, cmd_idx):
        feat = self.cnn(bev)
        emb = self.cmd_emb(cmd_idx)
        return self.head(torch.cat([feat, emb], dim=1))


@torch.no_grad()
def predict(model, bev_u8, cmd_idx, y_scale, H, device, batch_size=256):
    """Devolve as trajetórias preditas em METROS, formato (N, H, 2)."""
    model.eval()
    saidas = []
    for i in range(0, len(bev_u8), batch_size):
        xb = bev_u8[i:i + batch_size].to(device).float() / 255.0
        cb = cmd_idx[i:i + batch_size].to(device)
        saidas.append(model(xb, cb).cpu())
    return (torch.cat(saidas) * y_scale).numpy().reshape(-1, H, 2)


# --------------------------------------------------------------------------
# Figuras (Atividades 2 e 3)
# --------------------------------------------------------------------------
def nome_cmd(c):
    return f'{c} ({COMMAND_NAMES.get(int(c), "?")})'


def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ep = np.arange(1, len(history['train_mse']) + 1)
    ax1.plot(ep, history['train_mse'], label='treino')
    ax1.plot(ep, history['val_mse'], label='validação')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('MSE (m²)')
    ax1.set_title('Perda (erro quadrático médio)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(ep, history['val_ade'], label='ADE')
    ax2.plot(ep, history['val_fde'], label='FDE')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('metros')
    ax2.set_title('Erro de validação (ADE / FDE)')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def pick_examples(command_val, rng, per_cmd=2, max_total=6):
    """Escolhe exemplos de validação variados: até per_cmd por tipo de comando."""
    escolhidos = []
    for c in sorted(np.unique(command_val).tolist()):
        idx = np.where(command_val == c)[0]
        k = min(per_cmd, len(idx))
        escolhidos.extend(rng.choice(idx, size=k, replace=False).tolist())
    return escolhidos[:max_total]


def plot_examples(images, gts, preds, titles, save_path):
    """Uma linha por exemplo: BEV à esquerda; trajetória real x predita à direita."""
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.6 * n), squeeze=False)
    for row, (img, gt, pr, title) in enumerate(zip(images, gts, preds, titles)):
        ax_img, ax_tr = axes[row]
        ax_img.imshow(img)
        ax_img.set_title(title, fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_tr.plot(gt[:, 0], gt[:, 1], marker='o', markersize=3, label='real')
        ax_tr.plot(pr[:, 0], pr[:, 1], marker='x', markersize=3, label='predita')
        ax_tr.scatter([0], [0], marker='s', s=60, color='k', label='veículo em t')
        ax_tr.axis('equal')
        ax_tr.set_xlabel('x local (frente, m)')
        ax_tr.set_ylabel('y local (lateral, m)')
        ax_tr.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_group_bars(tabela, save_path):
    """Barras de ADE e FDE por comando: rede x baseline da trajetória média."""
    nomes = [n for n, r in tabela.items() if r['n'] > 0]
    x = np.arange(len(nomes))
    largura = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metrica in zip(axes, ('ade', 'fde')):
        rede = [tabela[n]['rede'][metrica] for n in nomes]
        media = [tabela[n]['media_por_comando'][metrica] for n in nomes]
        ax.bar(x - largura / 2, rede, largura, label='rede (BEV + comando)')
        ax.bar(x + largura / 2, media, largura, label='trajetória média do comando')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={tabela[n]['n']})" for n in nomes], fontsize=8)
        ax.set_ylabel(f'{metrica.upper()} (m)')
        ax.set_title(f'{metrica.upper()} por comando (validação)')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def build_table(pred, target, command, base_pred, grupos):
    """Junta, por grupo de comando, o erro da rede e o da baseline da média."""
    rede = errors_by_group(pred, target, command, grupos)
    media = errors_by_group(base_pred, target, command, grupos)
    return {nome: {'n': rede[nome]['n'], 'rede': rede[nome], 'media_por_comando': media[nome]}
            for nome in grupos}


def print_table(titulo, tabela):
    print(f'\n{titulo}')
    print(f"{'grupo':<22}{'n':>6}   {'ADE rede':>9}{'FDE rede':>10}   {'ADE média':>10}{'FDE média':>10}")
    for nome, r in tabela.items():
        if r['n'] == 0:
            print(f'{nome:<22}{0:>6}   (sem amostras na validação)')
            continue
        print(f"{nome:<22}{r['n']:>6}   {r['rede']['ade']:>9.3f}{r['rede']['fde']:>10.3f}"
              f"   {r['media_por_comando']['ade']:>10.3f}{r['media_por_comando']['fde']:>10.3f}")


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='real_trajectory_dataset.npz')
    parser.add_argument('--out-dir', default='figures/baseline_regressao')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3) ##taxa de aprendizado
    parser.add_argument('--val-routes', type=int, nargs='*', default=None,
                        help='rotas usadas na validação, ex: 8 9. Padrão: ~20%% finais.')
    parser.add_argument('--seed', type=int, default=3)
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

    # ---- dados ----
    bev, tau, command, route_id, t_arr = load_dataset(dataset_path)
    N, H, _ = tau.shape
    train_idx, val_idx, val_routes = split_indices(N, route_id, args.val_routes)
    print(f'Dataset: {N} amostras, H={H}, BEV {tuple(bev.shape[1:])}, dispositivo={device}')
    if val_routes is None:
        print('(aviso: .npz sem route_id; usando corte em bloco. Rode de novo '
              'figuras_trajetorias_futuras.py para a divisão por rota.)')
    else:
        print(f'Rotas de validação: {val_routes}')
    for nome, idx in (('treino', train_idx), ('validação', val_idx)):
        valores, contagens = np.unique(command[idx], return_counts=True)
        resumo = ', '.join(f'{nome_cmd(v)}: {c}' for v, c in zip(valores.tolist(), contagens.tolist()))
        print(f'  {nome}: {len(idx)} amostras -> {resumo}')

    cmd_idx = torch.from_numpy(commands_to_idx(command))
    bev_u8 = bev_to_uint8(bev)
    bev_train, cmd_train = bev_u8[train_idx], cmd_idx[train_idx]
    bev_val, cmd_val = bev_u8[val_idx], cmd_idx[val_idx]
    tau_train, tau_val = tau[train_idx], tau[val_idx]
    command_train, command_val = command[train_idx], command[val_idx]

    # Normaliza o alvo por um fator único (desvio-padrão do treino): deixa a
    # perda em torno de 1 em vez de dezenas de m². Todas as métricas abaixo
    # são calculadas de volta em METROS.
    y_scale = float(tau_train.std())
    Y_train = torch.from_numpy((tau_train.reshape(len(tau_train), -1) / y_scale).astype(np.float32))

    batch_size = min(args.batch_size, len(train_idx))
    loader = DataLoader(TensorDataset(bev_train, cmd_train, Y_train),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- modelo ----
    model = BEVTrajectoryRegressor(H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Modelo: {n_params:,} parâmetros | y_scale={y_scale:.3f} m\n')

    # ---- treino ----
    history = {'train_mse': [], 'val_mse': [], 'val_ade': [], 'val_fde': []}
    best_ade, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        soma, n_lotes = 0.0, 0
        for xb, cb, yb in loader:
            xb = xb.to(device).float() / 255.0
            cb, yb = cb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            opt.step()
            soma += loss.item()
            n_lotes += 1
        train_mse = soma / n_lotes * y_scale ** 2          # volta para m²

        pred_val = predict(model, bev_val, cmd_val, y_scale, H, device)
        val_mse = float(np.mean((pred_val - tau_val) ** 2))
        ade, fde = ade_fde(pred_val, tau_val)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['val_ade'].append(ade)
        history['val_fde'].append(fde)
        print(f'época {epoch:>3d}/{args.epochs} | MSE treino {train_mse:8.4f} | '
              f'MSE val {val_mse:8.4f} | ADE val {ade:6.3f} m | FDE val {fde:6.3f} m')

        if ade < best_ade:
            best_ade, best_epoch = ade, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- avaliação final com o melhor modelo (menor ADE de validação) ----
    model.load_state_dict(best_state)
    pred_val = predict(model, bev_val, cmd_val, y_scale, H, device)
    ade, fde = ade_fde(pred_val, tau_val)
    base_pred = mean_trajectory_baseline(tau_train, command_train, command_val)
    base_ade, base_fde = ade_fde(base_pred, tau_val)
    print(f'\nMelhor época: {best_epoch}')
    print(f'Rede              -> ADE {ade:.3f} m | FDE {fde:.3f} m')
    print(f'Trajetória média  -> ADE {base_ade:.3f} m | FDE {base_fde:.3f} m   (baseline sem imagem)')

    # ---- Atividade 3: erro por comando ----
    cmds_val = sorted(np.unique(command_val).tolist())
    por_comando = build_table(pred_val, tau_val, command_val, base_pred,
                              {nome_cmd(c): [c] for c in cmds_val})
    por_grupo = build_table(pred_val, tau_val, command_val, base_pred, GRUPOS_NOTEBOOK)
    print_table('Erro por comando (validação):', por_comando)
    print_table('Erro por grupo do enunciado (esquerda / frente / direita):', por_grupo)

    # ---- Atividade 2: salvar curvas e exemplos ----
    plot_curves(history, out_dir / 'curvas_de_perda.png')
    plot_group_bars(por_comando, out_dir / 'erro_por_comando.png')

    exemplos = pick_examples(command_val, rng)
    imagens, gts, preds, titulos = [], [], [], []
    for i in exemplos:
        imagens.append(np.transpose(bev[val_idx[i]], (1, 2, 0)))
        gts.append(tau_val[i])
        preds.append(pred_val[i])
        origem = (f'route_{int(route_id[val_idx[i]]):02d}, t={int(t_arr[val_idx[i]]):04d}'
                  if route_id is not None and t_arr is not None else f'amostra {int(val_idx[i])}')
        a, _ = ade_fde(pred_val[i:i + 1], tau_val[i:i + 1])
        titulos.append(f'{nome_cmd(command_val[i])} | {origem} | ADE={a:.2f} m')
    plot_examples(imagens, gts, preds, titulos, out_dir / 'exemplos_predicao.png')

    # ---- salvar modelo e métricas ----
    torch.save({'state_dict': best_state, 'H': H, 'y_scale': y_scale,
                'command_to_idx': COMMAND_TO_IDX}, out_dir / 'baseline_regressao_bev.pth')
    metricas = {
        'melhor_epoca': best_epoch,
        'val_routes': val_routes,
        'y_scale': y_scale,
        'rede': {'ade': ade, 'fde': fde},
        'trajetoria_media_por_comando': {'ade': base_ade, 'fde': base_fde},
        'por_comando': por_comando,
        'por_grupo': por_grupo,
        'historico': history,
    }
    with open(out_dir / 'metricas.json', 'w') as f:
        json.dump(metricas, f, indent=2)
    print(f'\nArquivos salvos em {out_dir}/: curvas_de_perda.png, exemplos_predicao.png, '
          f'erro_por_comando.png, metricas.json, baseline_regressao_bev.pth')


if __name__ == '__main__':
    main()
