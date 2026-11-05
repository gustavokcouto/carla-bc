"""
Monta o dataset real de trajetórias futuras a partir de gail_experts/ e
cumpre os 3 itens da atividade do notebook 02_dataset_extracao_trajetorias:

  1. build_dataset_from_episode adaptada ao formato real (dataset_utils.py).
  2. Conta e imprime quantas amostras foram descartadas por horizonte futuro
     insuficiente.
  3. Gera 20 figuras aleatórias de trajetórias futuras em
     figures/dataset_check/.

Uso (modo automático: monta o dataset inteiro, sorteia 20 figuras):
    python figuras_trajetorias_futuras.py
    python figuras_trajetorias_futuras.py --routes 0 1 2 --source birdview --out dataset.npz

Uso (NOVO, modo manual: você escolhe qual imagem BEV quer ver; não monta o
dataset inteiro, então é rápido). Figuras em figures/dataset_check_manual/:
    # passando o caminho de uma ou mais imagens
    python figuras_trajetorias_futuras.py --manual-image gail_experts/route_03/ep_00/birdview_masks/0150_00.png
    # modo interativo: digite um caminho OU "rota t" (ex.: 3 150); ENTER vazio sai
    python figuras_trajetorias_futuras.py --manual
"""
## python figuras_trajetorias_futuras.py --manual-image gail_experts/route_03/ep_00/birdview_masks/0150_00.png -EXEMPLO DE CHAMADA MANUAL

# build real dataset
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # salva em arquivo, não abre janela
import matplotlib.pyplot as plt

from extracao_trajetorias import build_dataset_from_routes, load_episode, extract_future_trajectory


# NOVO: geometria do BEV. Estes valores PRECISAM bater com
# obs_configs['hero']['birdview'] em data_collect.py (width_in_pixels,
# pixels_ev_to_bottom e pixels_per_meter). O carro fica centralizado na
# largura, a 40 px da borda de baixo, olhando para o topo da imagem.
BEV_WIDTH_PX = 192
BEV_EV_TO_BOTTOM_PX = 40
BEV_PIXELS_PER_METER = 5.0
# Tempo simulado de cada passo (o ambiente roda em modo síncrono, 10 passos/s).
# Só usado para converter "passos à frente" em segundos nos textos da figura.
STEP_SECONDS = 0.1


def local_to_bev_pixels(local_xy, img_w, img_h):
    """
    Converte pontos no referencial local do carro (x = frente, y = direita,
    em metros) para coordenadas de pixel (coluna, linha) da imagem do BEV,
    usando a mesma transformação de chauffeurnet.py (_get_warp_transform):
        coluna = (W-1)/W * (y*ppm + W/2)
        linha  = (W-1) - (W-1)/W * (x*ppm + pixels_ev_to_bottom)
    (W = 192, a largura original do BEV). Se a imagem exibida foi
    redimensionada (ex.: --img-size 96), escala junto, respeitando o centro dos
    pixels, como o cv2.resize faz.
    """
    local_xy = np.asarray(local_xy, dtype=float)
    W = BEV_WIDTH_PX
    f_px = local_xy[:, 0] * BEV_PIXELS_PER_METER   # deslocamento à frente, em pixels
    r_px = local_xy[:, 1] * BEV_PIXELS_PER_METER   # deslocamento à direita, em pixels
    col = (W - 1) / W * (r_px + W / 2)
    row = (W - 1) - (W - 1) / W * (f_px + BEV_EV_TO_BOTTOM_PX)
    col = (col + 0.5) * (img_w / W) - 0.5
    row = (row + 0.5) * (img_h / W) - 0.5
    return col, row


def plot_local_trajectory(sample, save_path, stride=1):
    tau = sample['future_trajectory']
    n_steps = len(tau) * stride   # o último ponto está a t + n_steps passos

    # NOVO: dois painéis lado a lado — a imagem real do birdview que gerou
    # essa amostra (painel esquerdo) e o gráfico da trajetória futura
    # (painel direito), para dar pra conferir visualmente que os dois
    # batem (ex.: se o birdview mostra uma curva à esquerda, a trajetória
    # também deveria curvar nesse sentido).
    fig, (ax_img, ax_traj) = plt.subplots(1, 2, figsize=(8, 4.5))

    bev_img = np.transpose(sample['bev'], (1, 2, 0))  # (C,H,W) -> (H,W,C), já em [0,1]
    ax_img.imshow(bev_img)
    ax_img.set_title(f"route_{sample['route_id']:02d}, t={sample['t']:04d}"
                      if isinstance(sample['route_id'], int)
                      else f"route_{sample['route_id']}, t={sample['t']:04d}")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # NOVO: desenha, EM CIMA da imagem do BEV, os mesmos pontos que o gráfico
    # da direita mostra (a trajetória futura), com o ÚLTIMO ponto destacado:
    # é até onde a trajetória prevista vai. Só faz sentido para o BEV (vista de
    # cima, mesma geometria do cálculo); em imagens de câmera (perspectiva) a
    # conversão de metros para pixels seria outra, então não desenhamos nada.
    img_h, img_w = bev_img.shape[:2]
    legenda_img = Path(sample['image_path']).name
    eh_bev = Path(sample['image_path']).parent.name == 'birdview_masks'
    x_fim, y_fim = float(tau[-1, 0]), float(tau[-1, 1])
    if eh_bev:
        col, row = local_to_bev_pixels(tau, img_w, img_h)
        ev_col, ev_row = local_to_bev_pixels(np.zeros((1, 2)), img_w, img_h)
        ax_img.plot(col, row, color='cyan', marker='o', markersize=3, linewidth=1.5,
                    label=f'trajetória futura ({len(tau)} pontos)')
        ax_img.scatter(col[-1], row[-1], marker='*', s=150, color='white',
                       edgecolors='black', zorder=5, label=f'último ponto (t+{n_steps})')
        ax_img.scatter(ev_col, ev_row, marker='s', s=45, color='white',
                       edgecolors='black', zorder=5, label='veículo em t')
        # a imagem tem limites fixos: pontos além deles ficam cortados
        ax_img.set_xlim(-0.5, img_w - 0.5)
        ax_img.set_ylim(img_h - 0.5, -0.5)
        ax_img.legend(loc='lower left', fontsize=6, framealpha=0.7)
        fora = int(((col < -0.5) | (col > img_w - 0.5) | (row < -0.5) | (row > img_h - 0.5)).sum())
        lado = 'direita' if y_fim > 0 else 'esquerda'
        legenda_img += (f"\núltimo ponto: t+{n_steps} passos (≈{n_steps * STEP_SECONDS:.1f} s)"
                        f"\n{x_fim:.1f} m à frente, {abs(y_fim):.1f} m à {lado}")
        if fora:
            legenda_img += f"\n({fora} ponto(s) fora da imagem)"
    else:
        legenda_img += '\n(imagem de câmera: sem sobreposição)'
    ax_img.set_xlabel(legenda_img, fontsize=8)

    ax_traj.plot(tau[:, 0], tau[:, 1], marker='o')
    ax_traj.scatter([0], [0], marker='s', s=80, label='veículo em t')
    ax_traj.scatter([x_fim], [y_fim], marker='*', s=150, color='white', edgecolors='black',
                    zorder=5, label=f'último ponto (t+{n_steps})')
    ax_traj.axhline(0, linestyle='--', linewidth=1)
    ax_traj.axvline(0, linestyle='--', linewidth=1)
    ax_traj.axis('equal')
    ax_traj.set_title(f"comando={sample['command']}")
    ax_traj.set_xlabel('x local (frente, m)')
    # NOVO: orientação do eixo Y. No referencial do CARLA (usado em
    # global_to_local), o eixo lateral aponta para a DIREITA do carro:
    #   y > 0 -> direita      y < 0 -> esquerda
    # (e x > 0 -> à frente do carro).
    ax_traj.set_ylabel('y local (lateral, m)\n← esquerda (−)    direita (+) →',
                       fontsize=9)
    ax_traj.text(0.08, 0.98, 'y positivo = direita', transform=ax_traj.transAxes,
                 ha='left', va='top', fontsize=8, color='gray')
    ax_traj.text(0.08, 0.02, 'y negativo = esquerda', transform=ax_traj.transAxes,
                 ha='left', va='bottom', fontsize=8, color='gray')
    ax_traj.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NOVO: modo manual — você escolhe qual imagem BEV (instante t) quer ver
# ---------------------------------------------------------------------------
IMAGE_FOLDERS = ('birdview_masks', 'central_rgb', 'left_rgb', 'right_rgb')


def parse_manual_entry(text, args):
    """
    Aceita dois formatos de entrada manual:
      1) caminho de uma imagem:
         gail_experts/route_03/ep_00/birdview_masks/0150_00.png
      2) "rota t" (atalho), ex.: 3 150
         -> monta o caminho da imagem da rota 3, instante 150, usando --source
            (birdview por padrão) e --ep (episódio 0 por padrão).
    Devolve o caminho (Path) da imagem.
    """
    text = text.strip().strip('"').strip("'")
    partes = text.split()
    if len(partes) == 2 and all(p.isdigit() for p in partes):
        route, t = int(partes[0]), int(partes[1])
        episode_dir = Path(args.gail_experts_dir) / f'route_{route:02d}' / f'ep_{args.ep:02d}'
        if args.source == 'birdview':
            return episode_dir / 'birdview_masks' / f'{t:04d}_00.png'
        return episode_dir / args.source / f'{t:04d}.png'
    return Path(text)


def build_manual_sample(image_path, H, stride):
    """
    A partir do caminho de UMA imagem (.../route_XX/ep_YY/<pasta>/NNNN[_MM].png),
    descobre a rota e o instante t, lê o episode.json da mesma pasta e monta um
    'sample' com a trajetória futura daquele instante, no mesmo formato que
    plot_local_trajectory já usa. Levanta ValueError (com mensagem clara) se
    algo não bater.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f'imagem não encontrada: {image_path}')

    episode_dir = image_path.parent.parent  # .../route_XX/ep_YY
    if image_path.parent.name not in IMAGE_FOLDERS or not (episode_dir / 'episode.json').exists():
        raise ValueError(
            f'{image_path} não está no formato esperado '
            f'(gail_experts/route_XX/ep_YY/{{{",".join(IMAGE_FOLDERS)}}}/NNNN.png, '
            'com episode.json na pasta ep_YY)')

    try:
        t = int(image_path.stem.split('_')[0])
    except ValueError:
        raise ValueError(f'não consegui ler o instante t do nome {image_path.name} '
                         '(esperado algo como 0150_00.png ou 0150.png)')

    route_name = episode_dir.parent.name
    try:
        route_id = int(route_name.split('_')[-1])
    except ValueError:
        route_id = route_name

    ep = load_episode(episode_dir)  # ValueError claro se faltar pose/command
    poses = ep['pose']
    if t >= len(poses):
        raise ValueError(f't={t} fora da rota (a rota tem {len(poses)} passos, t vai de 0 a {len(poses) - 1})')

    tau = extract_future_trajectory(poses, t, H=H, stride=stride)
    if tau is None:
        ultimo_t = len(poses) - 1 - H * stride
        raise ValueError(
            f't={t} não tem {H} passos futuros (stride={stride}) nesta rota de {len(poses)} passos. '
            f'O último t com trajetória completa é {ultimo_t}.')

    img = plt.imread(image_path)
    if img.ndim == 2:  # imagem em tons de cinza
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]  # descarta canal alfa, se houver

    return {
        'bev': np.transpose(img, (2, 0, 1)),  # (H,W,C) -> (C,H,W), igual aos samples do dataset
        'command': int(np.asarray(ep['command'][t]).reshape(-1)[0]),
        'future_trajectory': tau.astype(np.float32),
        'route_id': route_id,
        't': t,
        'image_path': str(image_path),
    }


def save_manual_figure(entry_text, args, out_dir):
    """Processa UMA entrada manual: monta a amostra, desenha e salva a figura."""
    try:
        image_path = parse_manual_entry(entry_text, args)
        sample = build_manual_sample(image_path, args.H, args.stride)
    except ValueError as e:
        print(f'  ✗ {e}')
        return None

    image_path = Path(sample['image_path'])
    rid = sample['route_id']
    rota = f'route{rid:02d}' if isinstance(rid, int) else f'route{rid}'
    ep_name = image_path.parent.parent.name.replace('_', '')  # ep_00 -> ep00
    save_path = out_dir / f"manual_{rota}_{ep_name}_t{sample['t']:04d}_{image_path.parent.name}.png"
    plot_local_trajectory(sample, save_path, stride=args.stride)
    print(f"  ✓ {rota}, t={sample['t']}, comando={sample['command']} -> {save_path}")
    return save_path


def run_manual_mode(args):
    out_dir = Path(args.manual_figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Modo manual: as figuras serão salvas em {out_dir}/ '
          f'(H={args.H}, stride={args.stride})')

    for entrada in args.manual_image or []:
        print(f'> {entrada}')
        save_manual_figure(entrada, args, out_dir)

    if args.manual:
        print('\nDigite o caminho de uma imagem BEV, ou "rota t" (ex.: 3 150).')
        print('ENTER vazio (ou "q") encerra.')
        while True:
            try:
                entrada = input('BEV> ')
            except EOFError:
                break
            if not entrada.strip() or entrada.strip().lower() in ('q', 'sair'):
                break
            save_manual_figure(entrada, args, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gail-experts-dir', default='gail_experts')
    parser.add_argument('--routes', type=int, nargs='*', default=None,
                         help='ids das rotas a usar, ex: 0 1 2. Padrão: todas as que existirem.')
    parser.add_argument('--source', default='birdview',
                         choices=['birdview', 'central_rgb', 'left_rgb', 'right_rgb'])
    parser.add_argument('--img-size', type=int, default=192)
    parser.add_argument('--H', type=int, default=20)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--out', default='real_trajectory_dataset.npz')
    parser.add_argument('--n-figures', type=int, default=20)
    parser.add_argument('--figures-dir', default='figures/dataset_check')
    parser.add_argument('--seed', type=int, default=1)
    # NOVO: modo manual (ver docstring do arquivo)
    parser.add_argument('--manual-image', nargs='+', default=None, metavar='CAMINHO',
                        help='caminho(s) de imagem(ns) BEV a visualizar, ex.: '
                             'gail_experts/route_03/ep_00/birdview_masks/0150_00.png')
    parser.add_argument('--manual', action='store_true',
                        help='modo interativo: pergunta no terminal qual imagem '
                             '(caminho ou "rota t") você quer visualizar')
    parser.add_argument('--ep', type=int, default=0,
                        help='episódio usado quando a entrada manual é "rota t" (padrão: 0)')
    parser.add_argument('--manual-figures-dir', default='figures/dataset_check_manual')
    args = parser.parse_args()

    # NOVO: no modo manual só desenhamos as imagens pedidas — não montamos o
    # dataset inteiro (que é lento e usa bastante memória).
    if args.manual or args.manual_image:
        run_manual_mode(args)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    samples, discarded = build_dataset_from_routes(
        args.gail_experts_dir,
        route_ids=args.routes,
        source=args.source,
        img_size=args.img_size,
        H=args.H,
        stride=args.stride,
    )

    print()
    print(f'Amostras válidas criadas: {len(samples)}')
    print(f'Amostras descartadas por horizonte insuficiente ou imagem ausente: {discarded}')

    if len(samples) == 0:
        print('Nenhuma amostra válida: confira se gail_experts/ existe e se o '
              'episode.json já tem as colunas pose/command (veja data_collect.py).')
        return

    # --- item 3: 20 figuras aleatórias ---
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_figures = min(args.n_figures, len(samples))
    chosen = random.sample(range(len(samples)), n_figures)
    for i, idx in enumerate(chosen):
        plot_local_trajectory(samples[idx], figures_dir / f'trajectory_{i:02d}_sample{idx:06d}.png',
                              stride=args.stride)
    print(f'{n_figures} figuras salvas em {figures_dir}/')

    # --- salvar o dataset consolidado ---
    bev_array = np.stack([s['bev'] for s in samples])
    command_array = np.array([s['command'] for s in samples])
    pose_array = np.stack([s['pose'] for s in samples])
    action_array = np.stack([s['action'] for s in samples])
    tau_array = np.stack([s['future_trajectory'] for s in samples])
    # NOVO: de qual rota e de qual instante t veio cada amostra. Usado pelo
    # baseline de regressão (04) para separar treino/validação POR ROTA.
    route_id_array = np.array([s['route_id'] for s in samples])
    t_array = np.array([s['t'] for s in samples])

    np.savez(
        args.out,
        bev=bev_array,
        command=command_array,
        pose=pose_array,
        action=action_array,
        tau=tau_array,
        route_id=route_id_array,
        t=t_array,
    )
    print(f'Dataset salvo em {args.out} '
          f'(bev {bev_array.shape}, tau {tau_array.shape})')


if __name__ == '__main__':
    main()