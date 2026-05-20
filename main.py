"""
SPECTRABREAST — PUNTO DI INGRESSO
==================================
Legge config.yaml e lancia la pipeline nella modalita' indicata da `mode`.

Modalita' supportate:
  "single" : pipeline normale (HSI con ArUco visibili) su una coppia mesh+aruco
  "roi"    : HSI duale: PNG LiveView con ArUco + cubo ROI senza ArUco
             (singolo sample)
  "batch"  : scandisce input_dir e processa piu' coppie sample (single o ROI
             a seconda di batch.roi_mode)
  "sweep"  : esegue la pipeline su piu' coppie di risoluzioni (single o ROI
             a seconda di sweep.roi_mode)

Utilizzo:
    python3 main.py                          # usa config.yaml di default
    python3 main.py --config altro.yaml      # config alternativo
    python3 main.py --sample SB020           # override sample_name
    python3 main.py --resolution 1.0         # override render resolution
    python3 main.py --mode batch             # override modalita'
    python3 main.py --cpu                    # forza render su CPU
    python3 main.py --device cuda:1          # device torch specifico
    python3 main.py --dry-run                # valida senza eseguire
"""

import argparse
import os
import sys
import time

import cv2
import yaml

from sweep import run_sweep
from batch import run_batch
from spectrabreast.pipeline import (
    load_mesh,
    save_render,
    save_turbo_render,
)
from spectrabreast.pipeline_roi import run_full_pipeline_roi

# Render PyTorch (GPU se disponibile, altrimenti CPU torch)
render_orthographic_topview_gpu = None

try:
    import torch
    from spectrabreast.render_gpu import render_orthographic_topview_gpu
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE  = torch.cuda.is_available()
except Exception as e:
    print(f"[Init] ERRORE import torch/render_gpu: {type(e).__name__}: {e}")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE  = False


# =============================================================================
# Caricamento e validazione config
# =============================================================================

VALID_MODES = ("single", "roi", "batch", "sweep")


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"[Config] ERRORE: file non trovato -> {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict, base_dir: str) -> dict:
    """Converte i path relativi in assoluti rispetto alla root del progetto."""
    p = cfg['paths']
    for key in ('hsi_hdr', 'mesh', 'aruco_json', 'output_dir', 'liveview_png'):
        if key in p and p[key] and not os.path.isabs(p[key]):
            p[key] = os.path.join(base_dir, p[key])
    return cfg


def validate_mode(cfg: dict) -> str:
    """Legge e valida cfg['mode']. Ritorna la modalita' normalizzata."""
    mode = str(cfg.get('mode', 'single')).strip().lower()
    if mode not in VALID_MODES:
        print(f"[Config] ERRORE: mode='{mode}' non valido. "
              f"Valori ammessi: {VALID_MODES}")
        sys.exit(1)
    return mode


def _batch_uses_roi(cfg: dict) -> bool:
    """In batch e sweep la modalita' ROI e' attivata da un flag dedicato."""
    return bool(cfg.get('batch', {}).get('roi_mode', False))


def _sweep_uses_roi(cfg: dict) -> bool:
    return bool(cfg.get('sweep', {}).get('roi_mode', False))


def validate_inputs(cfg: dict, mode: str) -> bool:
    """
    Controlla che i file di input richiesti dalla modalita' esistano.
    In batch, mesh/aruco/liveview vengono scoperti dinamicamente -> non si validano qui.
    """
    p  = cfg['paths']
    ok = True

    checks = [('hsi_hdr', 'HSI .hdr')]

    if mode == 'single':
        checks += [('mesh', 'Mesh'), ('aruco_json', 'ArUco JSON')]
    elif mode == 'roi':
        checks += [('mesh', 'Mesh'),
                   ('aruco_json', 'ArUco JSON'),
                   ('liveview_png', 'LiveView PNG')]
    elif mode == 'sweep':
        checks += [('mesh', 'Mesh'), ('aruco_json', 'ArUco JSON')]
        if _sweep_uses_roi(cfg):
            checks += [('liveview_png', 'LiveView PNG (sweep ROI)')]
    # batch: solo hsi_hdr (e' il default; il resto e' scoperto automaticamente)

    for key, label in checks:
        path = p.get(key)
        if not path:
            print(f"[Validate] MANCANTE — {label}: chiave 'paths.{key}' non definita")
            ok = False
        elif not os.path.exists(path):
            print(f"[Validate] MANCANTE — {label}: {path}")
            ok = False
        else:
            print(f"[Validate] OK        — {label}: {path}")
    return ok


# =============================================================================
# Parsing argomenti CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SpectraBreast — Pipeline HSI + 3D Mesh",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--config', default='config.yaml',
                        help="File di configurazione YAML (default: config.yaml)")
    parser.add_argument('--sample', default=None,
                        help="Sovrascrive sample_name dal config")
    parser.add_argument('--resolution', type=float, default=None,
                        help="Sovrascrive render.resolution_mm_per_px")
    parser.add_argument('--mode', default=None, choices=VALID_MODES,
                        help="Sovrascrive mode dal config")
    parser.add_argument('--cpu', action='store_true',
                        help="Forza il render su CPU")
    parser.add_argument('--device', default=None,
                        help="Device torch (es. 'cuda:0', 'cpu')")
    parser.add_argument('--dry-run', action='store_true',
                        help="Valida config e input senza eseguire")
    return parser.parse_args()


# =============================================================================
# Mappatura dizionario ArUco
# =============================================================================

ARUCO_DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


# =============================================================================
# Stampa riepilogo minimale
# =============================================================================

def print_summary(cfg: dict, mode: str) -> None:
    """
    Stampa SOLO: modalita', paths, marker size, risoluzione attiva, save flags.
    """
    save_pc  = cfg['export'].get('save_pointcloud', False)
    save_img = cfg['export'].get('save_images', False)

    print("\n" + "=" * 68)
    print("SPECTRABREAST — RIEPILOGO")
    print("=" * 68)
    print(f"  Modalita'     : {mode.upper()}")
    print(f"  Sample        : {cfg['sample_name']}")
    print(f"  HSI           : {cfg['paths']['hsi_hdr']}")
    if mode != 'batch':
        print(f"  Mesh          : {cfg['paths']['mesh']}")
        print(f"  ArUco JSON    : {cfg['paths']['aruco_json']}")
        if mode == 'roi' or (mode == 'sweep' and _sweep_uses_roi(cfg)):
            print(f"  LiveView PNG  : {cfg['paths'].get('liveview_png', 'N/A')}")
    else:
        print(f"  Input dir     : {cfg.get('batch', {}).get('input_dir', 'N/A')}")
        print(f"  ROI mode      : {'SI' if _batch_uses_roi(cfg) else 'NO'}")
    print(f"  Marker side   : {cfg['registration']['marker_side_mm']} mm")

    if mode == 'sweep':
        pairs = cfg.get('sweep', {}).get('resolution_pairs', []) or []
        print(f"  Render        : sweep su {len(pairs)} coppie [reg, pc]")
        print(f"  ROI mode      : {'SI' if _sweep_uses_roi(cfg) else 'NO'}")
    else:
        res_pc  = cfg['render']['resolution_mm_per_px']
        res_reg = cfg['render'].get('resolution_reg_mm_per_px', res_pc)
        print(f"  Render        : res_reg={res_reg} mm/px   res_pc={res_pc} mm/px")

    print(f"  Save PC       : {'SI' if save_pc else 'NO'}")
    print(f"  Save immagini : {'SI' if save_img else 'NO'}")
    print("=" * 68)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = load_config(os.path.join(base_dir, args.config))
    cfg = resolve_paths(cfg, base_dir)

    if args.sample:
        cfg['sample_name'] = args.sample
    if args.resolution:
        cfg['render']['resolution_mm_per_px'] = args.resolution
    if args.mode:
        cfg['mode'] = args.mode

    mode = validate_mode(cfg)

    # Device torch
    if not TORCH_AVAILABLE:
        torch_device = None
    elif args.cpu:
        torch_device = 'cpu'
    elif args.device is not None:
        torch_device = args.device
    elif CUDA_AVAILABLE:
        torch_device = 'cuda'
    else:
        torch_device = 'cpu'
    use_torch_render = TORCH_AVAILABLE

    print_summary(cfg, mode)

    print("\n[Validate] Controllo file di input...")
    if not validate_inputs(cfg, mode):
        print("\n[Validate] ERRORE: uno o piu' file mancanti.")
        sys.exit(1)
    print("[Validate] Tutti i file di input trovati.\n")

    if args.dry_run:
        print("[Dry-run] Configurazione valida. Pipeline NON eseguita.")
        sys.exit(0)

    aruco_key  = cfg['registration'].get('aruco_dict', '4X4_50')
    aruco_dict = ARUCO_DICT_MAP.get(aruco_key, cv2.aruco.DICT_4X4_50)

    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if mode == 'batch':
        _render_fn = render_orthographic_topview_gpu if use_torch_render else None
        run_batch(
            cfg              = cfg,
            aruco_dict_cv    = aruco_dict,
            torch_device     = torch_device,
            use_torch_render = use_torch_render,
            base_dir         = base_dir,
            render_fn        = _render_fn,
        )
        sys.exit(0)

    if mode == 'sweep':
        _render_fn = render_orthographic_topview_gpu if use_torch_render else None
        run_sweep(
            cfg              = cfg,
            aruco_dict_cv    = aruco_dict,
            torch_device     = torch_device,
            use_torch_render = use_torch_render,
            render_fn        = _render_fn,
        )
        sys.exit(0)

    # ── mode == "single" o "roi" ─────────────────────────────────────────────
    save_pointcloud = cfg['export'].get('save_pointcloud', True)
    save_images     = cfg['export'].get('save_images', True)

    precomputed_render     = None
    precomputed_render_reg = None

    if use_torch_render:
        res_pc  = cfg['render']['resolution_mm_per_px']
        res_reg = cfg['render'].get('resolution_reg_mm_per_px', res_pc)
        dual    = abs(res_reg - res_pc) > 1e-6

        mesh = load_mesh(cfg['paths']['mesh'], scale_m_to_mm=True)

        print(f"\n[Render PC] {res_pc} mm/px su {torch_device}...")
        t0r = time.time()
        precomputed_render = render_orthographic_topview_gpu(
            mesh,
            resolution_mm_per_px = res_pc,
            margin_mm            = cfg['render']['margin_mm'],
            device               = torch_device,
        )
        print(f"[Render PC] Completato in {time.time()-t0r:.1f}s")
        if save_images:
            r_rgb, d_map, xyz_map, _, _ = precomputed_render
            save_render(r_rgb, d_map,
                        output_dir = cfg['paths']['output_dir'],
                        prefix     = f"{cfg['sample_name']}_render_pc")
            save_turbo_render(xyz_map,
                              os.path.join(cfg['paths']['output_dir'],
                                           f"{cfg['sample_name']}_render_pc_turbo.png"))

        if dual:
            print(f"\n[Render REG] {res_reg} mm/px su {torch_device}...")
            t0r = time.time()
            precomputed_render_reg = render_orthographic_topview_gpu(
                mesh,
                resolution_mm_per_px = res_reg,
                margin_mm            = cfg['render']['margin_mm'],
                device               = torch_device,
            )
            print(f"[Render REG] Completato in {time.time()-t0r:.1f}s")
            if save_images:
                r_rgb, d_map, xyz_map, _, _ = precomputed_render_reg
                save_render(r_rgb, d_map,
                            output_dir = cfg['paths']['output_dir'],
                            prefix     = f"{cfg['sample_name']}_render_reg")
                save_turbo_render(xyz_map,
                                  os.path.join(cfg['paths']['output_dir'],
                                               f"{cfg['sample_name']}_render_reg_turbo.png"))
        else:
            precomputed_render_reg = precomputed_render

    # ── Parametri specifici ROI ──────────────────────────────────────────────
    liveview_png_path = None
    roi_align_cfg     = None
    if mode == 'roi':
        liveview_png_path = cfg['paths'].get('liveview_png')
        roi_align_cfg     = cfg.get('roi', {}) or None

    # ── Pipeline ─────────────────────────────────────────────────────────────
    t0 = time.time()
    result = run_full_pipeline_roi(
        hsi_hdr_path             = cfg['paths']['hsi_hdr'],
        mesh_path                = cfg['paths']['mesh'],
        aruco_json_path          = cfg['paths']['aruco_json'],
        output_dir               = cfg['paths']['output_dir'],
        liveview_png_path        = liveview_png_path,
        roi_align_cfg            = roi_align_cfg,
        hsi_extraction_method    = cfg['registration']['hsi_extraction_method'],
        aruco_dict_type          = aruco_dict,
        suspicious_pixels_hsi    = None,
        render_resolution_mm     = cfg['render']['resolution_mm_per_px'],
        render_resolution_reg_mm = cfg['render'].get('resolution_reg_mm_per_px', None),
        render_margin_mm         = cfg['render']['margin_mm'],
        marker_side_mm           = cfg['registration']['marker_side_mm'],
        use_subpix               = cfg['registration'].get('use_subpix', True),
        subpix_winsize           = cfg['registration'].get('subpix_winsize', 5),
        border_px                = cfg['pointcloud']['border_px'],
        reflectance_norm         = cfg['pointcloud']['reflectance_norm'],
        pc_chunk_size            = cfg['pointcloud'].get('pc_chunk_size', 100_000),
        export_ply_file          = cfg['export']['ply']  if save_pointcloud else False,
        export_npy_file          = cfg['export']['npy']  if save_pointcloud else False,
        export_csv_file          = cfg['export']['csv']  if save_pointcloud else False,
        csv_max_points           = cfg['export']['csv_max_points'],
        save_pointcloud          = save_pointcloud,
        save_images              = save_images,
        sample_name              = cfg['sample_name'],
        precomputed_render       = precomputed_render,
        precomputed_render_reg   = precomputed_render_reg,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 68)
    print("PIPELINE COMPLETATA")
    print("=" * 68)
    print(f"  Tempo totale   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    n_pts   = result.get('n_valid', 0)
    n_bands = result.get('bands', 0)
    if save_pointcloud:
        print(f"  Punti cloud    : {n_pts:,}")
        print(f"  Bande spettrali: {n_bands}")
    else:
        print("  Point cloud    : NON salvata")
    if result.get('wavelengths'):
        wl = result['wavelengths']
        print(f"  Lunghezze d'onda: {wl[0]:.1f} — {wl[-1]:.1f} nm")
    if result.get('is_roi_mode'):
        info = result['roi_align_info']
        print(f"  ROI->PNG match : {info['n_inliers']}/{info['n_good_matches']} "
              f"inliers ({100*info['inlier_ratio']:.1f}%), "
              f"reproj={info['reproj_error_mean_px']:.3f} px")
    print(f"  Output         : {cfg['paths']['output_dir']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
