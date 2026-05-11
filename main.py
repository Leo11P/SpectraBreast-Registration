"""
SPECTRABREAST — PUNTO DI INGRESSO
==================================
Legge config.yaml e lancia la pipeline completa.
 
Utilizzo:
    python3 main.py                          # usa config.yaml di default
    python3 main.py --config altro.yaml      # usa un file di config alternativo
    python3 main.py --sample SB020           # sovrascrive sample_name da CLI
    python3 main.py --resolution 1.0         # sovrascrive render resolution da CLI
    python3 main.py --cpu                    # forza il render su CPU
    python3 main.py --device cuda:1          # seleziona uno specifico device torch
    python3 main.py --dry-run                # valida config senza eseguire
"""
 
import argparse
import os
import sys
import time
 
import cv2
import numpy as np
import yaml


from sweep import run_sweep
from spectrabreast.pipeline import (
    run_full_pipeline,
    extract_suspicious_centroids,
    load_mesh,
    save_render,
    save_turbo_render,
)
 
# Render PyTorch (GPU se disponibile, altrimenti CPU torch)
try:
    import torch
    from spectrabreast.render_gpu import render_orthographic_topview_gpu
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE  = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE  = False
 
 
# =============================================================================
# Caricamento config
# =============================================================================
 
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"[Config] ERRORE: file non trovato -> {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Caricato: {config_path}")
    return cfg
 
 
def resolve_paths(cfg: dict, base_dir: str) -> dict:
    """Converte i path relativi in assoluti rispetto alla root del progetto."""
    p = cfg['paths']
    for key in ('hsi_hdr', 'mesh', 'aruco_json', 'output_dir'):
        p[key] = os.path.join(base_dir, p[key])
    return cfg
 
 
def validate_config(cfg: dict) -> bool:
    """Controlla che i file di input esistano. Ritorna True se tutto ok."""
    p = cfg['paths']
    ok = True
    for key, label in [('hsi_hdr', 'HSI .hdr'), ('mesh', 'Mesh'), ('aruco_json', 'ArUco JSON')]:
        if not os.path.exists(p[key]):
            print(f"[Validate] MANCANTE — {label}: {p[key]}")
            ok = False
        else:
            print(f"[Validate] OK        — {label}: {p[key]}")
    return ok
 
 
# =============================================================================
# Parsing argomenti CLI
# =============================================================================
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="SpectraBreast — Pipeline HSI + 3D Mesh (PyTorch)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config', default='config.yaml',
        help="Percorso al file di configurazione YAML (default: config.yaml)"
    )
    parser.add_argument(
        '--sample', default=None,
        help="Sovrascrive sample_name dal config"
    )
    parser.add_argument(
        '--resolution', type=float, default=None,
        help="Sovrascrive render.resolution_mm_per_px dal config"
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help="Forza il render torch su CPU (anche se CUDA è disponibile)"
    )
    parser.add_argument(
        '--device', default=None,
        help="Device torch esplicito (es. 'cuda:0', 'cpu'). "
             "Se non specificato: cuda se disponibile."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Valida config e input senza eseguire la pipeline"
    )
    return parser.parse_args()
 
 
# =============================================================================
# Mappatura dizionario ArUco
# =============================================================================
 
ARUCO_DICT_MAP = {
    "4X4_50":   cv2.aruco.DICT_4X4_50,
    "4X4_100":  cv2.aruco.DICT_4X4_100,
    "5X5_50":   cv2.aruco.DICT_5X5_50,
    "5X5_100":  cv2.aruco.DICT_5X5_100,
    "6X6_50":   cv2.aruco.DICT_6X6_50,
    "6X6_100":  cv2.aruco.DICT_6X6_100,
}
 
 
# =============================================================================
# Main
# =============================================================================
 
def main():
    args = parse_args()
 
    # Root del progetto = cartella dove si trova main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
 
    # Carica e risolvi config
    cfg = load_config(os.path.join(base_dir, args.config))
    cfg = resolve_paths(cfg, base_dir)
 
    # Override da CLI
    if args.sample:
        cfg['sample_name'] = args.sample
        print(f"[CLI] sample_name -> {args.sample}")
    if args.resolution:
        cfg['render']['resolution_mm_per_px'] = args.resolution
        print(f"[CLI] resolution  -> {args.resolution} mm/px")
 
    # Flag export (con fallback True per retrocompatibilità con config vecchi)
    save_pointcloud = cfg['export'].get('save_pointcloud', True)
    save_images     = cfg['export'].get('save_images', True)
 
    # Risoluzione device torch
    if not TORCH_AVAILABLE:
        torch_device = None
        device_label = "PyTorch non installato — fallback CPU trimesh"
    elif args.cpu:
        torch_device = 'cpu'
        device_label = "PyTorch CPU (forzato da --cpu)"
    elif args.device is not None:
        torch_device = args.device
        device_label = f"PyTorch {args.device}"
    elif CUDA_AVAILABLE:
        torch_device = 'cuda'
        device_label = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
    else:
        torch_device = 'cpu'
        device_label = "PyTorch CPU (CUDA non disponibile)"
 
    use_torch_render = TORCH_AVAILABLE
 
    # Stampa riepilogo config
    print("\n" + "=" * 68)
    print("SPECTRABREAST — RIEPILOGO CONFIGURAZIONE")
    print("=" * 68)
    print(f"  Campione      : {cfg['sample_name']}")
    print(f"  HSI           : {cfg['paths']['hsi_hdr']}")
    print(f"  Mesh          : {cfg['paths']['mesh']}")
    print(f"  ArUco JSON    : {cfg['paths']['aruco_json']}")
    print(f"  Output        : {cfg['paths']['output_dir']}")
    print(f"  Risoluzione   : {cfg['render']['resolution_mm_per_px']} mm/px")
    print(f"  Metodo HSI    : {cfg['registration']['hsi_extraction_method']}")
    print(f"  Marker side   : {cfg['registration']['marker_side_mm']} mm")
    print(f"  Render        : {device_label}")
    print(f"  Save PC       : {'SI' if save_pointcloud else 'NO'}")
    print(f"  Save immagini : {'SI' if save_images else 'NO'}")
    print("=" * 68)
 
    # Validazione
    print("\n[Validate] Controllo file di input...")
    if not validate_config(cfg):
        print("\n[Validate] ERRORE: uno o più file mancanti. Controlla config.yaml.")
        sys.exit(1)
    print("[Validate] Tutti i file di input trovati.\n")

 
    if args.dry_run:
        print("[Dry-run] Configurazione valida. Pipeline NON eseguita (--dry-run).")
        sys.exit(0)
    # ── BRANCH SWEEP ─────────────────────────────────────────────────────────
    # Se sweep.enabled e' true, esegue la pipeline su tutte le coppie definite
    # in config.yaml -> sweep.resolution_pairs e produce SOLO l'Excel riepilogo.
    if cfg.get('sweep', {}).get('enabled', False):
        print("\n[Main] Modalita' SWEEP attiva — ignoro le risoluzioni singole.")
        run_sweep(
            cfg              = cfg,
            aruco_dict_cv    = aruco_dict,
            torch_device     = torch_device,
            use_torch_render = use_torch_render,
        )
        sys.exit(0)
    # ─────────────────────────────────────────────────────────────────────────  
    # Dizionario ArUco
    aruco_key  = cfg['registration'].get('aruco_dict', '4X4_50')
    aruco_dict = ARUCO_DICT_MAP.get(aruco_key, cv2.aruco.DICT_4X4_50)
 
    # Crea output dir
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)


 
    # ── BRANCH SWEEP ─────────────────────────────────────────────────────────
    # Se sweep.enabled e' true, esegue la pipeline su tutte le coppie definite
    # in config.yaml -> sweep.resolution_pairs e produce SOLO l'Excel riepilogo.
    if cfg.get('sweep', {}).get('enabled', False):
        print("\n[Main] Modalita' SWEEP attiva — ignoro le risoluzioni singole.")
        run_sweep(
            cfg              = cfg,
            aruco_dict_cv    = aruco_dict,
            torch_device     = torch_device,
            use_torch_render = use_torch_render,
            render_fn        = render_orthographic_topview_gpu if use_torch_render else None,  # <-- AGGIUNGI
        )
        sys.exit(0)
    # ─────────────────────────────────────────────────────────────────────────
 
    # ── Render PyTorch GPU (pre-calcola entrambi i render prima della pipeline) ──
    # main.py calcola SEMPRE entrambi i render su GPU qui, in modo che
    # run_full_pipeline non debba mai fare rendering su CPU.
    # Se le due risoluzioni coincidono, viene fatto un solo render (condiviso).
    precomputed_render     = None
    precomputed_render_reg = None
 
    if use_torch_render:
        res_pc  = cfg['render']['resolution_mm_per_px']
        res_reg = cfg['render'].get('resolution_reg_mm_per_px', res_pc)
        dual    = abs(res_reg - res_pc) > 1e-6
 
        mesh = load_mesh(cfg['paths']['mesh'], scale_m_to_mm=True)
 
        # ── Render POINT CLOUD (risoluzione fine) ────────────────────────────
        print(f"\n[Render PC] Avvio ray casting GPU ({res_pc} mm/px) su {torch_device}...")
        t0r = time.time()
        render_rgb_pc, depth_map_pc, xyz_map_pc, origin_xy_pc, res_out_pc = \
            render_orthographic_topview_gpu(
                mesh,
                resolution_mm_per_px = res_pc,
                margin_mm            = cfg['render']['margin_mm'],
                device               = torch_device,
            )
        print(f"[Render PC] Completato in {time.time()-t0r:.1f}s")
        if save_images:
            save_render(render_rgb_pc, depth_map_pc,
                        output_dir = cfg['paths']['output_dir'],
                        prefix     = f"{cfg['sample_name']}_render_pc")
            save_turbo_render(xyz_map_pc,
                              os.path.join(cfg['paths']['output_dir'],
                                           f"{cfg['sample_name']}_render_pc_turbo.png"))
        precomputed_render = (render_rgb_pc, depth_map_pc, xyz_map_pc, origin_xy_pc, res_out_pc)
 
        # ── Render REGISTRAZIONE (risoluzione grossolana, se diversa) ────────
        if dual:
            print(f"\n[Render REG] Avvio ray casting GPU ({res_reg} mm/px) su {torch_device}...")
            t0r = time.time()
            render_rgb_reg, depth_map_reg, xyz_map_reg, origin_xy_reg, res_out_reg = \
                render_orthographic_topview_gpu(
                    mesh,
                    resolution_mm_per_px = res_reg,
                    margin_mm            = cfg['render']['margin_mm'],
                    device               = torch_device,
                )
            print(f"[Render REG] Completato in {time.time()-t0r:.1f}s")
            if save_images:
                save_render(render_rgb_reg, depth_map_reg,
                            output_dir = cfg['paths']['output_dir'],
                            prefix     = f"{cfg['sample_name']}_render_reg")
                save_turbo_render(xyz_map_reg,
                                  os.path.join(cfg['paths']['output_dir'],
                                               f"{cfg['sample_name']}_render_reg_turbo.png"))
            precomputed_render_reg = (render_rgb_reg, depth_map_reg,
                                      xyz_map_reg, origin_xy_reg, res_out_reg)
        else:
            # Stessa risoluzione: un solo render condiviso
            precomputed_render_reg = precomputed_render
            print("[Render] Risoluzione reg == pc — render condiviso.")
 
    # (Opzionale) suspicious pixels da segmentazione — None di default
    suspicious_pixels = None
    # Esempio:
    # mask = np.load("data/input/segmentation_mask.npy")
    # suspicious_pixels, _ = extract_suspicious_centroids(mask, min_area=5)
 
    # Avvio pipeline
    t0 = time.time()
 
    result = run_full_pipeline(
        hsi_hdr_path             = cfg['paths']['hsi_hdr'],
        mesh_path                = cfg['paths']['mesh'],
        aruco_json_path          = cfg['paths']['aruco_json'],
        output_dir               = cfg['paths']['output_dir'],
        hsi_extraction_method    = cfg['registration']['hsi_extraction_method'],
        aruco_dict_type          = aruco_dict,
        suspicious_pixels_hsi    = suspicious_pixels,
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
 
    # Riepilogo finale
    print("\n" + "=" * 68)
    print("PIPELINE COMPLETATA")
    print("=" * 68)
    print(f"  Tempo totale   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    # In modalità streaming xyz/spectra non sono in RAM: usa n_valid e bands
    n_pts   = result['n_valid'] if result.get('xyz')     is None else result['xyz'].shape[0]
    n_bands = result['bands']   if result.get('spectra') is None else result['spectra'].shape[1]
    if save_pointcloud:
        print(f"  Punti cloud    : {n_pts:,}")
        print(f"  Bande spettrali: {n_bands}")
    else:
        print("  Point cloud    : NON salvata (save_pointcloud: false)")
    if result['wavelengths']:
        wl = result['wavelengths']
        print(f"  Lunghezze d'onda: {wl[0]:.1f} — {wl[-1]:.1f} nm")
    print(f"  Output salvati in: {cfg['paths']['output_dir']}")
    print("=" * 68)
 
 
if __name__ == "__main__":
    main()
