"""
SPECTRABREAST — RESOLUTION SWEEP
=================================
Esegue la pipeline di registrazione su una lista di coppie
(resolution_reg_mm_per_px, resolution_pc_mm_per_px) lette da config.yaml
e produce UN SOLO file Excel riepilogativo nella output_dir.

Durante lo sweep:
  - save_pointcloud  -> forzato False
  - save_images      -> forzato False
<<<<<<< HEAD
  - Excel per-run    -> NON salvato (lo sopprimiamo passando un output_dir
                        diverso solo per le run e non emettendo l'xlsx)
  - SOLO l'Excel riepilogativo finale viene scritto.
=======
  - Excel per-run    -> run_full_pipeline lo scrive in una subdir temporanea
                        _sweep_tmp/run_N/ che viene rimossa al termine dello sweep
  - SOLO l'Excel riepilogativo finale viene scritto in output_dir.
>>>>>>> sweep

Excel riepilogo (1 riga per coppia):
  res_reg_mm_pix | res_pc_mm_pix
  2D_mean_px | 2D_median_px | 2D_mean_mm | 2D_median_mm
  3D_REG_bilinear_mean_mm | 3D_REG_bilinear_median_mm
  3D_REG_bicubic_mean_mm  | 3D_REG_bicubic_median_mm
  3D_PC_bilinear_mean_mm  | 3D_PC_bilinear_median_mm
  3D_PC_bicubic_mean_mm   | 3D_PC_bicubic_median_mm
  n_corners_2D | n_corners_3D_REG_bilinear | n_corners_3D_REG_bicubic
  n_corners_3D_PC_bilinear | n_corners_3D_PC_bicubic
  elapsed_s | status
"""

from __future__ import annotations

import os
<<<<<<< HEAD
=======
import shutil
>>>>>>> sweep
import time
import traceback
from typing import Optional

import numpy as np

from spectrabreast.pipeline import run_full_pipeline, load_mesh

<<<<<<< HEAD
try:
    import torch
    from render_gpu import render_orthographic_topview_gpu
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
=======
# render_orthographic_topview_gpu NON viene importato qui:
# sweep.py lo riceve come parametro da main.py (che lo importa già correttamente).
TORCH_AVAILABLE = True  # placeholder; la disponibilità reale è gestita da main.py
>>>>>>> sweep


# =============================================================================
# Helper: statistiche robuste su array con NaN
# =============================================================================

def _stats(arr) -> tuple[float, float, int]:
    """Ritorna (mean, median, n_valid) ignorando NaN.
    Se arr è None o tutto NaN, ritorna (nan, nan, 0)."""
    if arr is None:
        return float('nan'), float('nan'), 0
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float('nan'), float('nan'), 0
    v = a[~np.isnan(a)]
    if v.size == 0:
        return float('nan'), float('nan'), 0
    return float(v.mean()), float(np.median(v)), int(v.size)


def _px_per_mm_from_data(data_hsi: dict, marker_side_mm: float) -> float:
    """Replica della funzione interna del pipeline per scala px/mm da marker HSI."""
    sides = []
    for corners in data_hsi.values():
        c = np.asarray(corners, dtype=np.float32)
        sides.append(float(np.mean([
            np.linalg.norm(c[(j + 1) % 4] - c[j]) for j in range(4)
        ])))
    return float(np.mean(sides)) / marker_side_mm if sides else 1.0


# =============================================================================
# Parsing / validazione delle coppie dal config
# =============================================================================

def parse_pairs(sweep_cfg: dict) -> list[tuple[float, float]]:
    """
    Estrae la lista di coppie [reg_mm_per_px, pc_mm_per_px] dal blocco
    `sweep.resolution_pairs` di config.yaml.

    Formati YAML accettati:
        resolution_pairs:
          - [0.5, 0.3]
          - [0.7, 0.4]
          - [1.0, 0.5]

    Lancia ValueError se il formato è invalido o la lista è vuota.
    """
    raw = sweep_cfg.get('resolution_pairs')
    if raw is None:
        raise ValueError(
            "sweep.resolution_pairs non trovato in config.yaml. "
            "Aggiungi una lista di coppie [reg, pc] (es. [[0.5, 0.3], [1.0, 0.5]])."
        )
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError("sweep.resolution_pairs deve essere una lista NON vuota.")

    pairs: list[tuple[float, float]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"sweep.resolution_pairs[{i}]: atteso [reg, pc] (2 numeri), "
                f"ricevuto {item!r}"
            )
        try:
            r = float(item[0])
            p = float(item[1])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"sweep.resolution_pairs[{i}]: valori non numerici -> {item!r}"
            ) from e
        if r <= 0 or p <= 0:
            raise ValueError(
                f"sweep.resolution_pairs[{i}]: risoluzioni devono essere > 0, "
                f"ricevuto reg={r}, pc={p}"
            )
        pairs.append((r, p))
    return pairs


# =============================================================================
# Singola run dello sweep (NESSUN salvataggio: solo metriche)
# =============================================================================

def _run_single_pair(
    cfg: dict,
    aruco_dict_cv: int,
    res_reg: float,
    res_pc: float,
    torch_device: Optional[str],
    use_torch_render: bool,
<<<<<<< HEAD
    mesh,  # mesh già caricata (riusata tra le run -> evita I/O ripetuto)
) -> dict:
    """
    Esegue render + run_full_pipeline per una coppia, senza salvare nulla.
=======
    mesh,           # mesh già caricata (riusata tra le run -> evita I/O ripetuto)
    tmp_dir: str,   # subdir temporanea dedicata a questa run
    render_fn,      # render_orthographic_topview_gpu passata da main.py (o None)
) -> dict:
    """
    Esegue render + run_full_pipeline per una coppia.
    run_full_pipeline scrive il suo Excel per-run in tmp_dir (che viene
    rimossa dal chiamante al termine dello sweep).
>>>>>>> sweep
    Ritorna un dict con le metriche aggregate per l'Excel riepilogo.
    """
    print(f"\n{'=' * 68}")
    print(f"  SWEEP RUN  —  res_reg = {res_reg} mm/px   res_pc = {res_pc} mm/px")
    print(f"{'=' * 68}")
    t0 = time.time()

<<<<<<< HEAD
=======
    os.makedirs(tmp_dir, exist_ok=True)
>>>>>>> sweep
    dual = abs(res_reg - res_pc) > 1e-6

    # ── Render GPU (pre-calcolato, identico a main.py) ──────────────────────
    precomputed_render = None
    precomputed_render_reg = None

    if use_torch_render:
        # PC (risoluzione fine)
        print(f"[Sweep][Render PC] {res_pc} mm/px su {torch_device}...")
        t0r = time.time()
        render_rgb_pc, depth_map_pc, xyz_map_pc, origin_xy_pc, res_out_pc = \
<<<<<<< HEAD
            render_orthographic_topview_gpu(
=======
            render_fn(
>>>>>>> sweep
                mesh,
                resolution_mm_per_px=res_pc,
                margin_mm=cfg['render']['margin_mm'],
                device=torch_device,
            )
        print(f"[Sweep][Render PC] done in {time.time() - t0r:.1f}s")
        precomputed_render = (render_rgb_pc, depth_map_pc, xyz_map_pc,
                              origin_xy_pc, res_out_pc)

        # REG (risoluzione grossolana se diversa, altrimenti riusa)
        if dual:
            print(f"[Sweep][Render REG] {res_reg} mm/px su {torch_device}...")
            t0r = time.time()
            render_rgb_reg, depth_map_reg, xyz_map_reg, origin_xy_reg, res_out_reg = \
<<<<<<< HEAD
                render_orthographic_topview_gpu(
=======
                render_fn(
>>>>>>> sweep
                    mesh,
                    resolution_mm_per_px=res_reg,
                    margin_mm=cfg['render']['margin_mm'],
                    device=torch_device,
                )
            print(f"[Sweep][Render REG] done in {time.time() - t0r:.1f}s")
            precomputed_render_reg = (render_rgb_reg, depth_map_reg,
                                      xyz_map_reg, origin_xy_reg, res_out_reg)
        else:
            precomputed_render_reg = precomputed_render
            print("[Sweep][Render] reg == pc — render condiviso.")

<<<<<<< HEAD
    # ── Pipeline (TUTTI i save forzati a False) ─────────────────────────────
=======
    # ── Pipeline ─────────────────────────────────────────────────────────────
    # output_dir = tmp_dir: run_full_pipeline scrive qui il suo Excel per-run
    # (Step 7 della pipeline è sempre eseguito). Tutto il resto è disabilitato.
>>>>>>> sweep
    result = run_full_pipeline(
        hsi_hdr_path             = cfg['paths']['hsi_hdr'],
        mesh_path                = cfg['paths']['mesh'],
        aruco_json_path          = cfg['paths']['aruco_json'],
<<<<<<< HEAD
        output_dir               = cfg['paths']['output_dir'],  # non usato (save=False)
=======
        output_dir               = tmp_dir,              # <-- subdir temporanea
>>>>>>> sweep
        hsi_extraction_method    = cfg['registration']['hsi_extraction_method'],
        aruco_dict_type          = aruco_dict_cv,
        suspicious_pixels_hsi    = None,
        render_resolution_mm     = res_pc,
        render_resolution_reg_mm = res_reg,
        render_margin_mm         = cfg['render']['margin_mm'],
        marker_side_mm           = cfg['registration']['marker_side_mm'],
        use_subpix               = cfg['registration'].get('use_subpix', True),
        subpix_winsize           = cfg['registration'].get('subpix_winsize', 5),
        border_px                = cfg['pointcloud']['border_px'],
        reflectance_norm         = cfg['pointcloud']['reflectance_norm'],
        pc_chunk_size            = cfg['pointcloud'].get('pc_chunk_size', 100_000),
        export_ply_file          = False,
        export_npy_file          = False,
        export_csv_file          = False,
        csv_max_points           = cfg['export']['csv_max_points'],
        save_pointcloud          = False,   # <-- NIENTE point cloud
        save_images              = False,   # <-- NIENTE immagini
        sample_name              = cfg['sample_name'],
        precomputed_render       = precomputed_render,
        precomputed_render_reg   = precomputed_render_reg,
    )

    # ── Estrazione metriche ────────────────────────────────────────────────
    err2d_px       = result.get('err2d_px')
    err3d_dict     = result.get('err3d_dict')     # REG
    err3d_dict_pc  = result.get('err3d_dict_pc')  # PC (None se non dual)
    data_hsi       = result.get('data_hsi')

    # Scala px -> mm dai marker HSI rilevati
    if data_hsi:
        px_per_mm = _px_per_mm_from_data(
            data_hsi, cfg['registration']['marker_side_mm']
        )
    else:
        px_per_mm = float('nan')

    # 2D (px e mm)
    if err2d_px is not None:
        err2d = np.asarray(err2d_px, dtype=np.float64)
        m2px, med2px, n2 = _stats(err2d)
        if not np.isnan(px_per_mm) and px_per_mm > 0:
            err2d_mm = err2d / px_per_mm
            m2mm, med2mm, _ = _stats(err2d_mm)
        else:
            m2mm, med2mm = float('nan'), float('nan')
    else:
        m2px = med2px = m2mm = med2mm = float('nan')
        n2 = 0

    # 3D REG
    if err3d_dict is not None:
        m_rb_bil, md_rb_bil, n_rb_bil = _stats(err3d_dict.get('bilinear'))
        m_rb_bic, md_rb_bic, n_rb_bic = _stats(err3d_dict.get('bicubic'))
    else:
        m_rb_bil = md_rb_bil = m_rb_bic = md_rb_bic = float('nan')
        n_rb_bil = n_rb_bic = 0

    # 3D PC
    if err3d_dict_pc is not None:
        m_pc_bil, md_pc_bil, n_pc_bil = _stats(err3d_dict_pc.get('bilinear'))
        m_pc_bic, md_pc_bic, n_pc_bic = _stats(err3d_dict_pc.get('bicubic'))
    else:
        # Non-dual: la "griglia PC" coincide con quella REG => copio i valori REG
        # per chiarezza nell'Excel (e segnalo con n_corners_3D_PC_* = n_REG).
        m_pc_bil, md_pc_bil, n_pc_bil = m_rb_bil, md_rb_bil, n_rb_bil
        m_pc_bic, md_pc_bic, n_pc_bic = m_rb_bic, md_rb_bic, n_rb_bic

    elapsed = time.time() - t0
    print(f"\n[Sweep] Run completata in {elapsed:.1f}s")

    return {
        'res_reg_mm_pix'                : res_reg,
        'res_pc_mm_pix'                 : res_pc,
        '2D_mean_px'                    : m2px,
        '2D_median_px'                  : med2px,
        '2D_mean_mm'                    : m2mm,
        '2D_median_mm'                  : med2mm,
        '3D_REG_bilinear_mean_mm'       : m_rb_bil,
        '3D_REG_bilinear_median_mm'     : md_rb_bil,
        '3D_REG_bicubic_mean_mm'        : m_rb_bic,
        '3D_REG_bicubic_median_mm'      : md_rb_bic,
        '3D_PC_bilinear_mean_mm'        : m_pc_bil,
        '3D_PC_bilinear_median_mm'      : md_pc_bil,
        '3D_PC_bicubic_mean_mm'         : m_pc_bic,
        '3D_PC_bicubic_median_mm'       : md_pc_bic,
        'n_corners_2D'                  : n2,
        'n_corners_3D_REG_bilinear'     : n_rb_bil,
        'n_corners_3D_REG_bicubic'      : n_rb_bic,
        'n_corners_3D_PC_bilinear'      : n_pc_bil,
        'n_corners_3D_PC_bicubic'       : n_pc_bic,
        'elapsed_s'                     : round(elapsed, 2),
        'status'                        : 'ok',
    }


def _empty_row(res_reg: float, res_pc: float, status: str) -> dict:
    """Riga di placeholder quando una run fallisce."""
    nan = float('nan')
    return {
        'res_reg_mm_pix'                : res_reg,
        'res_pc_mm_pix'                 : res_pc,
        '2D_mean_px'                    : nan,
        '2D_median_px'                  : nan,
        '2D_mean_mm'                    : nan,
        '2D_median_mm'                  : nan,
        '3D_REG_bilinear_mean_mm'       : nan,
        '3D_REG_bilinear_median_mm'     : nan,
        '3D_REG_bicubic_mean_mm'        : nan,
        '3D_REG_bicubic_median_mm'      : nan,
        '3D_PC_bilinear_mean_mm'        : nan,
        '3D_PC_bilinear_median_mm'      : nan,
        '3D_PC_bicubic_mean_mm'         : nan,
        '3D_PC_bicubic_median_mm'       : nan,
        'n_corners_2D'                  : 0,
        'n_corners_3D_REG_bilinear'     : 0,
        'n_corners_3D_REG_bicubic'      : 0,
        'n_corners_3D_PC_bilinear'      : 0,
        'n_corners_3D_PC_bicubic'       : 0,
        'elapsed_s'                     : 0.0,
        'status'                        : status,
    }


# =============================================================================
# Scrittura Excel riepilogo
# =============================================================================

def _write_summary_xlsx(rows: list[dict], output_path: str, sample_name: str) -> None:
    """Scrive l'Excel riepilogo dello sweep.

    Usa openpyxl direttamente (zero dipendenze in più: il pipeline lo usa già)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = 'Sweep Summary'

    # Stili
    hdr_font = Font(bold=True, color='FFFFFF', size=11)
    hdr_fill = PatternFill('solid', start_color='305496')
    hdr_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    thin = Side(border_style='thin', color='BFBFBF')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal='center', vertical='center')
    res_fill = PatternFill('solid', start_color='FFF2CC')   # giallo chiaro
    err2d_fill = PatternFill('solid', start_color='DDEBF7') # azzurro chiaro
    reg_fill   = PatternFill('solid', start_color='D9E1F2') # blu chiaro
    pc_fill    = PatternFill('solid', start_color='E2EFDA') # verde chiaro
    meta_fill  = PatternFill('solid', start_color='F2F2F2') # grigio chiaro

    # Header: titolo
    ws.cell(row=1, column=1, value=f"Resolution Sweep — {sample_name}").font = \
        Font(bold=True, size=14, color='1F4E78')
    ws.cell(row=2, column=1,
            value=f"Run: {len(rows)} configurazioni  "
                  f"|  Generato: {time.strftime('%Y-%m-%d %H:%M:%S')}").font = \
        Font(italic=True, color='595959')

    # Definizione colonne
    columns: list[tuple[str, str, PatternFill]] = [
        ('res_reg_mm_pix',           'res_reg\n(mm/px)',          res_fill),
        ('res_pc_mm_pix',            'res_pc\n(mm/px)',           res_fill),

        ('2D_mean_px',               '2D mean\n(px)',             err2d_fill),
        ('2D_median_px',             '2D median\n(px)',           err2d_fill),
        ('2D_mean_mm',               '2D mean\n(mm)',             err2d_fill),
        ('2D_median_mm',             '2D median\n(mm)',           err2d_fill),

        ('3D_REG_bilinear_mean_mm',  '3D REG bilinear\nmean (mm)',   reg_fill),
        ('3D_REG_bilinear_median_mm','3D REG bilinear\nmedian (mm)', reg_fill),
        ('3D_REG_bicubic_mean_mm',   '3D REG bicubic\nmean (mm)',    reg_fill),
        ('3D_REG_bicubic_median_mm', '3D REG bicubic\nmedian (mm)',  reg_fill),

        ('3D_PC_bilinear_mean_mm',   '3D PC bilinear\nmean (mm)',    pc_fill),
        ('3D_PC_bilinear_median_mm', '3D PC bilinear\nmedian (mm)',  pc_fill),
        ('3D_PC_bicubic_mean_mm',    '3D PC bicubic\nmean (mm)',     pc_fill),
        ('3D_PC_bicubic_median_mm',  '3D PC bicubic\nmedian (mm)',   pc_fill),

        ('n_corners_2D',                'N corners\n2D',                meta_fill),
        ('n_corners_3D_REG_bilinear',   'N corners\n3D REG bil',        meta_fill),
        ('n_corners_3D_REG_bicubic',    'N corners\n3D REG bic',        meta_fill),
        ('n_corners_3D_PC_bilinear',    'N corners\n3D PC bil',         meta_fill),
        ('n_corners_3D_PC_bicubic',     'N corners\n3D PC bic',         meta_fill),

        ('elapsed_s',                'Elapsed\n(s)',              meta_fill),
        ('status',                   'Status',                    meta_fill),
    ]

    HEADER_ROW = 4

    # Scrivi header
    for ci, (_, label, fill) in enumerate(columns, start=1):
        c = ws.cell(row=HEADER_ROW, column=ci, value=label)
        c.font = hdr_font
        c.fill = fill if False else hdr_fill   # tutte le header in blu scuro
        c.alignment = hdr_align
        c.border = border
    ws.row_dimensions[HEADER_ROW].height = 36

    # Scrivi righe dati
    def _fmt(v):
        """Formatta NaN come stringa 'N/A', resto invariato."""
        if isinstance(v, float) and np.isnan(v):
            return 'N/A'
        return v

    for ri, row in enumerate(rows, start=HEADER_ROW + 1):
        for ci, (key, _, fill) in enumerate(columns, start=1):
            val = row.get(key, '')
            c = ws.cell(row=ri, column=ci, value=_fmt(val))
            c.fill = fill
            c.alignment = center
            c.border = border
            # Formattazione numerica per i valori float "mm" e "px"
            if isinstance(val, float) and not np.isnan(val):
                if key.endswith(('_mm', '_px')) or key == 'elapsed_s':
                    c.number_format = '0.0000'

    # Larghezze colonna (auto-fit grezzo basato su lunghezza header)
    for ci, (_, label, _) in enumerate(columns, start=1):
        max_len = max(len(line) for line in label.split('\n'))
        ws.column_dimensions[get_column_letter(ci)].width = max(max_len + 3, 12)

    # Freeze pane sotto l'header e sulla colonna res_pc
    ws.freeze_panes = ws.cell(row=HEADER_ROW + 1, column=3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wb.save(output_path)
    print(f"\n[Sweep] Excel riepilogo salvato -> {output_path}")


# =============================================================================
# Entry point dello sweep
# =============================================================================

def run_sweep(cfg: dict, aruco_dict_cv: int, torch_device: Optional[str],
<<<<<<< HEAD
              use_torch_render: bool) -> str:
=======
              use_torch_render: bool, render_fn=None) -> str:
>>>>>>> sweep
    """
    Esegue lo sweep completo e ritorna il path dell'Excel riepilogo.

    Parameters
    ----------
    cfg : dict
        Config completo (già passato da load_config + resolve_paths).
    aruco_dict_cv : int
        Costante cv2.aruco.DICT_* risolta da main.py.
    torch_device : str | None
        Device torch (es. 'cuda', 'cpu') — None se torch non disponibile.
    use_torch_render : bool
        True se il render GPU/torch è disponibile.
<<<<<<< HEAD
=======
    render_fn : callable | None
        render_orthographic_topview_gpu importata da main.py.
        Se None e use_torch_render=True, run_full_pipeline farà il render
        internamente (fallback CPU).
>>>>>>> sweep

    Returns
    -------
    path dell'Excel riepilogo scritto in cfg['paths']['output_dir'].
    """
    sweep_cfg = cfg.get('sweep', {}) or {}
    pairs = parse_pairs(sweep_cfg)

    print("\n" + "=" * 68)
    print(f"  SWEEP RESOLUTION  —  {len(pairs)} coppie da testare")
    print("=" * 68)
    for i, (r, p) in enumerate(pairs):
        print(f"  [{i + 1:2d}/{len(pairs)}]  res_reg = {r:>6.3f} mm/px   "
              f"res_pc = {p:>6.3f} mm/px")
    print()

<<<<<<< HEAD
    # Mesh caricata UNA volta sola (ottimizzazione: evita I/O ripetuto)
=======
    # Crea la output dir principale e la cartella temporanea per le run
    base_out = cfg['paths']['output_dir']
    os.makedirs(base_out, exist_ok=True)
    sweep_tmp_root = os.path.join(base_out, '_sweep_tmp')
    os.makedirs(sweep_tmp_root, exist_ok=True)
    print(f"[Sweep] Subdir temporanee in: {sweep_tmp_root}")
    print(f"[Sweep] (verranno eliminate al termine dello sweep)\n")

    # Mesh caricata UNA volta sola per tutte le run
>>>>>>> sweep
    if use_torch_render:
        print(f"[Sweep] Carico mesh: {cfg['paths']['mesh']}")
        mesh = load_mesh(cfg['paths']['mesh'], scale_m_to_mm=True)
    else:
        mesh = None  # run_full_pipeline farà il fallback CPU internamente

<<<<<<< HEAD
    # Crea la output dir per l'Excel riepilogo
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    rows: list[dict] = []
    for i, (res_reg, res_pc) in enumerate(pairs):
=======
    rows: list[dict] = []
    for i, (res_reg, res_pc) in enumerate(pairs):
        # Subdir dedicata a questa run (run_full_pipeline scrive qui il suo xlsx)
        tag = f"reg{str(res_reg).replace('.','p')}_pc{str(res_pc).replace('.','p')}"
        tmp_dir = os.path.join(sweep_tmp_root, f"run{i+1:02d}_{tag}")

>>>>>>> sweep
        try:
            row = _run_single_pair(
                cfg=cfg,
                aruco_dict_cv=aruco_dict_cv,
                res_reg=res_reg,
                res_pc=res_pc,
                torch_device=torch_device,
                use_torch_render=use_torch_render,
                mesh=mesh,
<<<<<<< HEAD
=======
                tmp_dir=tmp_dir,
                render_fn=render_fn,
>>>>>>> sweep
            )
        except Exception as e:
            print(f"\n[Sweep] ERRORE su coppia ({res_reg}, {res_pc}): {e}")
            traceback.print_exc()
<<<<<<< HEAD
            row = _empty_row(res_reg, res_pc, status=f'error: {type(e).__name__}')
=======
            row = _empty_row(res_reg, res_pc, status=f'error: {type(e).__name__}: {e}')
>>>>>>> sweep

        rows.append(row)
        print(f"\n[Sweep] Progresso: {i + 1}/{len(pairs)} run completate.\n")

<<<<<<< HEAD
    # Scrittura Excel riepilogo
    sample = cfg['sample_name']
    out_path = os.path.join(
        cfg['paths']['output_dir'],
        f"{sample}_sweep_resolution_summary.xlsx"
    )
=======
    # Rimozione cartella temporanea (tutto il contenuto)
    try:
        shutil.rmtree(sweep_tmp_root)
        print(f"[Sweep] Cartella temporanea rimossa: {sweep_tmp_root}")
    except Exception as e:
        print(f"[Sweep] Avviso: impossibile rimuovere {sweep_tmp_root}: {e}")

    # Scrittura Excel riepilogo
    sample = cfg['sample_name']
    out_path = os.path.join(base_out, f"{sample}_sweep_resolution_summary.xlsx")
>>>>>>> sweep
    _write_summary_xlsx(rows, out_path, sample_name=sample)

    # Riepilogo a video
    print("\n" + "=" * 68)
    print("SWEEP COMPLETATO")
    print("=" * 68)
    print(f"  Coppie eseguite : {len(rows)}")
    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    print(f"  OK / Falliti    : {n_ok} / {len(rows) - n_ok}")
    print(f"  Output Excel    : {out_path}")
    print("=" * 68)

    return out_path
