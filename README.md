# SpectraBreast — Pipeline HSI + 3D Mesh

Registrazione di immagini iperspettrali (HSI) su mesh 3D e generazione di point cloud spettrali.

---

## Cosa fa

1. Carica il cubo HSI (formato ENVI) e la mesh 3D
2. Rileva marker ArUco sull'immagine HSI e li associa alle posizioni 3D note (JSON)
3. Calcola una omografia HSI → render per allineare i due spazi
4. Per ogni pixel della mesh con coordinate 3D valide, legge lo spettro completo dall'HSI
5. Esporta il point cloud spettrale in `.ply` (CloudCompare), `.npz` (Python), `.csv` (opzionale)
6. Salva un report Excel con gli errori di registrazione 2D e 3D

---

## Struttura del progetto

```
SpectraBreast/
├── main.py                        # punto di ingresso
├── config.yaml                    # tutti i parametri
├── Pipeline_registrazione_2D3D.py # logica pipeline (non modificare)
├── requirements.txt               # dipendenze pip
├── README.md
├── .gitignore
└── data/
    ├── input/                     # file di input (non versionati)
    │   ├── SB019_raw.hdr
    │   ├── SB019_raw.bin          # (o .raw / .img — stesso nome del .hdr)
    │   ├── surface_mesh.ply
    │   └── aruco_markers_3d.json
    └── output/                    # risultati generati automaticamente
        ├── SB019_render_topview.png
        ├── SB019_render_turbo.png
        ├── SB019_registration.png
        ├── SB019_coverage.png
        ├── SB019_registration_errors.xlsx
        ├── spectral_pointcloud_SB019.ply
        └── spectral_pointcloud_SB019.npz
```

---

## Installazione

### 1. Clona il repository

```bash
git clone <url-repo>
cd SpectraBreast
```

### 2. Crea un ambiente virtuale (raccomandato)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

> **Ray casting accelerato (opzionale ma raccomandato)**
> Se disponibile, installa `pyembree` per ridurre il tempo di render da minuti a secondi.
> Segui le istruzioni su https://github.com/scopatz/pyembree

---

## Configurazione

Apri `config.yaml` e modifica i parametri. I path sono **relativi alla root del progetto**.

```yaml
paths:
  hsi_hdr:    "data/input/SB019_raw.hdr"
  mesh:       "data/input/surface_mesh.ply"
  aruco_json: "data/input/aruco_markers_3d.json"
  output_dir: "data/output"

sample_name: "SB019"

render:
  resolution_mm_per_px: 2.0   # aumenta per più velocità, riduci per più dettaglio
```

### Parametro chiave: `resolution_mm_per_px`

| Valore | Velocità | Densità point cloud | Uso consigliato |
|--------|----------|----------------------|-----------------|
| `5.0`  | ⚡ rapido | bassa                | test rapido |
| `2.0`  | ✅ buono  | media                | uso standard |
| `1.0`  | 🐢 lento  | alta                 | analisi dettagliata |
| `0.5`  | 🐢🐢 molto lento | molto alta  | solo con pyembree |

---

## Utilizzo

### Esecuzione standard

```bash
python main.py
```

### Opzioni da riga di comando

```bash
# Usa un config alternativo (es. per un campione diverso)
python main.py --config config_SB020.yaml

# Sovrascrive il nome campione senza modificare il config
python main.py --sample SB020

# Sovrascrive la risoluzione del render
python main.py --resolution 1.0

# Valida config e input senza eseguire (utile per check rapido)
python main.py --dry-run
```

### Esempio: aggiungere un nuovo campione

1. Copia `config.yaml` in `config_SB020.yaml`
2. Modifica i path e `sample_name: "SB020"` nel nuovo file
3. Lancia: `python main.py --config config_SB020.yaml`

---

## Input richiesti

| File | Formato | Note |
|------|---------|------|
| HSI cube | ENVI (`.hdr` + dati binari) | interleave BSQ / BIL / BIP |
| Mesh 3D | `.ply` o `.obj` | coordinate in **metri** |
| ArUco JSON | `.json` | generato dal software di ricostruzione 3D |

### Formato ArUco JSON

```json
{
  "markers": {
    "0": { "corners_3d": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]] },
    "1": { "corners_3d": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]] }
  }
}
```

Le coordinate sono in **metri** (la pipeline converte automaticamente in mm).

---

## Output generati

| File | Descrizione |
|------|-------------|
| `*_render_topview.png` | Vista ortografica in scala di grigi (profondità) |
| `*_render_turbo.png` | Vista ortografica colorata con colormap TURBO |
| `*_registration.png` | Overlay HSI + render con marker ArUco e punti proiettati |
| `*_coverage.png` | Mappa di copertura: quali pixel HSI sono mappati sul 3D |
| `*_registration_errors.xlsx` | Errori 2D (px, mm) e 3D (mm) per ogni corner ArUco |
| `spectral_pointcloud_*.ply` | Point cloud spettrale — apri in CloudCompare |
| `spectral_pointcloud_*.npz` | Point cloud spettrale — ricarica rapida in Python |

### Ricaricare il point cloud in Python

```python
import numpy as np

data = np.load("data/output/spectral_pointcloud_SB019.npz")
xyz      = data['xyz']         # (N, 3)  coordinate in mm
spectra  = data['spectra']     # (N, bands)  valori spettrali
wl       = data['wavelengths'] # (bands,)  lunghezze d'onda in nm

print(f"Punti: {xyz.shape[0]:,}  —  Bande: {spectra.shape[1]}")
```

---

## Punti sospetti (opzionale)

Per localizzare in 3D regioni di interesse identificate sull'HSI (es. output di una segmentazione),
modifica `main.py` nella sezione `suspicious pixels`:

```python
mask = np.load("data/input/segmentation_mask.npy")  # maschera binaria (H, W)
suspicious_pixels, _ = extract_suspicious_centroids(mask, min_area=5)
```

Le coordinate 3D risultanti vengono stampate a terminale e visualizzate sull'immagine di registrazione.

---

## Dipendenze

| Libreria | Versione minima | Scopo |
|----------|----------------|-------|
| numpy | 1.24 | calcolo numerico |
| opencv-contrib-python | 4.8 | ArUco, omografia |
| trimesh | 4.0 | mesh + ray casting |
| pyyaml | 6.0 | lettura config |
| openpyxl | 3.1 | report Excel |
| rtree | 1.0 | BVH fallback per trimesh |
| pyembree | qualsiasi | ray casting accelerato (opzionale) |

---

## Troubleshooting

**La mesh non viene renderizzata (0 hits)**
Verifica che le unità della mesh siano in metri. Se sono già in mm, imposta `scale_m_to_mm: false` nella funzione `load_mesh` oppure adatta il JSON ArUco di conseguenza.

**Nessun marker ArUco trovato sull'HSI**
Prova a cambiare `hsi_extraction_method` da `mean` a `visible_band` nel config. Se i marker non sono visibili in nessuna banda, verifica che siano fisicamente presenti nell'immagine.

**Il render è molto lento**
Aumenta `resolution_mm_per_px` (es. da `2.0` a `5.0`) oppure installa `pyembree`.

**Errore "Data file not found"**
Il file binario dei dati HSI deve trovarsi nella stessa cartella del `.hdr` e avere lo stesso nome base (es. `SB019_raw.hdr` → `SB019_raw.raw` oppure `SB019_raw`).
