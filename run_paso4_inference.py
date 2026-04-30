"""
Paso 4 — Inferencia CNN-2D por genotipo sobre el test set.
Análisis de leakage por genotipo — Corrección #20 del jurado.
"""
import os, json, joblib, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.features import rasterize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from spectralcrop.utils.path_manager import PathManager


def _resolve(p):
    if isinstance(p, list):
        return [x for x in p if 'spectralcrop' not in x][0]
    return p


PM             = PathManager()
RAW_PATH       = _resolve(PM.get_abs_path_folder('raw'))
PROCESSED_PATH = _resolve(PM.get_abs_path_folder('processed'))
INTERIM_PATH   = _resolve(PM.get_abs_path_folder('interim'))
MODELS_PATH    = _resolve(PM.get_abs_path_folder('models'))
REPORTS_PATH   = _resolve(PM.get_abs_path_folder('reports'))

FIGURES_DIR = os.path.join(REPORTS_PATH, 'figures', 'category_D')
os.makedirs(FIGURES_DIR, exist_ok=True)

GENOTYPE_MAP = {
    1: 'L1-12702',  2: 'L2-G11819', 3: 'L3-G50834', 4: 'L4-50840',
    5: 'L15-51433', 6: 'L17-G51018', 7: 'Liborino',  8: 'Cargamanto',
    9: 'entry_9_NO_doc', 10: 'entry_10_NO_doc',
}
OFFICIAL_ENTRIES = list(range(1, 9))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_start = time.time()


# ============================================================
# A — Cargar modelo CNN-2D
# ============================================================
print("[A] Cargando modelo CNN-2D...")

class SpectralSpatialCNN2D(nn.Module):
    def __init__(self, n_channels=63, n_classes=2, dropout=0.3, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=pad)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1   = nn.Linear(128 * 2 * 2, 128)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


with open(os.path.join(MODELS_PATH, 'cnn2d_final_model_info.json')) as f:
    model_info = json.load(f)

model2d = SpectralSpatialCNN2D(
    n_channels=model_info['n_channels'],
    dropout=model_info['dropout'],
    kernel_size=model_info['kernel_size'],
).to(device)
model2d.load_state_dict(
    torch.load(os.path.join(MODELS_PATH, 'cnn2d_final_model_weights.pt'),
               map_location=device, weights_only=False)
)
model2d.eval()

PATCH    = model_info['patch_size']
BEST_THR = model_info['best_thr']
print(f"    OK: {sum(p.numel() for p in model2d.parameters()):,} params, patch={PATCH}, thr={BEST_THR:.4f}")


# ============================================================
# B — Construir feature cube desde Zarr (63 canales)
# Orden confirmado: [NDVI, NDRE, CIgreen, PRI, PSRI, band_1, ..., band_374]
# ============================================================
print("[B] Construyendo feature cube desde masked_reflectance.zarr...")
t0 = time.time()

ZARR_PATH  = os.path.join(INTERIM_PATH, 'masked_reflectance.zarr')
ds_zarr    = xr.open_zarr(ZARR_PATH)

bands_df       = pd.read_csv(os.path.join(INTERIM_PATH, 'bands_selected_by_segment.csv'))
selected_bands = bands_df['band_index'].tolist()

VI_NAMES       = ['NDVI', 'NDRE', 'CIgreen', 'PRI', 'PSRI']
indices_da     = ds_zarr[VI_NAMES].to_array(dim='band')
reflectance_sel = ds_zarr['reflectance'].isel(band=selected_bands)

features_stack = xr.concat([indices_da, reflectance_sel], dim='band')
features_stack = features_stack.transpose('y', 'x', 'band')

print("    Materializando a numpy...")
cube = features_stack.values.astype(np.float32)
H, W, n_feat = cube.shape
print(f"    cube: {cube.shape}, elapsed: {time.time()-t0:.1f}s")


# ============================================================
# C — RobustScaler
# ============================================================
print("[C] Aplicando RobustScaler...")
scaler = joblib.load(os.path.join(MODELS_PATH, 'robust_scaler.pkl'))
print(f"    scaler n_features: {scaler.n_features_in_}")

cube_scaled = scaler.transform(cube.reshape(-1, n_feat)).reshape(H, W, n_feat).astype(np.float32)
print(f"    cube_scaled: {cube_scaled.shape}, elapsed: {time.time()-t0:.1f}s")


# ============================================================
# D — Split raster, labels_bin, entry_2d
# ============================================================
print("[D] Cargando rasters de split, etiquetas y entry...")

SPLIT_TIF = os.path.join(PROCESSED_PATH, 'splits', 'by_plot_split_id_binary.tif')
with rasterio.open(SPLIT_TIF) as src:
    split_id  = src.read(1)
    transform = src.transform
    crs       = src.crs

split_2d = np.where(split_id == 0, np.nan, split_id.astype(float))
print(f"    test pixels: {(split_id == 3).sum():,}")

gdf   = gpd.read_file(os.path.join(RAW_PATH, 'labels_export.gpkg'), layer='labels2')
gdf_r = gdf.to_crs(crs)

shapes_bin = [(g, int(v)) for g, v in zip(gdf_r.geometry, gdf_r['binary']) if g]
shapes_ent = [(g, int(v)) for g, v in zip(gdf_r.geometry, gdf_r['entry'])  if g]

labels_raw = rasterize(shapes_bin, out_shape=(H, W), transform=transform, fill=-1, dtype=np.int32)
labels_bin = np.where(labels_raw == -1, np.nan, labels_raw.astype(float))
entry_2d   = rasterize(shapes_ent, out_shape=(H, W), transform=transform, fill=0,  dtype=np.int32)

print(f"    labels unique: {np.unique(labels_bin[~np.isnan(labels_bin)])}")
print(f"    entry unique:  {np.unique(entry_2d)}")


# ============================================================
# E — Extraer parches del test set con coordenadas
# ============================================================
print("[E] Extrayendo parches test set...")
t1 = time.time()

r = PATCH // 2
X_list, y_list, coords = [], [], []
for i in range(r, H - r):
    for j in range(r, W - r):
        if split_2d[i, j] != 3:
            continue
        if np.isnan(labels_bin[i, j]):
            continue
        patch = cube_scaled[i-r:i+r+1, j-r:j+r+1, :]
        if np.isnan(patch).any():
            continue
        X_list.append(np.transpose(patch, (2, 0, 1)))
        y_list.append(labels_bin[i, j])
        coords.append((i, j))

X_test      = np.stack(X_list)
y_test      = np.array(y_list, dtype=np.int64)
coords_test = np.array(coords)
print(f"    Test set: {X_test.shape[0]:,} muestras, elapsed: {time.time()-t1:.1f}s")


# ============================================================
# F — Inferencia + CHECK DE SANIDAD
# ============================================================
print("[F] Corriendo inferencia...")
t2 = time.time()

loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=512, shuffle=False)
probs  = []
with torch.no_grad():
    for (xb,) in loader:
        prob = torch.softmax(model2d(xb), dim=1)[:, 1].numpy()
        probs.append(prob)
y_prob_test = np.concatenate(probs)
print(f"    Inferencia: {time.time()-t2:.1f}s")

pr_global  = average_precision_score(y_test, y_prob_test)
roc_global = roc_auc_score(y_test, y_prob_test)
print(f"\n>>> CHECK DE SANIDAD:")
print(f"    PR-AUC  test: {pr_global:.4f}  (esperado 0.9635 +/- 0.001)")
print(f"    ROC-AUC test: {roc_global:.4f}")
sanity_ok = abs(pr_global - 0.9635) <= 0.01
print(f"    {'OK: PASADO' if sanity_ok else '*** FALLO — DETENER ***'}")

if not sanity_ok:
    raise RuntimeError(f"Sanity check fallido: PR-AUC={pr_global:.4f}")


# ============================================================
# G — Metricas por genotipo
# ============================================================
print("\n[G] Calculando metricas por genotipo...")
y_pred_test  = (y_prob_test >= BEST_THR).astype(int)
entries_test = np.array([entry_2d[i, j] for i, j in coords_test])

rows = []
for entry_id in sorted(np.unique(entries_test)):
    mask  = entries_test == entry_id
    yt    = y_test[mask]
    yp    = y_pred_test[mask]
    ypr   = y_prob_test[mask]
    n_pos = int(yt.sum())
    n_neg = int((yt == 0).sum())
    both  = n_pos > 0 and n_neg > 0
    rows.append({
        'entry':       int(entry_id),
        'genotipo':    GENOTYPE_MAP.get(int(entry_id), f'entry_{int(entry_id)}'),
        'documentado': 'Si' if entry_id <= 8 else 'NO',
        'n_px':        len(yt),
        'n_pos':       n_pos,
        'n_neg':       n_neg,
        'PR_AUC':    round(average_precision_score(yt, ypr), 4) if both else float('nan'),
        'ROC_AUC':   round(roc_auc_score(yt, ypr), 4)           if both else float('nan'),
        'F1':        round(f1_score(yt, yp, zero_division=0), 4),
        'Precision': round(precision_score(yt, yp, zero_division=0), 4),
        'Recall':    round(recall_score(yt, yp, zero_division=0), 4),
        'Accuracy':  round(accuracy_score(yt, yp), 4),
    })

tabla_A = pd.DataFrame(rows)

print("\n=== TABLA A: Metricas por entry (todos los entries en test) ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
print(tabla_A.to_string(index=False))

mask_off   = np.isin(entries_test, OFFICIAL_ENTRIES)
pr_oficial = average_precision_score(y_test[mask_off], y_prob_test[mask_off])
tabla_B    = tabla_A[tabla_A['entry'].isin(OFFICIAL_ENTRIES)].copy()

print(f"\n=== TABLA B: Metricas restringidas a entries oficiales (1-8) ===")
print(tabla_B.to_string(index=False))
print(f"\nPR-AUC global restringido a entries 1-8: {pr_oficial:.4f}")
print(f"PR-AUC global (todos): {pr_global:.4f}")

tabla_A.to_csv(os.path.join(FIGURES_DIR, 'tabla_A_metricas_por_entry.csv'), index=False)
tabla_B.to_csv(os.path.join(FIGURES_DIR, 'tabla_B_metricas_oficiales.csv'), index=False)

print(f"\nTiempo total: {time.time()-t_start:.1f}s")
print("DONE")
