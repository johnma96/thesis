"""
Pruebas de ablación espectral — Corrección #20 del jurado.

Prueba 1: permutación espectral dentro del parche
  Permuta aleatoriamente los 63 canales de cada parche 5x5.
  Preserva estructura espacial, destruye contenido espectral.

Prueba 2: solo píxel central (sin contexto espacial)
  Zeriza todos los píxeles vecinos del parche 5x5.
  Solo conserva el espectro del píxel central.

Requiere: run_paso4_inference.py ejecutado previamente (X_test, y_test, y_prob_test).
"""
import os, json, joblib, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
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
FIGURES_DIR    = os.path.join(REPORTS_PATH, 'figures', 'category_D')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG    = np.random.default_rng(42)


# ============================================================
# Cargar modelo
# ============================================================
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
               map_location=device, weights_only=False))
model2d.eval()

PATCH    = model_info['patch_size']
BEST_THR = model_info['best_thr']
print(f"Modelo listo: patch={PATCH}, thr={BEST_THR:.4f}")


# ============================================================
# Reconstruir test set (mismo pipeline que run_paso4_inference.py)
# ============================================================
print("\nConstruyendo feature cube desde Zarr...")
t0 = time.time()

ds_zarr    = xr.open_zarr(os.path.join(INTERIM_PATH, 'masked_reflectance.zarr'))
bands_df   = pd.read_csv(os.path.join(INTERIM_PATH, 'bands_selected_by_segment.csv'))
sel_bands  = bands_df['band_index'].tolist()
VI_NAMES   = ['NDVI', 'NDRE', 'CIgreen', 'PRI', 'PSRI']

features_stack = xr.concat(
    [ds_zarr[VI_NAMES].to_array(dim='band'),
     ds_zarr['reflectance'].isel(band=sel_bands)],
    dim='band'
).transpose('y', 'x', 'band')

cube = features_stack.values.astype(np.float32)
H, W, n_feat = cube.shape

scaler      = joblib.load(os.path.join(MODELS_PATH, 'robust_scaler.pkl'))
cube_scaled = scaler.transform(cube.reshape(-1, n_feat)).reshape(H, W, n_feat).astype(np.float32)
print(f"  cube_scaled: {cube_scaled.shape}, {time.time()-t0:.1f}s")

SPLIT_TIF = os.path.join(PROCESSED_PATH, 'splits', 'by_plot_split_id_binary.tif')
with rasterio.open(SPLIT_TIF) as src:
    split_id  = src.read(1)
    transform = src.transform
    crs       = src.crs

split_2d = np.where(split_id == 0, np.nan, split_id.astype(float))

gdf   = gpd.read_file(os.path.join(RAW_PATH, 'labels_export.gpkg'), layer='labels2')
gdf_r = gdf.to_crs(crs)
shapes_bin = [(g, int(v)) for g, v in zip(gdf_r.geometry, gdf_r['binary']) if g]
labels_raw = rasterize(shapes_bin, out_shape=(H, W), transform=transform, fill=-1, dtype=np.int32)
labels_bin = np.where(labels_raw == -1, np.nan, labels_raw.astype(float))

print("Extrayendo parches test...")
r = PATCH // 2
X_list, y_list = [], []
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

X_test = np.stack(X_list)           # (N, 63, 5, 5)
y_test = np.array(y_list, dtype=np.int64)
print(f"  Test set: {X_test.shape[0]:,} muestras, {time.time()-t0:.1f}s")


# ============================================================
# Función de inferencia
# ============================================================
def run_inference(X, model, batch_size=512):
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    probs  = []
    with torch.no_grad():
        for (xb,) in loader:
            prob = torch.softmax(model(xb), dim=1)[:, 1].numpy()
            probs.append(prob)
    return np.concatenate(probs)


def report_metrics(label, y_true, y_prob, best_thr):
    pr  = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    yp  = (y_prob >= best_thr).astype(int)
    f1  = f1_score(y_true, yp, average='macro')
    print(f"  {label:<40s}  PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  F1-macro={f1:.4f}")
    return pr, roc, f1


# ============================================================
# BASELINE — modelo original (sanity check)
# ============================================================
print("\n[BASELINE] Modelo original...")
y_prob_orig = run_inference(X_test, model2d)
pr_base, roc_base, f1_base = report_metrics("modelo original", y_test, y_prob_orig, BEST_THR)
assert abs(pr_base - 0.9635) < 0.01, f"Sanity check fallido: {pr_base:.4f}"
print("  Sanity check OK")


# ============================================================
# PRUEBA 1 — Permutación espectral dentro del parche
# Mezcla aleatoriamente los 63 canales de cada parche.
# Estructura espacial 5x5 intacta; espectro destruido.
# ============================================================
print("\n[PRUEBA 1] Permutacion espectral dentro de cada parche...")

X_perm = X_test.copy()                              # (N, 63, 5, 5)
for i in range(X_perm.shape[0]):
    perm = RNG.permutation(n_feat)
    X_perm[i] = X_perm[i][perm]                     # reordena canales

y_prob_perm = run_inference(X_perm, model2d)
pr_perm, roc_perm, f1_perm = report_metrics("espectro permutado", y_test, y_prob_perm, BEST_THR)
delta_pr_1 = pr_base - pr_perm
print(f"  Caida de PR-AUC: {delta_pr_1:.4f} ({delta_pr_1/pr_base*100:.1f}% respecto al baseline)")


# ============================================================
# PRUEBA 2 — Solo píxel central (sin contexto espacial)
# Zeriza todos los píxeles vecinos del parche 5x5.
# Solo conserva el espectro del píxel central.
# ============================================================
print("\n[PRUEBA 2] Solo pixel central (vecindario zerizado)...")

X_center = np.zeros_like(X_test)                    # (N, 63, 5, 5) — todo ceros
cx = PATCH // 2                                      # índice central = 2 para patch=5
X_center[:, :, cx, cx] = X_test[:, :, cx, cx]       # conserva solo el pixel central

y_prob_center = run_inference(X_center, model2d)
pr_center, roc_center, f1_center = report_metrics("solo pixel central", y_test, y_prob_center, BEST_THR)
delta_pr_2 = pr_base - pr_center
print(f"  Caida de PR-AUC: {delta_pr_2:.4f} ({delta_pr_2/pr_base*100:.1f}% respecto al baseline)")


# ============================================================
# RESUMEN
# ============================================================
print("\n" + "="*70)
print("RESUMEN DE PRUEBAS DE ABLACION ESPECTRAL")
print("="*70)

results = pd.DataFrame([
    {"condicion": "Modelo original (baseline)",       "PR_AUC": pr_base,   "ROC_AUC": roc_base,   "F1_macro": f1_base},
    {"condicion": "Prueba 1: espectro permutado",     "PR_AUC": pr_perm,   "ROC_AUC": roc_perm,   "F1_macro": f1_perm},
    {"condicion": "Prueba 2: solo pixel central",     "PR_AUC": pr_center, "ROC_AUC": roc_center, "F1_macro": f1_center},
])
print(results.to_string(index=False))

results.to_csv(os.path.join(FIGURES_DIR, 'ablation_results.csv'), index=False)
print(f"\nResultados guardados en {FIGURES_DIR}/ablation_results.csv")
print(f"Tiempo total: {time.time()-t0:.1f}s")
