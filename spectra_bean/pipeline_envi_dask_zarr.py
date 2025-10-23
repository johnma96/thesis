#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ENVI (Dask/xarray+Zarr) con exportación COG y reporte HTML.

- Lectura lazy con rioxarray/xarray (GDAL ENVI) y chunking.
- Conversión a reflectancia física, máscara de NoData.
- Cálculo de NDVI, NDRE, PRI, PSRI, CIgreen con Dask.
- Exportación a **Zarr** (cubo reflectancia) y **COG multibanda**.
- Quicklooks (RGB, PCA usando muestreo computado).
- Reporte HTML con gráficos y logs.

Requisitos: numpy, dask, xarray, rioxarray, rasterio, matplotlib, pandas, scikit-learn.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# --------------- Utils ---------------

def setup_logger(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def parse_envi_hdr(hdr_path: Path) -> Dict:
    meta = {}
    txt = Path(hdr_path).read_text(encoding='utf-8', errors='ignore')
    def clean(v: str):
        v = v.strip()
        if v.startswith('{') and v.endswith('}'):
            v = v[1:-1]
        return v.strip()
    for line in txt.splitlines():
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip().lower()
        val = clean(val)
        if ',' in val and not val.replace(',', '').replace('.', '').replace('-', '').isdigit():
            items = [x.strip() for x in val.split(',')]
            meta[key] = items
        else:
            meta[key] = val
    if 'wavelength' in meta and isinstance(meta['wavelength'], str):
        import re
        m = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m:
            arr = [a.strip() for a in m.group(1).replace('
', ' ').split(',') if a.strip()]
            meta['wavelength'] = arr
    if 'fwhm' in meta and isinstance(meta['fwhm'], str):
        import re
        m = re.search(r'fwhm\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m:
            arr = [a.strip() for a in m.group(1).replace('
', ' ').split(',') if a.strip()]
            meta['fwhm'] = arr
    return meta


def get_wavelengths(meta: Dict) -> Optional[np.ndarray]:
    w = meta.get('wavelength', None)
    if w is None:
        return None
    try:
        return np.array([float(x) for x in w], dtype=np.float32)
    except Exception:
        return None


def get_fwhm(meta: Dict) -> Optional[np.ndarray]:
    f = meta.get('fwhm', None)
    if f is None:
        return None
    try:
        return np.array([float(x) for x in f], dtype=np.float32)
    except Exception:
        return None


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.nanargmin(np.abs(wavelengths - target_nm)))


# --------------- Core ---------------

def open_envi_xr(hdr_path: Path, chunks: Dict) -> xr.DataArray:
    da = rxr.open_rasterio(hdr_path, chunks=chunks, masked=True)  # (band, y, x)
    return da


def to_reflectance(da: xr.DataArray, scale: float, ignore_values: List[float]) -> xr.DataArray:
    arr = da.astype('float32') / scale
    for v in ignore_values:
        arr = arr.where(da != v)
    arr = arr.where((arr >= 0) & (arr <= 1.2))
    return arr


def compute_indices_dask(da_refl: xr.DataArray, wavelengths: np.ndarray) -> Dict[str, xr.DataArray]:
    def b(nm):
        return nearest_band(wavelengths, nm)
    R = da_refl.isel(band=b(660))
    N = da_refl.isel(band=b(800))
    RE = da_refl.isel(band=b(720))
    B531 = da_refl.isel(band=b(531))
    B570 = da_refl.isel(band=b(570))
    B678 = da_refl.isel(band=b(678))
    B500 = da_refl.isel(band=b(500))
    B750 = da_refl.isel(band=b(750))
    B554 = da_refl.isel(band=b(554))

    NDVI = (N - R) / (N + R + 1e-6)
    NDRE = (N - RE) / (N + RE + 1e-6)
    PRI  = (B531 - B570) / (B531 + B570 + 1e-6)
    PSRI = (B678 - B500) / (B750 + 1e-6)
    CIg  = (N / (B554 + 1e-6)) - 1.0
    return {'NDVI': NDVI, 'NDRE': NDRE, 'PRI': PRI, 'PSRI': PSRI, 'CIgreen': CIg}


def export_indices_geotiff(indices: Dict[str, xr.DataArray], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, da in indices.items():
        path = out_dir / f"{k}.tif"
        da.rio.write_nodata(-9999, inplace=True)
        da.astype('float32').rio.to_raster(path, tiled=True, compress='DEFLATE', windowed=True)
        with rasterio.open(path, 'r+') as dst:
            dst.build_overviews([2,4,8,16], Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')
        paths[k] = path
    return paths


def compute_veg_mask_tif(ndvi_path: Path, threshold: float = 0.3) -> Path:
    out = ndvi_path.with_name('veg_mask.tif')
    with rasterio.open(ndvi_path) as src:
        prof = src.profile.copy(); prof.update(dtype='uint8')
        with rasterio.open(out, 'w', **prof) as dst:
            for ji, window in src.block_windows(1):
                arr = src.read(1, window=window)
                mask = (arr > threshold) & (arr != -9999)
                dst.write(mask.astype('uint8'), 1, window=window)
            dst.set_nodata(0)
    return out


def export_cog_multiband_from_da(da_refl: xr.DataArray, out_path: Path, wavelengths: Optional[np.ndarray], exclude_water: bool = True) -> Tuple[Path, List[int]]:
    bands = np.arange(da_refl.sizes['band'])
    if wavelengths is not None and exclude_water:
        water = ((wavelengths > 1340) & (wavelengths < 1440)) | ((wavelengths > 1800) & (wavelengths < 1950))
        bands = np.where(~water)[0]
    logging.info(f"COG: exportando {len(bands)} bandas")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Escribir con rasterio leyendo ventanas del dataset fuente
    with rasterio.open(da_refl.encoding['source']) as src:
        profile = src.profile.copy()
        profile.update({'driver':'GTiff','count':len(bands),'dtype':'float32','compress':'DEFLATE','tiled':True,'blockxsize':512,'blockysize':512,'interleave':'band'})
        with rasterio.open(out_path, 'w', **profile) as dst:
            for i, b in enumerate(bands, start=1):
                for ji, window in src.block_windows(1):
                    r0, c0 = int(window.row_off), int(window.col_off)
                    r1, c1 = r0 + int(window.height), c0 + int(window.width)
                    chunk = np.array(da_refl.isel(band=int(b)).values[r0:r1, c0:c1].compute())
                    dst.write(np.nan_to_num(chunk, nan=-9999).astype('float32'), i, window=window)
            dst.set_nodata(-9999)
    with rasterio.open(out_path, 'r+') as dst:
        dst.build_overviews([2,4,8,16], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
    return out_path, bands.tolist()


def pca_quicklook_from_da(da_refl: xr.DataArray, wavelengths: Optional[np.ndarray], out_png: Path, sample_rows: int = 300) -> Path:
    h = da_refl.sizes['y']; w = da_refl.sizes['x']; B = da_refl.sizes['band']
    valid = np.arange(B)
    if wavelengths is not None:
        water = ((wavelengths > 1340) & (wavelengths < 1440)) | ((wavelengths > 1800) & (wavelengths < 1950))
        valid = np.where(~water)[0]
    rows = np.linspace(0, h-1, min(sample_rows, h), dtype=int)
    X_list = []
    for r in rows:
        row = da_refl.isel(y=int(r), band=valid).transpose('x','band')  # [W,B]
        X_list.append(row)
    X = xr.concat(X_list, dim='x').values
    X = np.array(X.compute())
    X = X[~np.any(~np.isfinite(X), axis=1)]
    pca = PCA(n_components=3, random_state=0).fit(X)

    out = np.zeros((h, w, 3), dtype=np.float32)
    step = 256
    for r0 in range(0, h, step):
        r1 = min(h, r0+step)
        block = da_refl.isel(y=slice(r0, r1), band=valid).transpose('y','x','band').values
        block = np.array(block.compute())
        shp = block.shape
        Xb = block.reshape(-1, shp[-1])
        mask = ~np.any(~np.isfinite(Xb), axis=1)
        Y = np.zeros((Xb.shape[0], 3), dtype=np.float32)
        Y[mask] = pca.transform(Xb[mask])
        Y = Y.reshape(shp[0], shp[1], 3)
        for k in range(3):
            lo, hi = np.nanpercentile(Y[...,k], [1,99])
            if hi-lo<1e-6:
                Y[...,k]=0
            else:
                Y[...,k]=np.clip((Y[...,k]-lo)/(hi-lo),0,1)
        out[r0:r1] = Y
    plt.figure(figsize=(7,7)); plt.imshow(np.nan_to_num(out)); plt.axis('off'); plt.title('PCA False Color'); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    return out_png


def build_html_report(out_path: Path, context: Dict):
    html = ["<html><head><meta charset='utf-8'><title>Reporte hiperespectral (Dask)</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:980px;margin:auto;padding:20px} img{max-width:100%}</style>",
            "</head><body>"]
    html.append("<h1>Reporte (Dask/xarray)</h1>")
    html.append(f"<p><b>Archivo:</b> {context.get('input_path','')}</p>")
    html.append(f"<p><b>Dimensiones:</b> {context.get('dims','')}</p>")
    if 'rgb' in context:
        html.append("<h2>Quicklook RGB</h2>")
        html.append(f"<img src='{Path(context['rgb']).name}' alt='RGB'>")
    if 'pca' in context:
        html.append("<h2>PCA False Color</h2>")
        html.append(f"<img src='{Path(context['pca']).name}' alt='PCA'>")
    if 'indices' in context:
        html.append("<h2>Índices</h2><ul>")
        for k, p in context['indices'].items():
            html.append(f"<li>{k}: {Path(p).name}</li>")
        html.append("</ul>")
    if 'zarr' in context:
        html.append(f"<h2>Zarr</h2><p>Almacenado en: {context['zarr']}</p>")
    if 'cog' in context:
        html.append(f"<h2>COG</h2><p>Archivo: {Path(context['cog']).name}</p>")
    if 'band_csv' in context:
        html.append(f"<p>CSV bandas exportadas: {Path(context['band_csv']).name}</p>")
    html.append("<hr><p>Generado automáticamente.</p>")
    html.append("</body></html>")
    out_path.write_text('
'.join(html), encoding='utf-8')


# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser(description='Pipeline ENVI (Dask/xarray+Zarr) con COG y reporte HTML')
    ap.add_argument('--hdr', required=True, help='Ruta al archivo .hdr de ENVI')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--ndvi_threshold', type=float, default=0.3)
    ap.add_argument('--exclude_water', action='store_true')
    ap.add_argument('--chunks', default='{"band":-1,"y":1024,"x":1024}', help='JSON con chunk sizes para xarray')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    setup_logger(outdir / 'pipeline_dask.log')

    hdr_path = Path(args.hdr)
    meta = parse_envi_hdr(hdr_path)
    wavelengths = get_wavelengths(meta)
    fwhm = get_fwhm(meta)

    scale = float(meta.get('reflectance scale factor', 10000.0))
    ignore_values = []
    if 'data ignore value' in meta:
        try: ignore_values.append(float(meta['data ignore value']))
        except: pass
    if 'background' in meta:
        try: ignore_values.append(float(meta['background']))
        except: pass
    logging.info(f"Scale={scale} Ignore={ignore_values}")

    import json as _json
    chunks = _json.loads(args.chunks)

    da = open_envi_xr(hdr_path, chunks=chunks)  # (band, y, x)
    h, w, B = int(da.sizes['y']), int(da.sizes['x']), int(da.sizes['band'])
    logging.info(f"Dataset: {h}x{w}x{B} CRS={da.rio.crs}")

    da_refl = to_reflectance(da, scale, ignore_values)

    if wavelengths is not None:
        r = nearest_band(wavelengths, 650)
        g = nearest_band(wavelengths, 560)
        b = nearest_band(wavelengths, 480)
    else:
        r, g, b = 0, 1, 2
    R = da_refl.isel(band=r).values.compute()
    G = da_refl.isel(band=g).values.compute()
    Bc= da_refl.isel(band=b).values.compute()
    def stretch(a):
        lo, hi = np.nanpercentile(a, [1,99]);
        return np.clip((a-lo)/(hi-lo+1e-6), 0, 1)
    rgb = np.dstack([stretch(R), stretch(G), stretch(Bc)])
    rgb_path = outdir / 'quicklook_rgb.png'
    plt.figure(figsize=(7,7)); plt.imshow(np.nan_to_num(rgb)); plt.axis('off'); plt.title('Quicklook RGB'); plt.tight_layout(); plt.savefig(rgb_path, dpi=200); plt.close()

    pca_path = pca_quicklook_from_da(da_refl, wavelengths, outdir / 'quicklook_pca.png')

    if wavelengths is None:
        raise ValueError('Se requieren longitudes de onda para índices en este pipeline Dask')
    indices_da = compute_indices_dask(da_refl, wavelengths)
    indices_paths = export_indices_geotiff(indices_da, outdir / 'indices')

    veg_path = compute_veg_mask_tif(indices_paths['NDVI'], threshold=args.ndvi_threshold)

    zarr_dir = outdir / 'zarr' / 'reflectance.zarr'
    zarr_dir.parent.mkdir(parents=True, exist_ok=True)
    coords = {'band': np.arange(B)}
    if wavelengths is not None:
        coords['wavelength'] = ('band', wavelengths)
    ds = xr.Dataset({'reflectance': da_refl}, coords=coords)
    ds.to_zarr(zarr_dir, mode='w')

    cog_path, exported = export_cog_multiband_from_da(da_refl, outdir / 'cog' / 'reflectance_valid_bands.tif', wavelengths, exclude_water=args.exclude_water)

    csv_path = outdir / 'cog' / 'reflectance_valid_bands.csv'
    df = pd.DataFrame({'band_index_0based': exported})
    if wavelengths is not None:
        df['wavelength_nm'] = [float(wavelengths[i]) for i in exported]
    if fwhm is not None:
        df['fwhm_nm'] = [float(fwhm[i]) for i in exported]
    df.to_csv(csv_path, index=False)

    context = {
        'input_path': str(hdr_path),
        'dims': f"{h}x{w}x{B}",
        'rgb': str(rgb_path),
        'pca': str(pca_path),
        'indices': {k: str(v) for k,v in indices_paths.items()},
        'zarr': str(zarr_dir),
        'cog': str(cog_path),
        'band_csv': str(csv_path)
    }
    html_path = outdir / 'reporte_dask.html'
    build_html_report(html_path, context)


if __name__ == '__main__':
    main()
