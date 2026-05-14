#!/usr/bin/env python3
"""
Pipeline ENVI (CPU) para imagen hiperespectral con exportación COG y reporte HTML.

- Lectura de .hdr/.bsq (ENVI) con rasterio (GDAL driver ENVI).
- Procesamiento por ventanas para bajo uso de memoria.
- Conversión a reflectancia física, máscara de NoData.
- Cálculo de NDVI, NDRE, PRI, PSRI, CIgreen.
- Quicklooks RGB (bandas por defecto) y PCA (muestra espacial).
- Filtrado de bandas por ventanas de absorción de agua.
- Exportación **COG multibanda** (tiled, DEFLATE, overviews internas).
- CSV de bandas válidas (wavelength/FWHM), máscara de vegetación y reportes.

Requisitos: python>=3.8, numpy, rasterio, matplotlib, scikit-learn, pandas (para CSV), (opcional) scikit-image para SLIC.
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from sklearn.decomposition import PCA

# ------------------------- Utilidades -------------------------


def setup_logger(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def parse_envi_hdr(hdr_path: Path) -> dict:
    """Parsea un archivo ENVI .hdr a un diccionario, manejando listas y números."""
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
    # bloques multilínea
    if 'wavelength' in meta and isinstance(meta['wavelength'], str):
        m = re.search(r'wavelength\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m:
            arr = [a.strip() for a in m.group(1).replace("\n", ' ').split(',') if a.strip()]
            meta['wavelength'] = arr
    if 'fwhm' in meta and isinstance(meta['fwhm'], str):
        m = re.search(r'fwhm\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE | re.DOTALL)
        if m:
            arr = [a.strip() for a in m.group(1).replace("\n", ' ').split(',') if a.strip()]
            meta['fwhm'] = arr
    return meta


def get_wavelengths(meta: dict) -> np.ndarray | None:
    w = meta.get('wavelength', None)
    if w is None:
        return None
    try:
        return np.array([float(x) for x in w], dtype=np.float32)
    except Exception:
        return None


def get_fwhm(meta: dict) -> np.ndarray | None:
    f = meta.get('fwhm', None)
    if f is None:
        return None
    try:
        return np.array([float(x) for x in f], dtype=np.float32)
    except Exception:
        return None


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.nanargmin(np.abs(wavelengths - target_nm)))


# ------------------------- Procesamiento -------------------------


def compute_indices(
    ds: rasterio.io.DatasetReader,
    wavelengths: np.ndarray | None,
    scale: float,
    ignore_values: list[float],
    out_dir: Path,
    block: int = 1024,
) -> dict[str, Path]:
    """Computa NDVI, NDRE, PRI, PSRI, CIgreen y los guarda como GeoTIFF."""

    def band_to_refl(bi: int, window: Window | None = None):
        arr = ds.read(bi + 1, window=window).astype(np.float32)
        mask = np.zeros_like(arr, dtype=bool)
        for v in ignore_values:
            mask |= arr == v
        arr = arr / scale
        arr[(arr < 0) | (arr > 1.2) | mask] = np.nan
        return arr

    if wavelengths is None:
        raise ValueError("Se requieren longitudes de onda en el .hdr para calcular índices con bandas específicas.")

    # Selección de bandas
    b_red = nearest_band(wavelengths, 660)
    b_nir = nearest_band(wavelengths, 800)
    b_re  = nearest_band(wavelengths, 720)
    b531  = nearest_band(wavelengths, 531)
    b570  = nearest_band(wavelengths, 570)
    b678  = nearest_band(wavelengths, 678)
    b500  = nearest_band(wavelengths, 500)
    b750  = nearest_band(wavelengths, 750)
    b554  = nearest_band(wavelengths, 554)

    logging.info(
        f"Bandas seleccionadas -> RED:{b_red} NIR:{b_nir} RE:{b_re} "
        f"531:{b531} 570:{b570} 678:{b678} 500:{b500} 750:{b750} 554:{b554}"
    )

    h, w = ds.height, ds.width
    profile = {
        'driver': 'GTiff', 'height': h, 'width': w, 'count': 1, 'dtype': 'float32',
        'crs': ds.crs, 'transform': ds.transform, 'compress': 'DEFLATE', 'tiled': True,
        'blockxsize': 512, 'blockysize': 512
    }

    paths = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'NDVI': out_dir / 'NDVI.tif',
        'NDRE': out_dir / 'NDRE.tif',
        'PRI':  out_dir / 'PRI.tif',
        'PSRI': out_dir / 'PSRI.tif',
        'CIgreen': out_dir / 'CIgreen.tif'
    }

    writers = {k: rasterio.open(str(p), 'w', **profile) for k, p in files.items()}

    try:
        for r0 in range(0, h, block):
            r1 = min(r0 + block, h)
            win = Window(col_off=0, row_off=r0, width=w, height=r1 - r0)
            RED = band_to_refl(b_red, win)
            NIR = band_to_refl(b_nir, win)
            RE  = band_to_refl(b_re,  win)
            B531 = band_to_refl(b531, win)
            B570 = band_to_refl(b570, win)
            B678 = band_to_refl(b678, win)
            B500 = band_to_refl(b500, win)
            B750 = band_to_refl(b750, win)
            B554 = band_to_refl(b554, win)

            NDVI = (NIR - RED) / (NIR + RED + 1e-6)
            NDRE = (NIR - RE)  / (NIR + RE  + 1e-6)
            PRI  = (B531 - B570) / (B531 + B570 + 1e-6)
            PSRI = (B678 - B500) / (B750 + 1e-6)
            CIg  = (NIR / (B554 + 1e-6)) - 1.0

            for k, arr in {'NDVI': NDVI, 'NDRE': NDRE, 'PRI': PRI, 'PSRI': PSRI, 'CIgreen': CIg}.items():
                writers[k].write(np.nan_to_num(arr, nan=-9999).astype('float32'), 1, window=win)
    finally:
        for wv in writers.values():
            wv.set_nodata(-9999)
            wv.close()

    for p in files.values():
        with rasterio.open(p, 'r+') as dst:
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')
    paths.update(files)
    return paths


def compute_veg_mask(ndvi_path: Path, threshold: float = 0.3) -> Path:
    out = ndvi_path.with_name('veg_mask.tif')
    with rasterio.open(ndvi_path) as src:
        profile = src.profile.copy()
        profile.update({'dtype': 'uint8'})
        with rasterio.open(out, 'w', **profile) as dst:
            for _ji, window in src.block_windows(1):
                nd = src.read(1, window=window)
                mask = (nd > threshold) & (nd != -9999)
                dst.write(mask.astype('uint8'), 1, window=window)
            dst.set_nodata(0)
    return out


def quicklook_rgb(
    ds: rasterio.io.DatasetReader,
    wavelengths: np.ndarray | None,
    default_bands: list[int] | None,
    scale: float,
    ignore_values: list[float],
    out_path: Path,
) -> Path:
    if default_bands:
        indices = [max(0, int(b) - 1) for b in default_bands]
    elif wavelengths is not None:
        r = nearest_band(wavelengths, 650)
        g = nearest_band(wavelengths, 560)
        b = nearest_band(wavelengths, 480)
        indices = [r, g, b]
    else:
        indices = [0, 1, 2]
    logging.info(f"Quicklook RGB usando bandas (0-based): {indices}")

    def band_to_refl(bi: int):
        arr = ds.read(bi + 1).astype(np.float32)
        m = np.zeros_like(arr, dtype=bool)
        for v in ignore_values:
            m |= arr == v
        arr = arr / scale
        arr[(arr < 0) | (arr > 1.2) | m] = np.nan
        return arr

    R = band_to_refl(indices[0])
    G = band_to_refl(indices[1])
    B = band_to_refl(indices[2])

    def stretch(a):
        lo, hi = np.nanpercentile(a, [1, 99])
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.dstack([stretch(R), stretch(G), stretch(B)])
    plt.figure(figsize=(7, 7))
    plt.imshow(np.nan_to_num(rgb))
    plt.axis('off')
    plt.title('Quicklook RGB')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def pca_quicklook(
    ds: rasterio.io.DatasetReader,
    wavelengths: np.ndarray | None,
    scale: float,
    ignore_values: list[float],
    out_path: Path,
    sample_rows: int = 300,
    block: int = 256,
    exclude_water: bool = True,
) -> Path:
    h, w, B = ds.height, ds.width, ds.count
    valid_bands = np.arange(B)
    if wavelengths is not None and exclude_water:
        water = ((wavelengths > 1340) & (wavelengths < 1440)) | ((wavelengths > 1800) & (wavelengths < 1950))
        valid_bands = np.where(~water)[0]
    logging.info(f"PCA: usando {len(valid_bands)} bandas válidas / {B}")

    rows = np.linspace(0, h - 1, min(sample_rows, h), dtype=int)
    X = []
    for r in rows:
        row_stack = []
        for bi in valid_bands:
            arr = ds.read(bi + 1, window=Window(0, r, w, 1)).astype(np.float32)
            m = np.zeros_like(arr, dtype=bool)
            for v in ignore_values:
                m |= arr == v
            arr = arr / scale
            arr[(arr < 0) | (arr > 1.2) | m] = np.nan
            row_stack.append(arr[0])
        row_stack = np.stack(row_stack, axis=-1)
        mask = ~np.any(~np.isfinite(row_stack), axis=1)
        X.append(row_stack[mask])
    X = np.vstack(X)
    logging.info(f"PCA: muestra con {X.shape[0]} pixeles x {X.shape[1]} bandas")

    pca = PCA(n_components=3, random_state=0)
    pca.fit(X)

    out = np.zeros((h, w, 3), dtype=np.float32)
    for r0 in range(0, h, block):
        r1 = min(h, r0 + block)
        block_stack = []
        for bi in valid_bands:
            arr = ds.read(bi + 1, window=Window(0, r0, w, r1 - r0)).astype(np.float32)
            m = np.zeros_like(arr, dtype=bool)
            for v in ignore_values:
                m |= arr == v
            arr = arr / scale
            arr[(arr < 0) | (arr > 1.2) | m] = np.nan
            block_stack.append(arr)
        block_stack = np.stack(block_stack, axis=-1)
        shp = block_stack.shape
        Xb = block_stack.reshape(-1, shp[-1])
        mask = ~np.any(~np.isfinite(Xb), axis=1)
        Y = np.zeros((Xb.shape[0], 3), dtype=np.float32)
        Y[mask] = pca.transform(Xb[mask])
        Y = Y.reshape(shp[0], shp[1], 3)
        for k in range(3):
            lo, hi = np.nanpercentile(Y[..., k], [1, 99])
            if hi - lo < 1e-6:
                Y[..., k] = 0
            else:
                Y[..., k] = np.clip((Y[..., k] - lo) / (hi - lo), 0, 1)
        out[r0:r1] = Y

    plt.figure(figsize=(7, 7))
    plt.imshow(np.nan_to_num(out))
    plt.axis('off')
    plt.title('PCA False Color (3 componentes)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def export_cog_multiband(
    ds: rasterio.io.DatasetReader,
    wavelengths: np.ndarray | None,
    scale: float,
    ignore_values: list[float],
    out_path: Path,
    exclude_water: bool = True,
    dtype: str = 'float32',
) -> tuple[Path, list[int]]:
    B = ds.count
    valid = np.arange(B)
    if wavelengths is not None and exclude_water:
        water = ((wavelengths > 1340) & (wavelengths < 1440)) | ((wavelengths > 1800) & (wavelengths < 1950))
        valid = np.where(~water)[0]
    logging.info(f"COG: exportando {len(valid)} bandas de {B} totales")

    profile = ds.profile.copy()
    profile.update({
        'driver': 'GTiff',
        'count': len(valid),
        'dtype': dtype,
        'compress': 'DEFLATE',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'interleave': 'band'
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        for out_bi, src_bi in enumerate(valid, start=1):
            for _ji, window in ds.block_windows(1):
                arr = ds.read(src_bi + 1, window=window).astype(np.float32)
                mask = np.zeros_like(arr, dtype=bool)
                for v in ignore_values:
                    mask |= arr == v
                arr = arr / scale
                arr[(arr < 0) | (arr > 1.2) | mask] = np.nan
                dst.write(np.nan_to_num(arr, nan=-9999).astype(dtype), out_bi, window=window)
        dst.set_nodata(-9999)

    with rasterio.open(out_path, 'r+') as dst:
        factors = [2, 4, 8, 16]
        dst.build_overviews(factors, Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')

    return out_path, valid.tolist()


def stats_nodata_by_band(ds: rasterio.io.DatasetReader, ignore_values: list[float]) -> list[float]:
    ratios = []
    for bi in range(ds.count):
        nodata_pixels = 0
        total = 0
        for _ji, window in ds.block_windows(1):
            arr = ds.read(bi + 1, window=window)
            mask = np.zeros_like(arr, dtype=bool)
            for v in ignore_values:
                mask |= arr == v
            nodata_pixels += int(mask.sum())
            total += arr.size
        ratios.append(nodata_pixels / max(1, total))
    return ratios


def write_csv_bandlist(
    out_csv: Path,
    exported_indices: list[int],
    wavelengths: np.ndarray | None,
    fwhm: np.ndarray | None,
):
    df = pd.DataFrame({'band_index_0based': exported_indices})
    if wavelengths is not None:
        df['wavelength_nm'] = [float(wavelengths[i]) for i in exported_indices]
    if fwhm is not None:
        df['fwhm_nm'] = [float(fwhm[i]) for i in exported_indices]
    df.to_csv(out_csv, index=False)


def build_html_report(out_path: Path, context: dict):
    html = ["<html><head><meta charset='utf-8'><title>Reporte hiperespectral</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:980px;margin:auto;padding:20px} img{max-width:100%}</style>",
            "</head><body>"]
    html.append("<h1>Reporte de procesamiento hiperespectral</h1>")
    html.append(f"<p><b>Archivo:</b> {context.get('input_path', '')}</p>")
    html.append(
        f"<p><b>Dimensiones:</b> {context.get('dims', '')}</p>"
        f" <p><b>CRS:</b> {context.get('crs', '')}</p>"
    )
    if 'nodata_ratio' in context:
        html.append("<h2>NoData por banda</h2>")
        html.append("<p>Promedio: %.2f%%</p>" % (100 * np.mean(context['nodata_ratio'])))
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(len(context['nodata_ratio'])), np.array(context['nodata_ratio']) * 100)
        plt.xlabel('Banda (0-based)')
        plt.ylabel('% NoData')
        plt.tight_layout()
        plot_path = Path(context['out_dir']) / 'qc_nodata.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        html.append(f"<img src='{plot_path.name}' alt='NoData por banda'>")
    if 'rgb_path' in context:
        html.append("<h2>Quicklook RGB</h2>")
        html.append(f"<img src='{Path(context['rgb_path']).name}' alt='Quicklook RGB'>")
    if 'pca_path' in context:
        html.append("<h2>PCA False Color</h2>")
        html.append(f"<img src='{Path(context['pca_path']).name}' alt='PCA FC'>")
    if 'indices' in context:
        html.append("<h2>Índices espectrales</h2><ul>")
        for k, p in context['indices'].items():
            html.append(f"<li>{k}: {Path(p).name}</li>")
        html.append("</ul>")
    if 'cog_path' in context:
        html.append("<h2>COG multibanda</h2>")
        html.append(f"<p>Archivo: {Path(context['cog_path']).name}</p>")
        if 'band_list_csv' in context:
            html.append(f"<p>Lista de bandas exportadas: {Path(context['band_list_csv']).name}</p>")
    html.append("<hr><p>Generado automáticamente.</p>")
    html.append("</body></html>")
    out_path.write_text("\n".join(html), encoding='utf-8')


# ------------------------- Main -------------------------


def main():
    ap = argparse.ArgumentParser(description='Pipeline ENVI (CPU) con COG y reporte HTML')
    ap.add_argument('--hdr', required=True, help='Ruta al archivo .hdr de ENVI')
    ap.add_argument('--outdir', required=True, help='Directorio de salida')
    ap.add_argument('--ndvi_threshold', type=float, default=0.3)
    ap.add_argument('--exclude_water', action='store_true', help='Excluir bandas en ventanas de absorción de agua')
    ap.add_argument('--default_rgb', nargs='*', help='Bandas RGB por defecto del header (1-based), ej: 79 48 18')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logger(outdir / 'pipeline.log')

    hdr_path = Path(args.hdr)
    meta = parse_envi_hdr(hdr_path)
    wavelengths = get_wavelengths(meta)
    fwhm = get_fwhm(meta)

    scale = float(meta.get('reflectance scale factor', 10000.0))
    ignore_values = []
    if 'data ignore value' in meta:
        try:
            ignore_values.append(float(meta['data ignore value']))
        except Exception:
            pass
    if 'background' in meta:
        try:
            ignore_values.append(float(meta['background']))
        except Exception:
            pass
    logging.info(f"Scale={scale} Ignore={ignore_values}")

    with rasterio.open(hdr_path) as ds:
        h, w, B = ds.height, ds.width, ds.count
        context = {
            'input_path': str(hdr_path),
            'dims': f"{h}x{w}x{B}",
            'crs': str(ds.crs),
            'out_dir': str(outdir)
        }
        logging.info(f"Dataset: {h}x{w}x{B} CRS={ds.crs}")

        nodata_ratio = stats_nodata_by_band(ds, ignore_values)
        context['nodata_ratio'] = nodata_ratio

        default_bands = None
        if args.default_rgb:
            default_bands = [int(x) for x in args.default_rgb]
        elif 'default bands' in meta:
            try:
                default_bands = [int(x) for x in meta['default bands']]
            except Exception:
                default_bands = None
        rgb_path = quicklook_rgb(
            ds, wavelengths, default_bands, scale, ignore_values, outdir / 'quicklook_rgb.png'
        )
        context['rgb_path'] = str(rgb_path)

        pca_path = pca_quicklook(
            ds, wavelengths, scale, ignore_values, outdir / 'quicklook_pca.png',
            exclude_water=args.exclude_water,
        )
        context['pca_path'] = str(pca_path)

        indices_paths = compute_indices(ds, wavelengths, scale, ignore_values, outdir / 'indices')
        context['indices'] = {k: str(v) for k, v in indices_paths.items()}

        veg_path = compute_veg_mask(indices_paths['NDVI'], threshold=args.ndvi_threshold)
        context['veg_mask'] = str(veg_path)

        cog_path, exported = export_cog_multiband(
            ds, wavelengths, scale, ignore_values,
            outdir / 'cog' / 'reflectance_valid_bands.tif',
            exclude_water=args.exclude_water,
        )
        context['cog_path'] = str(cog_path)

    csv_path = outdir / 'cog' / 'reflectance_valid_bands.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_bandlist(csv_path, exported, wavelengths, fwhm)
    context['band_list_csv'] = str(csv_path)

    build_html_report(outdir / 'reporte.html', context)


if __name__ == '__main__':
    main()
