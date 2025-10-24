# -*- coding: utf-8 -*-
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc


class HypercubeProcessor:
    """
    Procesador POO para un cubo hiperespectral preprocesado (ENVI -> reflectancia).

    Parámetros
    ----------
    img : objeto con atributos (nbands, nrows, ncols)
        Metadatos geométricos del cubo (p. ej., lector ENVI o similar).
    cube : objeto con método read_band(i) -> np.ndarray
        Proveedor de lectura de bandas (BSQ/ENVI u otro).
    wavelengths : array-like, opcional
        Longitudes de onda por banda (nm). Se convierte a float32.
    fwhm : array-like, opcional
        Ancho de banda (nm). Se convierte a float32.
    scale : float
        Factor de escala para convertir a reflectancia (p. ej. 10000.0).
    ignore_vals : Sequence[float|int]
        Valores a tratar como máscara (p. ej., background, data ignore).
    water_windows : Tuple[Tuple[float,float], Tuple[float,float]]
        Ventanas de absorción de agua (nm) a excluir.
    """

    def __init__(self,
                 img,
                 cube,
                 wavelengths: Optional[Sequence] = None,
                 fwhm: Optional[Sequence] = None,
                 scale: float = 10000.0,
                 ignore_vals: Sequence = (-1.0, 15000.0),
                 water_windows: Tuple[Tuple[float, float], Tuple[float, float]] = ((1340, 1440), (1800, 1950))):
        self.img = img
        self.cube = cube
        self.scale = float(scale)
        self.ignore_vals = tuple(ignore_vals)
        self.water_windows = tuple(tuple(map(float, w)) for w in water_windows)

        # Parseo seguro a float32 si vienen como strings u objetos
        self.wavelengths = None
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths, dtype=np.float32)

        self.fwhm = None
        if fwhm is not None:
            self.fwhm = np.asarray(fwhm, dtype=np.float32)

    # -------------------------------
    # 1) Helpers para bandas de agua
    # -------------------------------

    @staticmethod
    def _is_in_windows(value: float,
                       windows: Tuple[Tuple[float, float], ...]) -> bool:
        """Retorna True si 'value' cae estrictamente dentro de alguna ventana (lo, hi)."""
        for lo, hi in windows:
            if (value > lo) and (value < hi):
                return True
        return False

    def build_valid_bands(self,
                          wavelengths: Optional[np.ndarray] = None,
                          water_windows: Optional[Tuple[Tuple[float, float], ...]] = None,
                          exclude_water: bool = True,
                          min_lambda: Optional[float] = None,
                          max_lambda: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Retorna índices 0-based de bandas válidas según:
        - excluir ventanas de agua,
        - recorte espectral opcional (min/max λ).
        Si no hay wavelengths -> retorna None (equivalente a 'todas').
        """
        wl = self.wavelengths if wavelengths is None else np.asarray(wavelengths, dtype=np.float32)
        if wl is None:
            return None

        windows = self.water_windows if water_windows is None else tuple(tuple(map(float, w)) for w in water_windows)

        B = len(wl)
        valid = np.arange(B, dtype=int)

        if exclude_water:
            keep = np.ones(B, dtype=bool)
            for i, w in enumerate(wl):
                if self._is_in_windows(float(w), windows):
                    keep[i] = False
            valid = valid[keep]

        if min_lambda is not None:
            valid = valid[wl[valid] >= float(min_lambda)]
        if max_lambda is not None:
            valid = valid[wl[valid] <= float(max_lambda)]

        return valid

    # -------------------------------------------------------
    # 2) to_reflectance con saltos de bandas de agua (opcional)
    # -------------------------------------------------------

    def to_reflectance(self,
                       band_i: int,
                       clip: Tuple[float, float] = (0.0, 1.2),
                       ignore_vals: Optional[Sequence] = None,
                       skip_if_water: bool = True,
                       wavelengths: Optional[np.ndarray] = None,
                       water_windows: Optional[Tuple[Tuple[float, float], ...]] = None) -> np.ndarray:
        """
        Convierte una banda a reflectancia física:
        - Aplica máscara (background, data ignore).
        - Clip a [clip].
        - Si skip_if_water=True y λ cae en ventanas de agua, devuelve NaN sin leer disco.
        """
        wl = self.wavelengths if wavelengths is None else np.asarray(wavelengths, dtype=np.float32)
        windows = self.water_windows if water_windows is None else tuple(tuple(map(float, w)) for w in water_windows)
        ignore = tuple(self.ignore_vals) if ignore_vals is None else tuple(ignore_vals)

        if skip_if_water and (wl is not None):
            wli = float(wl[band_i])
            if self._is_in_windows(wli, windows):
                return np.full((self.img.nrows, self.img.ncols), np.nan, dtype=np.float32)

        band = self.cube.read_band(int(band_i)).astype(np.float32)

        # Máscara por valores a ignorar
        mask = np.zeros_like(band, dtype=bool)
        for v in ignore:
            mask |= (band == v)

        lo, hi = map(float, clip)
        refl = band / self.scale
        # Clip + máscara -> NaN
        bad = (refl < lo) | (refl > hi) | mask
        refl = refl.astype(np.float32)
        refl[bad] = np.nan
        return refl

    # -------------------------------------------------------
    # 3) Guardar reflectancia cruda en Zarr (streaming)
    # -------------------------------------------------------

    def save_reflectance_to_zarr(self,
                                 zarr_path: str,
                                 exclude_water: bool = True,
                                 water_windows: Optional[Tuple[Tuple[float, float], ...]] = None,
                                 min_lambda: Optional[float] = None,
                                 max_lambda: Optional[float] = None,
                                 chunks: Tuple[int, int, int] = (64, 512, 512),
                                 compressor: Optional[str] = 'zstd',
                                 clevel: int = 5,
                                 add_coords: bool = True,
                                 add_ndvi: bool = False,
                                 ndvi_threshold: float = 0.3) -> None:
        """
        Escribe el cubo de reflectancia cruda en Zarr como variable 'reflectance'.
        - Excluye bandas en ventanas de agua (opcional).
        - Añade coords 'wavelength' y 'fwhm' (si existen).
        - (Opcional) Guarda NDVI y veg_mask en el mismo Zarr.

        chunks: (band, y, x) recomendado para acceso eficiente.
        """
        os.makedirs(os.path.dirname(zarr_path), exist_ok=True)

        B, H, W = self.img.nbands, self.img.nrows, self.img.ncols

        # Determinar bandas a exportar
        valid_bands = self.build_valid_bands(
            wavelengths=self.wavelengths,
            water_windows=water_windows,
            exclude_water=exclude_water,
            min_lambda=min_lambda,
            max_lambda=max_lambda
        )
        if valid_bands is None:
            valid_bands = np.arange(B, dtype=int)
        B_out = int(len(valid_bands))

        # Crear el Zarr
        store = zarr.DirectoryStore(zarr_path)
        cname = 'zstd' if compressor is None else compressor
        zcompressor = Blosc(cname=cname, clevel=int(clevel), shuffle=Blosc.SHUFFLE)

        root = zarr.group(store=store, overwrite=True)
        zarr_arr = root.create(
            name='reflectance',
            shape=(B_out, H, W),
            chunks=chunks,
            dtype='float32',
            compressor=zcompressor
        )

        # Escribir banda por banda
        for i, bi in enumerate(valid_bands):
            refl = self.to_reflectance(
                int(bi),
                clip=(0.0, 1.2),
                skip_if_water=True,  # redundante (ya filtramos), pero seguro
                wavelengths=self.wavelengths,
                water_windows=water_windows
            )
            zarr_arr[i, :, :] = refl  # NaN se almacenan tal cual

        # Metadatos/attrs
        root.attrs['attrs'] = {
            'description': 'Reflectance cube (raw) without Savitzky–Golay',
            'units': 'unitless',
            'scale_applied': float(self.scale),
            'clip_applied': '[0,1.2]',
            'nodata': 'NaN',
            'exclude_water_windows': bool(exclude_water),
            'water_windows_nm': tuple(tuple(map(float, w)) for w in (self.water_windows if water_windows is None else water_windows)),
        }

        # Datasets “coord-like” para fácil lectura con xarray
        root.create_dataset('band', data=np.arange(B_out, dtype=np.int32))
        if add_coords:
            if self.wavelengths is not None:
                root.create_dataset('wavelength', data=self.wavelengths[valid_bands].astype('float32'))
            if (self.fwhm is not None) and (self.wavelengths is not None) and (len(self.fwhm) == len(self.wavelengths)):
                root.create_dataset('fwhm', data=self.fwhm[valid_bands].astype('float32'))

        # (Opcional) añadir NDVI y veg_mask en una segunda pasada con xarray
        if add_ndvi and (self.wavelengths is not None):
            try:
                ds = xr.open_zarr(zarr_path)
                da = ds['reflectance']  # (band,y,x)

                wl_valid = ds['wavelength'].values if 'wavelength' in ds.variables else None
                if wl_valid is not None:
                    # bandas más cercanas a 660nm y 800nm
                    def nb(target):
                        return int(np.nanargmin(np.abs(wl_valid - float(target))))
                    b_red = nb(660.0)
                    b_nir = nb(800.0)

                    RED = da.isel(band=b_red).values
                    NIR = da.isel(band=b_nir).values
                    NDVI = (NIR - RED) / (NIR + RED + 1e-6)
                    veg_mask = (NDVI > float(ndvi_threshold)) & np.isfinite(NDVI)

                    ds2 = xr.Dataset({
                        'reflectance': da,
                        'NDVI': xr.DataArray(NDVI.astype('float32'), dims=('y', 'x')),
                        'veg_mask': xr.DataArray(veg_mask.astype('uint8'), dims=('y', 'x'),
                                                 attrs={'0': 'no-veg', '1': 'veg'})
                    })
                    # Mantener chunks razonables
                    ds2 = ds2.chunk({'band': da.chunks[0][0] if da.chunks else 64, 'y': 512, 'x': 512})
                    ds2.to_zarr(zarr_path, mode='w')
            except Exception as e:
                print(f"[WARN] NDVI/veg_mask no agregados: {e}")

        print(f"[OK] Zarr guardado en: {zarr_path}")
        print(f"    Bandas exportadas: {B_out} / {B}")
        if self.wavelengths is not None:
            wl_exp = self.wavelengths[valid_bands]
            print(f"    Rango λ exportado: {float(np.nanmin(wl_exp)):.1f}–{float(np.nanmax(wl_exp)):.1f} nm")

    def nearest_band(self, target_nm: float) -> int:
        assert self.wavelengths is not None, "No hay wavelengths."
        return int(np.nanargmin(np.abs(self.wavelengths - float(target_nm))))

    def quicklook_rgb(self, rgb_nm=(650, 560, 480), clip=(1, 99), skip_if_water=False):
        """Retorna un RGB normalizado (H, W, 3) listo para plt.imshow."""
        assert self.wavelengths is not None, "No hay 'wavelengths' en el processor."
        idx = [self.nearest_band(nm) for nm in rgb_nm]
        chans = []
        for bi in idx:
            arr = self.to_reflectance(bi, clip=(0.0, 1.2), skip_if_water=skip_if_water)
            lo, hi = np.nanpercentile(arr, clip)
            chans.append(np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1))
        return np.dstack(chans)

    
    def compute_ndvi(self,
                    red_nm: float = 660.0,
                    nir_nm: float = 800.0,
                    threshold: float = 0.3):
        """
        Retorna NDVI (H,W) y veg_mask (H,W) usando self.water_windows y self.wavelengths.
        """
        assert self.wavelengths is not None, "No hay 'wavelengths' en el processor."

        rb = self.nearest_band(red_nm)
        nb = self.nearest_band(nir_nm)

        RED = self.to_reflectance(
            band_i=rb,
            clip=(0.0, 1.2),
            skip_if_water=True  # usa self.water_windows internamente
        )
        NIR = self.to_reflectance(
            band_i=nb,
            clip=(0.0, 1.2),
            skip_if_water=True  # usa self.water_windows internamente
        )

        NDVI = (NIR - RED) / (NIR + RED + 1e-6)
        veg_mask = (NDVI > float(threshold)) & np.isfinite(NDVI)

        return NDVI.astype(np.float32), veg_mask.astype(np.uint8)


    def band_stats_on_mask(self, band_indices, mask):
        """Devuelve arrays (means, stds) sobre los pixeles True del mask."""
        means, stds = [], []
        for bi in band_indices:
            arr = self.to_reflectance(int(bi), clip=(0.0, 1.2), skip_if_water=True)
            vals = arr[mask & np.isfinite(arr)]
            if vals.size > 0:
                means.append(float(np.nanmean(vals)))
                stds.append(float(np.nanstd(vals)))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        return np.array(means), np.array(stds)