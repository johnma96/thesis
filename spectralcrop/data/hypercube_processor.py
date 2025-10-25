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


    def _open_zarr_group_compat(self, zarr_path: str, overwrite: bool = True):
        """
        Abre/crea un grupo Zarr en 'zarr_path' compatible con Zarr v2 y v3.
        Retorna (root, store, zarr_major) y borra la ruta si overwrite=True.
        """
        import os, shutil
        import zarr

        # Limpieza si se pide overwrite
        if overwrite and os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)

        # 1) Intento: Zarr v2 toplevel
        if hasattr(zarr, 'DirectoryStore'):
            try:
                store = zarr.DirectoryStore(zarr_path)
                root = zarr.group(store=store, overwrite=True)
                return root, store, 2
            except Exception:
                pass

        # 2) Intento: Zarr v2 desde zarr.storage
        try:
            from zarr.storage import DirectoryStore as _DirStore
            store = _DirStore(zarr_path)
            root = zarr.group(store=store, overwrite=True)
            return root, store, 2
        except Exception:
            pass

        # 3) Intento: Zarr v3 FSStore
        try:
            from zarr.storage import FSStore as _FSStore
            store = _FSStore(zarr_path, mode='w')  # crea si no existe
            # En v3, group(...) no acepta overwrite; el store ya está limpio por 'mode=w'
            root = zarr.group(store=store)
            return root, store, 3
        except Exception:
            pass

        # 4) Último recurso: fsspec mapper (sirve en v2/v3)
        try:
            import fsspec
            store = fsspec.get_mapper(zarr_path)
            try:
                root = zarr.group(store=store, overwrite=True)  # v2
            except TypeError:
                root = zarr.group(store=store)                  # v3
            return root, store, None
        except Exception as e:
            raise RuntimeError(
                "No se pudo abrir/crear un Zarr store compatible (v2/v3). "
                "Considera fijar versión con 'pip install \"zarr<3\"'."
            ) from e

    def save_reflectance_to_zarr(self,
                                zarr_path: str,
                                exclude_water: bool = True,
                                water_windows=None,      # si None, usa self.water_windows
                                min_lambda: float = None,
                                max_lambda: float = None,
                                chunks=(64, 512, 512),   # (band, y, x)
                                add_coords: bool = True,
                                add_ndvi: bool = True,
                                ndvi_threshold: float = 0.3) -> None:
        """
        Guarda reflectance (band,y,x) + coords + (opcional) NDVI/veg_mask en zarr_path,
        usando la API de alto nivel recomendada por Zarr:
        - zarr.open_group(<ruta>, mode="w")
        - root.create_array(...)
        """
        import os
        import shutil
        import numpy as np
        import zarr

        # 0) Limpiar destino si existe
        print('[INFO] Guardando reflectance en Zarr...')
        if os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)

        B, H, W = self.img.nbands, self.img.nrows, self.img.ncols

        # 1) Selección de bandas válidas
        print('[INFO] Seleccionando bandas válidas...')
        valid_bands = self.build_valid_bands(
            wavelengths=self.wavelengths,
            water_windows=water_windows,
            exclude_water=exclude_water,
            min_lambda=min_lambda,
            max_lambda=max_lambda
        )
        if valid_bands is None:
            valid_bands = np.arange(B, dtype=int)
        valid_bands = np.asarray(valid_bands, dtype=int)
        B_out = int(len(valid_bands))

        # 2) Abrir/crear grupo en disco (recomendado por docs oficiales)
        #    Crea el directorio <zarr_path> y organiza arrays como sub-rutas.
        print('[INFO] Creando Zarr group en disco...')
        root = zarr.open_group(zarr_path, mode='w')  # ← clave: API estable por ruta

        # 3) Crear array "reflectance" (band, y, x)
        print('[INFO] Creando array "reflectance"...')
        arr_reflect = root.create_array(
            name='reflectance',
            shape=(B_out, H, W),
            chunks=chunks,
            dtype='float32'
            # No pasamos "compressor": usamos el codec por defecto de Zarr
        )

        # 4) Escribir banda por banda
        print('[INFO] Escribiendo bandas de reflectance...')
        for i, bi in enumerate(valid_bands):
            refl = self.to_reflectance(
                band_i=int(bi),
                clip=(0.0, 1.2),
                skip_if_water=True,           # redundante tras filtrar, pero seguro
                wavelengths=self.wavelengths, # permitido por tu firma actual
                water_windows=water_windows
            )
            arr_reflect[i, :, :] = refl  # NaN se almacenan tal cual

        # 5) Atributos del grupo
        print('[INFO] Agregando atributos al grupo Zarr...')
        root.attrs.put({
            'description': 'Reflectance cube (raw) without Savitzky–Golay',
            'units': 'unitless',
            'scale_applied': float(self.scale),
            'clip_applied': '[0,1.2]',
            'nodata': 'NaN',
            'exclude_water_windows': bool(exclude_water),
            'water_windows_nm': tuple(tuple(map(float, w)) for w in (self.water_windows if water_windows is None else water_windows)),
        })

        # 6) Arrays "coord-like"
        #    Usamos create_array + asignación por consistencia entre versiones.
        print('[INFO] Agregando arrays de coordenadas...')
        root.create_array('band', shape=(B_out,), chunks=(min(B_out, max(1, chunks[0])),), dtype='int32')[:] = np.arange(B_out, dtype=np.int32)
        if add_coords and (self.wavelengths is not None):
            wl_valid = self.wavelengths[valid_bands].astype('float32')
            root.create_array('wavelength', shape=wl_valid.shape, chunks=(wl_valid.shape[0],), dtype='float32')[:] = wl_valid
            if (self.fwhm is not None) and (len(self.fwhm) == len(self.wavelengths)):
                fw_valid = self.fwhm[valid_bands].astype('float32')
                root.create_array('fwhm', shape=fw_valid.shape, chunks=(fw_valid.shape[0],), dtype='float32')[:] = fw_valid

        # 7) NDVI + veg_mask (opcional)
        print('[INFO] Agregando NDVI y veg_mask (opcional)...')
        if add_ndvi and (self.wavelengths is not None):
            # Buscar bandas cercanas dentro de lo exportado
            wl_export = self.wavelengths[valid_bands]
            def nb(target):
                return int(np.nanargmin(np.abs(wl_export - float(target))))
            try:
                b_red = nb(660.0)
                b_nir = nb(800.0)

                RED = arr_reflect[b_red, :, :].astype(np.float32)
                NIR = arr_reflect[b_nir, :, :].astype(np.float32)
                NDVI = (NIR - RED) / (NIR + RED + 1e-6)
                veg_mask = (NDVI > float(ndvi_threshold)) & np.isfinite(NDVI)

                # Chunks espaciales para 2D
                yx_chunks = (chunks[1], chunks[2]) if len(chunks) == 3 else None

                root.create_array('NDVI', shape=(H, W), chunks=yx_chunks, dtype='float32')[:] = NDVI.astype('float32')
                vm = root.create_array('veg_mask', shape=(H, W), chunks=yx_chunks, dtype='uint8')
                vm[:] = veg_mask.astype('uint8')
                vm.attrs.update({'0': 'no-veg', '1': 'veg'})
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