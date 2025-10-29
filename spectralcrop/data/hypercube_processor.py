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

    def _read_bands_block(self, bands_idx, yslice=None, xslice=None) -> np.ndarray:
        """
        Lee un bloque desde 'cube' (spectral ENVI memmap) y lo devuelve como (B_k, h, w).
        - bands_idx: iterable de índices de banda (int)
        - yslice, xslice: slices espaciales (e.g., slice(y0,y1), slice(x0,x1))
        """
        bands_idx = [int(b) for b in bands_idx]
        if yslice is None:
            yslice = slice(None)
        if xslice is None:
            xslice = slice(None)

        # spectral memmap es (H, W, B). Obtenemos (h, w, B_k) y lo movemos a (B_k, h, w).
        raw = self.cube[yslice, xslice, bands_idx]  # memmap -> lectura perezosa
        if raw.ndim == 2:
            raw = raw[..., None]  # (h,w,1) por si bands_idx tiene 1 banda
        return np.moveaxis(raw, 2, 0)  # (B_k, h, w)

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

    # ---------------------------------------------
    # Helpers para lectura vectorizada desde SPECTRAL
    # ---------------------------------------------
    def _read_bands_block(self, bands_idx, yslice=None, xslice=None) -> np.ndarray:
        """
        Lee un bloque desde 'cube' (spectral ENVI memmap) y lo devuelve como (B_k, h, w).
        - bands_idx: iterable de índices de banda (int)
        - yslice, xslice: slices espaciales (e.g., slice(y0,y1), slice(x0,x1))
        """
        bands_idx = [int(b) for b in bands_idx]
        if yslice is None:
            yslice = slice(None)
        if xslice is None:
            xslice = slice(None)

        # El memmap de spectral es (H, W, B)
        raw = self.cube[yslice, xslice, bands_idx]
        if raw.ndim == 2:  # una sola banda
            raw = raw[..., None]
        return np.moveaxis(raw, 2, 0)  # -> (B_k, h, w)


    def _write_reflectance_blockwise_bands(self,
                                        arr_reflect,
                                        valid_bands,
                                        water_windows=None,
                                        clip=(0.0, 1.2),
                                        band_block=4,
                                        progress=None):
        """
        Escribe reflectancia en bloques de bandas (espacio completo):
        - Menos overhead Python (≈ B_out / band_block iteraciones).
        - Operaciones vectorizadas 3D.
        """
        import numpy as np

        lo, hi = map(float, clip)
        ignore_vals = tuple(self.ignore_vals)
        wl = self.wavelengths
        windows = self.water_windows if water_windows is None else tuple(tuple(map(float, w)) for w in water_windows)

        H, W = self.img.nrows, self.img.ncols
        out_index = {int(b): i for i, b in enumerate(valid_bands)}

        def in_water(b):
            if wl is None:
                return False
            wli = float(wl[int(b)])
            return any((wli > lo_w) and (wli < hi_w) for lo_w, hi_w in windows)

        water_bands = [int(b) for b in valid_bands if in_water(b)]
        work_bands  = [int(b) for b in valid_bands if not in_water(b)]

        # Rellena bandas de agua sin leer disco
        if water_bands:
            arr_reflect[[out_index[b] for b in water_bands], :, :] = np.nan

        iterator = range(0, len(work_bands), band_block)
        if progress is not None:
            iterator = progress(iterator, total=(len(work_bands) + band_block - 1) // band_block, desc="Batches(bands)")

        for s in iterator:
            batch = work_bands[s:s+band_block]  # lista de bandas fuente
            raw = self._read_bands_block(batch)  # (B_k, H, W), dtype origen

            # Máscara ignore_vals (vectorizada)
            mask_ignore = np.zeros_like(raw, dtype=bool)
            for v in ignore_vals:
                mask_ignore |= (raw == v)

            # Escalado + filtro
            refl = raw.astype(np.float32, copy=True)
            refl /= float(self.scale)
            bad = (refl < lo) | (refl > hi) | mask_ignore
            refl[bad] = np.nan

            # Escritura a Zarr
            for j, bsrc in enumerate(batch):
                arr_reflect[out_index[bsrc], :, :] = refl[j]

    def _write_reflectance_chunkwise_tiled(self,
                                        arr_reflect,
                                        valid_bands,
                                        water_windows=None,
                                        clip=(0.0, 1.2),
                                        tile=(512, 512),
                                        progress=None):
        """
        Escritura tileada (y,x) PERO agrupando bandas por el tamaño del chunk en C.
        Así cada archivo de chunk (c_idx, y_idx, x_idx) se escribe una única vez por tile.

        - Requiere que 'arr_reflect' exponga su shape/chunks (Zarr Array).
        - 'valid_bands' define el orden de salida ==> los grupos c0:c1 son cortados sobre ese orden.
        """
        import numpy as np

        lo, hi = map(float, clip)
        ignore_vals = tuple(self.ignore_vals)
        wl = self.wavelengths
        windows = self.water_windows if water_windows is None else tuple(tuple(map(float, w)) for w in water_windows)

        H, W = self.img.nrows, self.img.ncols
        Ty, Tx = tile

        # Tamaño del chunk sobre el eje banda (C)
        try:
            c_chunk = arr_reflect.chunks[0]
        except Exception:
            # fallback al c_chunk que definiste al crear el array
            c_chunk = None
        if not c_chunk:
            raise ValueError("No se pudo determinar c_chunk; asegúrate de crear el Zarr con chunks=(C,Y,X).")

        B_out = len(valid_bands)

        # Helper: ¿banda está en ventana de agua?
        def in_water(b):
            if wl is None:
                return False
            wli = float(wl[int(b)])
            return any((wli > lo_w) and (wli < hi_w) for lo_w, hi_w in windows)

        # Iterar tiles espaciales
        tiles_y = (H + Ty - 1) // Ty
        tiles_x = (W + Tx - 1) // Tx
        total_steps = tiles_y * tiles_x * ((B_out + c_chunk - 1) // c_chunk)

        # Wrapper de progreso opcional
        class _Prog:
            def __init__(self, bar=None): self.bar = bar
            def update(self, n=1): 
                if self.bar is not None: self.bar.update(n)

        pwrap = _Prog(progress)

        for y0 in range(0, H, Ty):
            y1 = min(H, y0 + Ty); ysl = slice(y0, y1)
            for x0 in range(0, W, Tx):
                x1 = min(W, x0 + Tx); xsl = slice(x0, x1)

                # Grupos contiguos en el eje banda acorde al chunk en C
                for c0 in range(0, B_out, c_chunk):
                    c1 = min(B_out, c0 + c_chunk)
                    group_bands = valid_bands[c0:c1]               # ids de bandas fuente (en ENVI)
                    # Lectura vectorizada del memmap para ESTE tile y ESTE grupo de bandas
                    raw = self._read_bands_block(group_bands, yslice=ysl, xslice=xsl)  # (c_len, Ty', Tx')

                    # Construir máscara ignore y agua (todo vectorizado)
                    mask_ignore = np.zeros_like(raw, dtype=bool)
                    for v in ignore_vals:
                        mask_ignore |= (raw == v)

                    if wl is not None:
                        # marcas de agua por banda -> broadcasting a (c_len, Ty', Tx')
                        water_flags = np.array([in_water(b) for b in group_bands], dtype=bool)[:, None, None]
                    else:
                        water_flags = np.zeros((raw.shape[0], 1, 1), dtype=bool)

                    # Escalado + clip + NaN
                    refl = raw.astype(np.float32, copy=True)
                    refl /= float(self.scale)
                    bad = (refl < lo) | (refl > hi) | mask_ignore | water_flags
                    refl[bad] = np.nan

                    # ESCRITURA 3D DE UNA SOLA VEZ PARA ESTE CHUNK EN C:
                    arr_reflect[c0:c1, ysl, xsl] = refl

                    pwrap.update(1)

    def save_reflectance_to_zarr_fast(self,
                                    zarr_path: str,
                                    exclude_water: bool = True,
                                    water_windows=None,         # si None, usa self.water_windows
                                    min_lambda: float = None,
                                    max_lambda: float = None,
                                    chunks=(64, 512, 512),
                                    add_coords: bool = True,
                                    add_ndvi: bool = True,
                                    ndvi_threshold: float = 0.3,
                                    strategy: str = "tiled",    # "tiled" | "bands"
                                    band_block: int = 4,
                                    tile: tuple = (512, 512),
                                    resume: bool = False,       # reanudar escrituras
                                    atomic_swap: bool = False   # escribe en .tmp y renombra al final
                                    ) -> None:
        """
        Versión acelerada para escribir 'reflectance' (band,y,x) en Zarr con lectura vectorizada desde
        'spectral' (memmap ENVI) y cálculo por bloques de bandas y/o tiles espaciales.

        - strategy="bands": bloques de k bandas sobre toda la escena.
        - strategy="tiled": tiles (Ty,Tx) × grupos chunk-aware en C (recomendado si chunks=(...,512,512)).
        - resume=True: usa 'written_bands' para saltar escrituras.
        - atomic_swap=True: escribe en zarr_path+'.tmp' y renombra al final (consistencia atómica).

        Además añade metadatos de dimensiones requeridos por xarray:
        - reflectance: ['band', 'y', 'x']
        - band, wavelength, fwhm, written_bands: ['band']
        - NDVI, veg_mask: ['y', 'x']
        """
        import os
        import shutil
        import numpy as np
        import zarr

        print('[INFO] Iniciando guardado acelerado de reflectancia en Zarr...')

        # tqdm (opcional)
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        # Helper local para setear atributos de dimensiones (xarray-friendly)
        def _set_dim_attrs(zarr_array, dims):
            try:
                zarr_array.attrs['dimension_names'] = list(dims)
                zarr_array.attrs['_ARRAY_DIMENSIONS'] = list(dims)
            except Exception:
                # En caso de Zarr v2/v3 sin soporte de attrs.put directamente
                zarr_array.attrs.update({'dimension_names': list(dims),
                                        '_ARRAY_DIMENSIONS': list(dims)})

        # Validaciones básicas
        H, W, Bm = self.cube.shape
        assert (H == self.img.nrows) and (W == self.img.ncols), "Dimensiones memmap vs img no coinciden."
        assert Bm == self.img.nbands, "Bandas memmap vs img no coinciden."

        # 0) Selección de bandas válidas
        print("[INFO] Construyendo lista de bandas válidas...")
        valid_bands = self.build_valid_bands(
            wavelengths=self.wavelengths,
            water_windows=water_windows,
            exclude_water=exclude_water,
            min_lambda=min_lambda,
            max_lambda=max_lambda
        )
        if valid_bands is None:
            valid_bands = np.arange(self.img.nbands, dtype=int)
        valid_bands = np.asarray(valid_bands, dtype=int)
        B_out = int(len(valid_bands))

        # 1) Resolver destino (atomic swap opcional)
        print(f"[INFO] Preparando Zarr en: {zarr_path} (atomic_swap={atomic_swap})")
        target_path = zarr_path
        tmp_path = None
        if atomic_swap:
            tmp_path = zarr_path + ".tmp"
            # Limpiar solo la temporal
            if os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
            # Con atomic swap escribimos limpio
            resume = False
            target_path = tmp_path

        # 2) Abrir/crear grupo
        print("[INFO] Abriendo/creando Zarr group...")
        if resume:
            # modo append
            root = zarr.open_group(target_path, mode='a')
        else:
            # overwrite limpio
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            root = zarr.open_group(target_path, mode='w')  # Guía oficial: abrir grupo por ruta

        # 3) Crear/obtener array 'reflectance'
        print("[INFO] Preparando array 'reflectance'...")
        if 'reflectance' in root:
            arr_reflect = root['reflectance']
            if arr_reflect.shape != (B_out, H, W):
                raise ValueError(f"'reflectance' existente con shape {arr_reflect.shape}, esperado {(B_out, H, W)}")
            # Asegurar metadatos de dimensiones (por si faltaban)
            _set_dim_attrs(arr_reflect, ['band', 'y', 'x'])
        else:
            try:
                arr_reflect = root.create_array(
                    name='reflectance',
                    shape=(B_out, H, W),
                    chunks=chunks,
                    dtype='float32',
                    dimension_names=['band', 'y', 'x']
                )
                _set_dim_attrs(arr_reflect, ['band', 'y', 'x'])
            except:
                arr_reflect = root.create_array(
                    name='reflectance',
                    shape=(B_out, H, W),
                    chunks=chunks,
                    dtype='float32'
                )
                _set_dim_attrs(arr_reflect, ['band', 'y', 'x'])

        # 4) 'written_bands' (control de resume)
        print("[INFO] Preparando control de bandas escritas...")
        if resume:
            if 'written_bands' not in root:
                try:
                    wb = root.create_array('written_bands', shape=(B_out,),
                                        chunks=(min(B_out, 1024),), dtype='uint8',
                                        dimension_names=['band'])
                    wb[:] = 0
                    _set_dim_attrs(wb, ['band'])
                except:
                    wb = root.create_array('written_bands', shape=(B_out,),
                                        chunks=(min(B_out, 1024),), dtype='uint8')
                    wb[:] = 0
                    _set_dim_attrs(wb, ['band'])
            else:
                wb = root['written_bands']
                # Dim attrs por si faltaban
                _set_dim_attrs(wb, ['band'])
        else:
            # Metadatos del grupo (attrs raíz)
            root.attrs.put({
                'description': 'Reflectance cube (raw)',
                'units': 'unitless',
                'scale_applied': float(self.scale),
                'clip_applied': '[0,1.2]',
                'nodata': 'NaN',
                'exclude_water_windows': bool(exclude_water),
                'water_windows_nm': tuple(tuple(map(float, w)) for w in (self.water_windows if water_windows is None else water_windows)),
            })
            # Coordenadas/índices
            try:
                arr_band = root.create_array('band', shape=(B_out,),
                                            chunks=(min(B_out, 1024),), dtype='int32',
                                            dimension_names=['band'])
                arr_band[:] = np.arange(B_out, dtype=np.int32)
                _set_dim_attrs(arr_band, ['band'])
            except:
                arr_band = root.create_array('band', shape=(B_out,),
                                            chunks=(min(B_out, 1024),), dtype='int32')
                arr_band[:] = np.arange(B_out, dtype=np.int32)
                _set_dim_attrs(arr_band, ['band'])

            if add_coords and (self.wavelengths is not None):
                wl_valid = self.wavelengths[valid_bands].astype('float32')

                try:
                    arr_wl = root.create_array('wavelength', shape=wl_valid.shape,
                                            chunks=(wl_valid.shape[0],), dtype='float32',
                                            dimension_names=['band'])
                    arr_wl[:] = wl_valid
                    _set_dim_attrs(arr_wl, ['band'])
                except:
                    arr_wl = root.create_array('wavelength', shape=wl_valid.shape,
                                            chunks=(wl_valid.shape[0],), dtype='float32')
                    arr_wl[:] = wl_valid
                    _set_dim_attrs(arr_wl, ['band'])

                if (self.fwhm is not None) and (len(self.fwhm) == len(self.wavelengths)):
                    fw_valid = self.fwhm[valid_bands].astype('float32')

                    try:
                        arr_fw = root.create_array('fwhm', shape=fw_valid.shape,
                                                chunks=(fw_valid.shape[0],), dtype='float32',
                                                dimension_names=['band'])
                        arr_fw[:] = fw_valid
                        _set_dim_attrs(arr_fw, ['band'])
                    except:
                        arr_fw = root.create_array('fwhm', shape=fw_valid.shape,
                                                chunks=(fw_valid.shape[0],), dtype='float32')
                        arr_fw[:] = fw_valid
                        _set_dim_attrs(arr_fw, ['band'])

            # Vector de control
            try:
                wb = root.create_array('written_bands', shape=(B_out,),
                                    chunks=(min(B_out, 1024),), dtype='uint8', dimension_names=['band'])
                wb[:] = 0
                _set_dim_attrs(wb, ['band'])
            except:
                wb = root.create_array('written_bands', shape=(B_out,),
                                    chunks=(min(B_out, 1024),), dtype='uint8')
                wb[:] = 0
                _set_dim_attrs(wb, ['band'])

        # 5) Pendientes (si resume)
        print("[INFO] Preparando lista de bandas pendientes...")
        out_index = {int(b): i for i, b in enumerate(valid_bands)}
        pending_bands = [b for b in valid_bands if (not resume) or (wb[out_index[b]] == 0)]

        # 6) Escritura acelerada
        print("[INFO] Iniciando escritura acelerada de reflectancia...")
        print(f"[INFO] Estrategia: {strategy} | band_block={band_block} | tile={tile if strategy=='tiled' else 'N/A'}")

        if tqdm is not None:
            if strategy == "bands":
                pbar = tqdm(total=(len([b for b in pending_bands]) + band_block - 1)//band_block,
                            desc="Batches(bands)")
                def _progress(it, total=None, desc=None):
                    return it  # barra manejada fuera
            else:
                pbar = None
                _progress = None
        else:
            pbar = None
            _progress = None

        if strategy == "bands":
            # NOTA: esta rama no es chunk-aware; úsala si no te daba problemas en Windows.
            self._write_reflectance_blockwise_bands(
                arr_reflect=arr_reflect,
                valid_bands=valid_bands,
                water_windows=water_windows,
                clip=(0.0, 1.2),
                band_block=band_block,
                progress=(lambda it, total=None, desc=None: it) if _progress is None else _progress
            )
            if pbar is not None:
                pbar.close()
        elif strategy == "tiled":
            # Estrategia CHUNK-AWARE + TILEADO: evita colisiones de .partial en Windows
            if tqdm is not None:
                Ty, Tx = tile
                tiles_y = (H + Ty - 1) // Ty
                tiles_x = (W + Tx - 1) // Tx
                c_chunk = arr_reflect.chunks[0]
                total_steps = tiles_y * tiles_x * ((len(valid_bands) + c_chunk - 1) // c_chunk)
                pbar = tqdm(total=total_steps, desc="Tiles×C-chunks")

                class _Prog:
                    def __init__(self, bar): self.bar = bar
                    def update(self, n=1): self.bar.update(n)

                self._write_reflectance_chunkwise_tiled(
                    arr_reflect=arr_reflect,
                    valid_bands=valid_bands,
                    water_windows=water_windows,
                    clip=(0.0, 1.2),
                    tile=tile,
                    progress=_Prog(pbar)
                )
                pbar.close()
            else:
                self._write_reflectance_chunkwise_tiled(
                    arr_reflect=arr_reflect,
                    valid_bands=valid_bands,
                    water_windows=water_windows,
                    clip=(0.0, 1.2),
                    tile=tile,
                    progress=None
                )
        else:
            raise ValueError("strategy debe ser 'bands' o 'tiled'.")

        # 7) Marcar bandas como escritas
        print("[INFO] Marcando bandas como escritas...")
        wb[:] = 1

        # 8) NDVI + veg_mask (si se pidió y existe wavelength exportada)
        print("[INFO] Calculando NDVI y veg_mask (si aplica)...")
        if add_ndvi and ('wavelength' in root):
            wl_export = root['wavelength'][:]
            def nb(target):
                return int(np.nanargmin(np.abs(wl_export - float(target))))
            try:
                i_red = nb(660.0)
                i_nir = nb(800.0)
                RED = arr_reflect[i_red, :, :].astype(np.float32)
                NIR = arr_reflect[i_nir, :, :].astype(np.float32)
                NDVI = (NIR - RED) / (NIR + RED + 1e-6)
                veg_mask = (NDVI > float(ndvi_threshold)) & np.isfinite(NDVI)

                yx_chunks = (chunks[1], chunks[2]) if len(chunks) == 3 else None

                if 'NDVI' in root: del root['NDVI']
                if 'veg_mask' in root: del root['veg_mask']

                try:
                    arr_ndvi = root.create_array('NDVI', shape=(H, W),
                                                chunks=yx_chunks, dtype='float32',
                                                dimension_names=['y', 'x'])
                    arr_ndvi[:] = NDVI.astype('float32')
                    _set_dim_attrs(arr_ndvi, ['y', 'x'])
                except:
                    arr_ndvi = root.create_array('NDVI', shape=(H, W),
                                                chunks=yx_chunks, dtype='float32')
                    arr_ndvi[:] = NDVI.astype('float32')
                    _set_dim_attrs(arr_ndvi, ['y', 'x'])
                
                try:
                    vm = root.create_array('veg_mask', shape=(H, W),
                                        chunks=yx_chunks, dtype='uint8',
                                        dimension_names=['y', 'x'])
                    vm[:] = veg_mask.astype('uint8')
                    vm.attrs.update({'0': 'no-veg', '1': 'veg'})
                    _set_dim_attrs(vm, ['y', 'x'])
                except:
                    vm = root.create_array('veg_mask', shape=(H, W),
                                        chunks=yx_chunks, dtype='uint8')
                    vm[:] = veg_mask.astype('uint8')
                    vm.attrs.update({'0': 'no-veg', '1': 'veg'})
                    _set_dim_attrs(vm, ['y', 'x'])
            except Exception as e:
                print(f"[WARN] NDVI/veg_mask no agregados: {e}")

        # 9) Cierre por swap atómico (si aplica)
        print("[INFO] Finalizando escritura Zarr...")
        if atomic_swap and (tmp_path is not None):
            # renombrado atómico en el mismo filesystem
            if os.path.isdir(zarr_path):
                shutil.rmtree(zarr_path)
            os.replace(tmp_path, zarr_path)

        # 10) Log final
        print(f"[OK] Zarr guardado en: {zarr_path}")
        print(f"    Bandas exportadas: {B_out} / {self.img.nbands}")
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