
# Pipeline inicial para procesar imagen hiperespectral (ENVI/BSQ)

> **Meta**: Dejar el cubo listo para etiquetado/modelado: reflectancia física [0–1], máscara de vegetación, bandas ruidosas descartadas, índices espectrales precomputados, visualizaciones diagnósticas y exportes eficientes (COG/Zarr).

## 0) Consideraciones del dataset
- Formato ENVI **BSQ** (`samples=3438`, `lines=3660`, `bands=379`).
- Correcciones PARGE/ATCOR aplicadas.
- **Reflectancia escalada**: `reflectance scale factor = 10000` (valores reales = entero/10000).
- **Máscaras**: `background=-1`, `data ignore value=15000` → ignorar.
- **Geo**: UTM 18N (WGS84), tamaño de píxel 0.02 m.
- Longitudes de onda y **FWHM** (si existen en el .hdr) para filtrar y seleccionar bandas.

## 1) Estructura del proyecto
```
project/
├─ data/
│  ├─ raw/        # .bsq/.hdr originales
│  ├─ interim/    # cubos normalizados, máscaras, quicklooks
│  └─ derived/    # índices, PCA, superpixeles, tiles
└─ notebooks/
```

## 2) Lectura, normalización y máscaras
- Leer con acceso por ventanas (sin cargar todo en RAM).
- Convertir a **reflectancia física** dividiendo por 10000.
- Aplicar máscara de NoData (background, ignore value) y clip [0, 1.2].

## 3) Máscara de vegetación (NDVI/NDRE)
- Localizar bandas por **longitud de onda** (p. ej. 660 nm y 800 nm).
- Calcular NDVI y NDRE, seleccionar **veg_mask = NDVI > 0.3** (ajustable por histograma).

## 4) Limpieza espectral
- Eliminar bandas en ventanas de **absorción de vapor de agua** (p. ej., 1340–1440, 1800–1950 nm) si existen.
- (Opcional) Suavizado espectral **Savitzky–Golay**.

## 5) Quicklooks
- RGB con bandas por defecto del header.
- PCA (3 componentes) para falso color.

## 6) Índices espectrales
- NDVI, NDRE, PRI, PSRI, CIgreen (entre otros) usando bandas cercanas a las longitudes de onda estándar.
- Guardar como GeoTIFF con la misma georreferenciación.

## 7) Exportes eficientes
- **COG multibanda**: GeoTIFF optimizado (tiled, DEFLATE, overviews internas) de las **bandas válidas**.
- **Zarr + xarray/dask** (opcional) para flujos de ML con chunking.

## 8) (Opcional) Superpixeles
- SLIC sobre quicklook RGB para agregar por objeto (planta/hoja) y reducir ruido.

## 9) Control de calidad
- Porcentaje de NoData por banda.
- Estadísticas por banda (mean/std en vegetación).
- Correlación entre bandas contiguas para detectar inestabilidad.

## 10) Entregables
1. Cubo de reflectancia (subset de bandas válidas, `float32`).
2. Máscara de vegetación (`veg_mask.tif`).
3. Índices (NDVI, NDRE, PRI, PSRI, CIgreen).
4. Quicklooks (RGB, PCA) en PNG.
5. CSV con bandas válidas (wavelength, FWHM).
6. (Opcional) Superpixeles SLIC.
7. Reporte HTML con gráficos y logs.

## 11) Checklist
- [ ] Confirmar vector `wavelength`/`fwhm` en el `.hdr`.
- [ ] Revisar consistencia de `data type` vs archivo.
- [ ] Validar crs/transform en QGIS.
- [ ] Documentar bandas descartadas (absorción/ruido).
- [ ] Guardar parámetros (umbrales, ventanas SG) en YAML/JSON para reproducibilidad.
