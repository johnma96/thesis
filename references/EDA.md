# Resumen del EDA y Preprocesamiento

## ✅ 1. Transformación del cubo hiperespectral
- **Formato original**: ENVI (.bsq + .hdr), dimensiones (3660, 3438, 379).
- **Preprocesamiento aplicado**:
  - Correcciones radiométricas y atmosféricas.
  - Conversión a reflectancia física (escala ÷10000).
  - Máscara de valores inválidos (-1, 15000) → NaN.
  - Clip a rango [0.0, 1.2].
- **Transformación a Zarr**:
  - Archivo: `masked_reflectance.zarr`.
  - Estructura: (band, y, x), chunks (64, 512, 512).
  - Coordenadas: `wavelength`, `fwhm`, `band`.
  - Capas auxiliares: `NDVI`, `veg_mask`, `written_bands`.
  - Exclusión de bandas en ventanas de absorción de agua: 1340–1440 nm y 1800–1950 nm.

## ✅ 2. Limpieza y máscaras
- **Mapa NoData**:
  - Definición: píxel con NaN en ≥1 banda.
  - Resultado: 6,484,992 píxeles (≈51.54%).
- **Máscara de vegetación**:
  - Basada en NDVI (>0.30).
  - Validada contra capa `veg_mask` del Zarr.

## ✅ 3. Estadísticos espectrales (vegetación)
- **Mean y std por banda**:
  - VNIR: media baja en azul, valle profundo en rojo (~680 nm).
  - Red-edge (~705–740 nm): incremento brusco.
  - NIR: meseta estable.
  - SWIR: oscilaciones.
- **Rango min–max**: confirma heterogeneidad espacial.

## ✅ 4. SNR proxy (mean/std en vegetación)
- Valores: 1.5–3.2.
- Patrones:
  - Valles en azul y rojo.
  - Mejora en red-edge y NIR.
  - SWIR con picos altos.
- **Decisión**: usar SNR proxy como criterio (≥Q25).

## ✅ 5. Correlación entre bandas contiguas
- **Versión inicial**: caídas por gaps.
- **Versión limpia por segmentos**:
  - Segmentos:
    - seg0: 411–914 nm (174 bandas, paso ~2.9 nm)
    - seg1: 991–1114 nm (25 bandas, paso ~5.1 nm)
    - seg2: 1160–1329 nm (34 bandas, paso ~5.1 nm)
    - seg3: 1482–1784 nm (60 bandas, paso ~5.1 nm)
    - seg4: 2014–2449 nm (86 bandas, paso ~5.1 nm)
  - Correlación intra-segmento:
    - seg0–seg3: ≈0.99–1.00 → altísima redundancia.
    - seg4: 0.80–0.98 → más estructura útil.
- **Decorrelación (L95/L90)**:
  - seg0–seg3: ≥34–61 nm → sobre-muestreo.
  - seg4: ~6 nm → cambios rápidos.

## ✅ 6. Selección de bandas
- **Filtros aplicados**:
  - Cobertura en vegetación ≥48%.
  - SNR proxy ≥Q25.
  - Márgenes contra bordes de gaps.
  - Espaciado mínimo por segmento (basado en L95 o regla adaptativa).
  - Evitar correlación a pares >0.9995 (seg0-seg3) y > 0.98 (seg4).
- **Prioridad**: orden por SNR (bonus opcional en red-edge).
- **Salida**: `bands_selected_by_segment.csv`.

## ✅ 7. Productos generados
- **Máscaras**: `mask_nodata.zarr`.
- **Índices espectrales**: NDVI, NDRE, PRI, PSRI, CIgreen (GeoTIFF + Zarr).
- **Quicklook RGB**: PNG.
- **PCA**: componentes (GeoTIFF + Zarr) + varianza explicada.
- **Reportes CSV**:
  - `snr_proxy_per_band.csv`
  - `adjacent_corr_by_segment.csv`
  - `spectral_segments.csv`
  - `decorrelation_length_summary.csv`
  - `bands_selected_by_segment.csv`

## ✅ Próximo paso
- Validar bandas seleccionadas con baseline ML.
- Generar figura resumen para la tesis:
  - Curva SNR proxy + bandas seleccionadas.
  - Correlación por segmentos + gaps sombreados.
  - Tabla con segmentos, L95/L90 y n.º de bandas elegidas.
