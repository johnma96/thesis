# Tabla costo computacional — Correccion C #18

Hardware inferencia: CPU (modelos ML) | GPU NVIDIA GeForce RTX 3050 Laptop GPU (CNN-1D, CNN-2D)
CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | RAM: 15.7 GB
(M) = dato de MLflow | (E) = medicion empirica local

| Metrica | LogisticRegression | SGDClassifier | LightGBM | XGBoost | CNN-1D | CNN-2D |
|---|---|---|---|---|---|---|
| Tamano en disco | 2.8 KB | 2.9 KB | 30.2 KB | 35.1 MB | 91.0 KB | 1.43 MB |
| Estructura interna | 64 coeficientes | 64 coeficientes | 1 arbol(es), num_leaves=123 | 3552 arboles, max_depth=8 | 21,474 parametros | 372,994 parametros |
| Entrenamiento run final (M) | 3 min 28 s | 2 min 53 s | 1 min 54 s | 35 min 30 s | 58 min 15 s | 17 min 31 s |
| HPO total acumulado (M) | 2h 3min | 15 min 52 s | 9 min 02 s | 10h 22min | 15h 12min | 10h 22min |
| Epocas entrenadas (M) | N/A | N/A | N/A | N/A | 109 / 200 | 22 / 80 |
| Hardware inferencia (E) | CPU | CPU | CPU | CPU | GPU (RTX 3050) | GPU (RTX 3050) |
| Mem. pico inferencia MB (E) | 0.4 | 0.0 | 0.2 | 0.4 | 21.2 | 29.3 |
| Latencia batch-512 ms (E) | 0.91 | 0.90 | 1.88 | 14.27 | 0.79 | 4.67 |
| Throughput muestras/s (E) | 563,334 | 570,334 | 272,265 | 35,871 | 651,383 | 109,587 |
| Tiempo 1M pixeles s (E) | 1.8 | 1.8 | 3.7 | 27.9 | 1.5 | 9.1 |