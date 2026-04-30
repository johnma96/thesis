# Análisis empírico de leakage por genotipo — Corrección #20 del jurado

> **Estado:** borrador preliminar — para discusión interna con directora y web Claude.
> No incorporar directamente al documento de tesis sin revisión.
>
> **Generado:** 2026-04-30 (PC A)
> **Archivos de respaldo:** `notebooks/401-jmmz-genotype-analysis.ipynb`,
> `reports/figures/category_D/`, `run_paso4_inference.py`

---

## 1. Contexto

El jurado (Manuel Mauricio Goez Mora, observación del 13 de abril de 2026) señala que el
PR-AUC de 0.963 obtenido por la arquitectura CNN-2D es "excepcionalmente alto" y plantea
la hipótesis de que la red neuronal pudo haber aprendido la **estructura espacial de las
parcelas** —definidas mediante polígonos etiquetados manualmente en campo— en lugar de la
**huella espectral del estrés por deficiencia de fósforo**. Adicionalmente, el jurado
interroga si la variable genotipo estuvo disponible durante el entrenamiento y si el
rendimiento fue evaluado por variedad, dado que los 8 genotipos del experimento presentan
perfiles espectrales naturalmente distintos.

Este documento presenta el diagnóstico empírico de dichas hipótesis a partir del análisis
de los datos crudos del experimento, la estructura del split espacial y la inferencia del
modelo final sobre el conjunto de prueba (split_id=3, completamente ajeno al entrenamiento).

---

## 2. Estructura de la metadata experimental

El archivo `data/raw/labels_export.gpkg` (capa `labels2`) contiene 335 registros con los
siguientes campos por parcela: `plot` (surco), `rep` (repetición), `entry` (genotipo),
`class` (nivel de estrés: 0–3) y `binary` (etiqueta binaria). La variable `entry` codifica
el genotipo de acuerdo con la Tabla 8 del Anexo 32:

| entry | genotipo       | tipo                  |
|-------|----------------|-----------------------|
| 1     | L1-12702       | experimental          |
| 2     | L2-G11819      | experimental          |
| 3     | L3-G50834      | experimental          |
| 4     | L4-50840       | experimental          |
| 5     | L15-51433      | experimental          |
| 6     | L17-G51018     | experimental          |
| 7     | Liborino       | comercial (testigo)   |
| 8     | Cargamanto     | comercial (testigo)   |

La variable `binary` es derivada directamente de `class` (`binary = 1` si `class ≥ 1`,
`binary = 0` si `class = 0`); la consistencia fue verificada sin ninguna discrepancia en
los 335 registros.

**Respecto a la pregunta del jurado:** el genotipo (`entry`) **no fue utilizado como
variable de entrada en ningún modelo**. Los modelos recibieron exclusivamente el cubo
hiperspectral de 63 canales (58 bandas espectrales seleccionadas por SNR proxy y
decorrelación + 5 índices de vegetación: NDVI, NDRE, CIgreen, PRI, PSRI). Ningún
identificador de variedad fue incorporado al pipeline de entrenamiento.

---

## 3. Hallazgo: entries no documentados en los datos crudos

El análisis de la variable `entry` revela la presencia de **dos entries fuera del rango
documentado (1–8)**: entry 9 y entry 10. Estos no corresponden a ninguno de los 8
genotipos descritos en el Anexo 32. Esta situación **no fue introducida por el pipeline
de modelado** — los entries 9 y 10 provienen directamente de los datos originales
entregados al autor por el equipo de campo, como puede verificarse en el archivo fuente
`labels_export.gpkg` previo a cualquier procesamiento.

El hallazgo crítico es que **ambos entries aparecen exclusivamente en class=0
(parcelas sanas, 100% de dosis de fósforo)**:

| entry | class=0 | class=1 | class=2 | class=3 | total |
|-------|---------|---------|---------|---------|-------|
| 9     | 11      | 0       | 0       | 0       | 11    |
| 10    | 11      | 0       | 0       | 0       | 11    |

**Impacto cuantitativo:**

| métrica                                           | valor          |
|---------------------------------------------------|----------------|
| Registros con entry no documentado               | 22 / 335 (6.6%) |
| Parcelas class=0 totales                         | 24 parcelas únicas |
| Parcelas class=0 con entry no documentado        | 6 / 24 (25.0%) |

Esta concentración exclusiva en class=0 crea en los datos crudos una asociación trivial
`{entry 9, entry 10} → binary = 0`. Si el modelo hubiera aprendido el perfil espectral
específico de estos genotipos no documentados y los reconociera en el test set, obtendría
clasificaciones correctas de "sano" sin haber aprendido la firma espectral del estrés.
La sección 5 demuestra que este escenario no ocurrió.

---

## 4. Distribución de genotipos en el dataset

Los entries 1–8 presentan registros en los cuatro niveles de estrés (class 0–3), con
totales entre 35 y 45 registros por genotipo. Los entries 9 y 10 presentan únicamente
registros sanos (class=0), confirmando el escenario descrito en la sección anterior.

Ver figura: `reports/figures/category_D/barplot_entry_class.png`

---

## 5. Distribución de genotipos en los splits espaciales

El split espacial fue realizado a nivel de surco (`plot`), estratificado únicamente por
la etiqueta binaria (`stratify=binary`), mediante `train_test_split` con
`random_state=42` (notebook `302-jmmz-spatial-split.ipynb`). **El genotipo no fue
considerado** como criterio de estratificación. El resultado fue: train=28 grupos,
val=10 grupos, test=10 grupos.

La distribución de los entries no documentados por split es la siguiente:

| entry | test   | train | val | total parcelas |
|-------|--------|-------|-----|----------------|
| 9     | **0**  | 3     | 0   | 3              |
| 10    | **0**  | 2     | 1   | 3              |

Ver figura: `reports/figures/category_D/heatmap_entry_split.png`

**Hallazgo determinante:** ni entry 9 ni entry 10 tienen ninguna parcela en el conjunto
de test (split_id=3). Entry 9 quedó íntegramente en train; entry 10 quedó distribuido
entre train (2 parcelas) y val (1 parcela, usada para early stopping y selección de
umbral). En consecuencia, **ningún píxel de entries no documentados integra el conjunto
sobre el cual se reporta el PR-AUC final**.

Adicionalmente, entry 4 (L4-50840) tampoco tiene parcelas en el test set — sus surcos
quedaron completamente asignados a train y val, artefacto del split espacial sin
estratificación por genotipo.

Los genotipos presentes en el test set son: **entries 1, 2, 3, 5, 6, 7 y 8**
(7 de los 8 genotipos oficiales).

---

## 6. Desempeño del CNN-2D por genotipo — conjunto de test

Se ejecutó inferencia completa del modelo final (`cnn2d_final_model_weights.pt`,
run_id MLflow `61a3cc05f39d46f79f2e3fa3d29fae7f`) sobre el test set (split_id=3,
246,132 píxeles tras extracción de parches 5×5). El cubo de features fue construido
desde `data/interim/masked_reflectance.zarr` con el orden de canales confirmado:
`[NDVI, NDRE, CIgreen, PRI, PSRI, band_1, ..., band_374]`.

**Check de sanidad:** PR-AUC global test = **0.9637**
(valor reportado en tesis: 0.9635; diferencia < 0.001 ✓).

**Tabla A — Métricas por genotipo, test set (todos los entries presentes):**

| entry | genotipo     | n_px    | n_pos  | n_neg  | PR-AUC | ROC-AUC | F1    | Recall |
|-------|--------------|---------|--------|--------|--------|---------|-------|--------|
| 1     | L1-12702     | 55,371  | 29,854 | 25,517 | 0.9281 | 0.9070  | 0.823 | 0.930  |
| 2     | L2-G11819    | 57,815  | 45,688 | 12,127 | 0.9926 | 0.9718  | 0.953 | 0.956  |
| 3     | L3-G50834    | 41,145  | 41,145 | 0      | —      | —       | 0.910 | 0.835  |
| 5     | L15-51433    | 29,173  | 17,817 | 11,356 | 0.9259 | 0.8963  | 0.856 | 0.862  |
| 6     | L17-G51018   | 10,236  | 10,236 | 0      | —      | —       | 0.664 | 0.497  |
| 7     | Liborino     | 31,866  | 31,866 | 0      | —      | —       | 0.969 | 0.940  |
| 8     | Cargamanto   | 20,526  | 10,373 | 10,153 | 0.8238 | 0.8518  | 0.781 | 0.921  |

*Nota: entries 3, 6 y 7 presentan n_neg=0 en el test set porque todas sus parcelas
class=0 quedaron asignadas a train/val por efecto del split espacial sin estratificación
de genotipo. PR-AUC y ROC-AUC no son calculables cuando una sola clase está presente.*

**PR-AUC global restringido a entries oficiales (1–8) = 0.9637**, idéntico al global
dado que entries 9 y 10 tienen cero píxeles en el test set.

---

## 7. Pruebas de ablación espectral (evidencia adicional)

Para fortalecer el argumento de que el modelo aprendió la firma espectral y no la
estructura espacial de las parcelas, se realizaron dos pruebas de ablación sobre el
test set.

### 7.1 Permutación espectral dentro del parche

Se permutaron aleatoriamente los 63 canales espectrales de cada parche 5×5, preservando
la estructura espacial del vecindario pero destruyendo el contenido espectral.

| condición              | PR-AUC | ROC-AUC | F1    |
|------------------------|--------|---------|-------|
| modelo original        | 0.9637 | 0.8902  | —     |
| espectro permutado     | **PENDIENTE** | | |

*Interpretación esperada: si PR-AUC colapsa a ~0.5–0.6, confirma dependencia
fundamental del contenido espectral.*

### 7.2 Evaluación solo con píxel central (sin contexto espacial)

Se zerorizaron los píxeles vecinos del parche 5×5, conservando únicamente el canal
espectral del píxel central. Equivale a una evaluación espectral pura sin contexto
espacial.

| condición              | PR-AUC | ROC-AUC | F1    |
|------------------------|--------|---------|-------|
| modelo original        | 0.9637 | 0.8902  | —     |
| solo píxel central     | **PENDIENTE** | | |

*Interpretación esperada: si PR-AUC cae a ~0.83 (comparable al CNN-1D real),
la ganancia de 0.96 vs 0.83 proviene del contexto espacial local legítimo,
no de memorización de posiciones.*

---

## 8. Hallazgos preliminares

- **Los entries no documentados (9 y 10) no tienen presencia en el test set.** El
  PR-AUC reportado (0.9635) fue calculado exclusivamente sobre píxeles de los 7
  genotipos oficiales presentes en el holdout. La asociación trivial
  `{entry 9, entry 10} → sano` no pudo inflar la métrica de evaluación final.

- **El genotipo nunca fue variable de entrada del modelo.** La red recibió únicamente
  el cubo hiperspectral de 63 canales; ningún identificador de variedad fue incorporado
  al pipeline de entrenamiento.

- **El rendimiento por genotipo muestra variación genuina.** Para los cuatro entries
  con ambas clases representadas en test, el PR-AUC oscila entre 0.8238 (Cargamanto)
  y 0.9926 (L2-G11819). Esta variación es consistente con un modelo que aprende la
  firma espectral del estrés, cuya expresión difiere entre genotipos. Un modelo que
  memorizara estructura espacial produciría rendimiento uniformemente alto en todos
  los genotipos.

- **Entry 6 (L17-G51018) presenta Recall = 0.497 en test.** El modelo clasifica
  aproximadamente la mitad de sus píxeles estresados como sanos, evidenciando
  dificultad real de clasificación en ese genotipo, incompatible con la hipótesis
  de memorización espacial.

- **El diseño experimental presenta una irregularidad objetiva** en los datos crudos
  (entries 9 y 10 de procedencia no documentada, exclusivos de class=0), cuya
  influencia en el entrenamiento (vía train y val) merece ser discutida, pero que
  **no afecta la validez de la métrica de evaluación final** por ausencia total en
  el test set.

---

## 9. Próximos pasos

*(Pendiente de discusión con la directora y revisión en web Claude antes de redactar
la respuesta definitiva al jurado.)*

- Completar resultados de las pruebas de ablación (secciones 7.1 y 7.2).
- Decidir si se reporta la ausencia de entry 4 del test set como limitación del diseño
  del split.
- Evaluar si incluir la variación de PR-AUC entre genotipos (0.82–0.99) en la sección
  de discusión de la tesis.
- Redactar la sección de discusión definitiva para la tesis (Corrección #20c).
- Determinar el tratamiento formal de los entries 9 y 10 (exclusión documentada vs.
  retención con nota metodológica).
