# Carta de respuesta a las observaciones del jurado evaluador

**Trabajo Final de Maestría:** *Diagnóstico no invasivo del estado de salud del fríjol común (Phaseolus vulgaris L.) en Colombia: Un enfoque basado en la huella espectral y la inteligencia artificial.*

**Estudiante:** John Mario Montoya Zapata
**Director(a):** Maria Constanza Torres Madroñero
**Jurado evaluador:** Manuel Mauricio Goez Mora — Instituto Tecnológico Metropolitano (ITM)
**Fecha de las observaciones:** 13 de abril de 2026
**Fecha de respuesta:** 12 de mayo de 2026

---

## Presentación

Se presenta a continuación la respuesta a las veinte observaciones formuladas por el jurado evaluador sobre el Trabajo Final de Maestría. Cada observación se transcribe literalmente y se acompaña de la descripción de las modificaciones realizadas en el documento, así como de la ubicación específica de los cambios. En todos los casos, las modificaciones se han incorporado al documento final preservando los resultados experimentales originales del trabajo, sin nuevos entrenamientos ni modificaciones de los modelos previamente registrados.

Las observaciones se han agrupado por afinidad temática y se han atendido siguiendo un orden de prioridad determinado por su impacto sobre la coherencia general del documento. Para facilitar la verificación, las referencias a secciones, figuras y tablas corresponden a la numeración del documento corregido.

Agradezco al jurado evaluador la rigurosidad y profundidad de las observaciones planteadas, las cuales han contribuido de manera sustancial al fortalecimiento metodológico y argumentativo del trabajo.

---

## Tabla de seguimiento de observaciones

| N° | Observación | Categoría | Estado | Sección modificada |
|---|---|---|---|---|
| 1 | Normalización de separadores decimales | Forma | **Atendida** | Todo el documento |
| 2 | Corrección ortográfica y de digitación | Forma | **Atendida** | Todo el documento |
| 3 | Redundancia en la definición de imágenes hiperespectrales | Forma | **Atendida** | §1.2 y §2.2 |
| 4 | Justificación del umbral NDVI | Metodología | **Atendida** | §4.1 |
| 5 | Criterio de "alto bienestar" y representatividad | Metodología | **Atendida** | §4.1 |
| 6 | Ampliación de selección de bandas y métodos descartados | Metodología | **Atendida** | §3.3 |
| 7 | Ventaja técnica de la división validación–prueba | Metodología | **Atendida** | §3.4.2.2 |
| 8 | Aclaración sobre la única toma aérea | Metodología | **Atendida** | §3.1.2 |
| 9 | Justificación de los 12 algoritmos iniciales | Metodología | **Atendida** | §3.4.3.1 |
| 10 | Profundización del análisis de Random Forest | Modelado | **Atendida** | §4.3.1 |
| 11 | Hiperparametrización sin cambios significativos | Modelado | **Atendida** | §4.4 |
| 12 | Importancia relativa de FP vs FN | Discusión | **Atendida** | §4.6 |
| 13 | Caso práctico: parcelas con plantas mixtas | Discusión | **Atendida** | §4.6 |
| 14 | Reestructuración de conclusiones | Estructura | **Atendida** | §5.1 |
| 15 | Trabajo futuro: validación ante otros estreses y genotipos | Estructura | **Atendida** | §5.2 |
| 16 | Matrices de confusión en porcentajes y reducción de datos | Análisis | **Atendida** | §4.3.4.3 |
| 17 | Influencia real de los índices de vegetación | Análisis | **Atendida** | §4.3.5 |
| 18 | Métricas de costo computacional para modelos DL | Análisis | **Atendida** | §4.4.1 |
| 19 | Diagrama del diseño experimental | Análisis | **Atendida** | §3.1.1 |
| 20 | Riesgo de aprendizaje de estructura espacial en CNN-2D | Modelado | **Atendida** | §4.5 |

**Total de observaciones atendidas: 20 de 20.**

---

# Bloque I — Observaciones metodológicas críticas

---

## Observación 20 — Posible aprendizaje de la estructura espacial por parte del modelo CNN-2D

### Texto original del jurado

> *"Respecto al modelado, el desempeño del modelo CNN-2D con presento un PR-AUC de 0.963 resulta excepcionalmente alto, lo que sugiere la necesidad de revisar si la red está aprendiendo la estructura espacial de las parcelas en lugar de la firma espectral del estrés, dado el etiquetado por polígonos manuales. Es fundamental mencionar si el modelo incluyó el 'Genotipo' como variable o si se evaluó el desempeño por variedad, considerando que las firmas espectrales varían naturalmente entre los 8 genotipos utilizados; en este sentido, debe aclararse si la omisión de este criterio en la conformación de los conjuntos de validación y prueba genera alguna afectación."*

### Descripción de las modificaciones realizadas

Se incorporó al Capítulo 4 una nueva sección titulada **"4.5 Análisis de robustez del modelo CNN-2D"**, dedicada exclusivamente a evaluar empíricamente la inquietud planteada por el jurado. La sección se estructura en tres análisis complementarios sobre el modelo final, ejecutados sin reentrenamiento alguno y utilizando exclusivamente el conjunto de prueba (espacialmente disjunto del conjunto de entrenamiento por construcción).

**Análisis 1 — Desempeño desagregado por genotipo.** Se aclaró explícitamente que el genotipo no fue utilizado como variable de entrada en ninguno de los modelos desarrollados; la red CNN-2D recibió únicamente el cubo hiperspectral compuesto por las 58 bandas seleccionadas y los 5 índices de vegetación (NDVI, NDRE, CIgreen, PRI, PSRI). Sobre el conjunto de prueba se desagregaron las métricas por entrada (entry) y se observó una variación de aproximadamente 0,17 puntos de PR-AUC entre genotipos (rango: 0,824 para Cargamanto hasta 0,993 para L2-G11819). Esta heterogeneidad es **incompatible con la hipótesis de memorización espacial** —que produciría desempeño uniforme entre genotipos por independencia entre la información geométrica y la varietal— y **consistente con un modelo que aprende la huella espectral del estrés**, cuya manifestación bioquímica varía entre variedades por diferencias en pigmentos, contenido de agua y arquitectura foliar. El caso del entry 6 (L17-G51018), con un recall de 0,497, refuerza esta interpretación: un modelo que hubiera memorizado la geometría no presentaría diferencias de desempeño entre variedades.

**Análisis 2 — Verificación de identificadores no documentados.** Se identificaron en los datos crudos las entradas 9 y 10, no incluidas en la Tabla 8 del Anexo 32, presentes únicamente en la clase 0 (no estresada). Su existencia se reporta de manera transparente en el documento. Estas entradas no aparecen en el conjunto de prueba y, por tanto, no afectan las métricas reportadas.

**Análisis 3 — Pruebas de ablación espectral.** Se diseñaron y ejecutaron dos pruebas para aislar la contribución del contenido espectral frente a la del contexto espacial:

- **Prueba 1 (permutación espectral):** se permutó aleatoriamente el orden de las bandas en cada parche del conjunto de prueba, preservando la estructura espacial pero destruyendo la coherencia espectral. El ROC-AUC colapsó a 0,506, indistinguible del azar.
- **Prueba 2 (píxel central):** se evaluó el modelo replicando el espectro del píxel central en todas las posiciones del parche 5×5, conservando información espectral pero eliminando el contexto espacial. El PR-AUC obtenido (0,815) resulta prácticamente equivalente al de la arquitectura CNN-1D real (0,83), que opera por construcción sobre un vector espectral sin vecindad.

La diferencia entre el PR-AUC global del CNN-2D (0,964) y el obtenido en la Prueba 2 (0,815) cuantifica en aproximadamente 0,15 puntos la ganancia atribuible al contexto espacial local, ganancia que corresponde a información espectral aportada por los píxeles vecinos dentro del mismo surco —no a memorización de la geometría de los polígonos. Esta interpretación es coherente con la escala del campo receptivo del modelo: un parche de 5×5 píxeles representa solo 10×10 cm sobre el terreno, frente a un surco típico de 3,6 × 1 m equivalente a aproximadamente 9.000 píxeles, por lo que la red dispone de información local suficiente para integrar textura y vecindad espectral pero carece de la cobertura espacial necesaria para reconstruir la forma de los polígonos manualmente delimitados.

**Sobre el efecto de la no estratificación por genotipo en los splits.** Se reconoce explícitamente como una limitación del diseño experimental que la partición espacial fue estratificada únicamente por la etiqueta binaria y no por genotipo. Esta limitación se documenta en el texto y se incorpora al apartado de trabajo futuro como una línea de mejora metodológica para estudios subsiguientes.

**Citas bibliográficas incorporadas:** Roberts et al. (2017) sobre estrategias de validación cruzada para datos con estructura espacial; Kattenborn et al. (2022) sobre la inflación del desempeño en CNN cuando hay autocorrelación espacial entre muestras de entrenamiento y validación; Okyere et al. (2023) sobre la complementariedad espectro-espacial en modelos profundos para identificación del estado nutricional con HSI.

### Ubicación en el documento

- §4.5 "Análisis de robustez del modelo CNN-2D" (sección nueva).
- Subsecciones: 4.5.1 Desempeño desagregado por genotipo; 4.5.2 Verificación de identificadores no documentados; 4.5.3 Pruebas de ablación espectral; 4.5.4 Síntesis del análisis.

---

# Bloque II — Observaciones metodológicas

---

## Observación 4 — Justificación del umbral NDVI

### Texto original del jurado

> *"En términos metodológicos, se requiere clarificar la selección empírica del umbral para la máscara de vegetación, explicando su impacto en el proceso de clasificación. Debe discutirse si se exploraron variaciones en dicha máscara y si los valores obtenidos son consistentes con los rangos tradicionales del índice NDVI, apoyándose en citas bibliográficas."*

### Descripción de las modificaciones realizadas

Se reescribió completamente la sección §4.1 "Máscara de vegetación" para abordar los tres requerimientos del jurado: clarificación del proceso de selección del umbral, justificación bibliográfica de su valor, y análisis cuantitativo de su impacto y de las variaciones exploradas.

**Especificación técnica del umbral.** Se documenta que el NDVI fue calculado a partir de las bandas espectrales más cercanas a las longitudes de onda canónicas del rojo (660 nm) y del infrarrojo cercano (800 nm), respetando la tolerancia espectral del sensor. Se establece el umbral de NDVI ≥ 0,3 como criterio de inclusión de píxeles válidos para análisis.

**Justificación bibliográfica del umbral.** Se contextualiza el valor adoptado dentro de los rangos canónicos reportados en la literatura: valores cercanos a 0 y negativos para suelo desnudo, agua y superficies inertes; rango 0,2 a 0,5 para vegetación dispersa, en estados tempranos o bajo estrés evidente; valores superiores a 0,5 para vegetación densa y sana (Meneses-Tovar, 2011). Para enmascaramientos específicos en agricultura de precisión, los valores de corte suelen oscilar entre 0,2 y 0,4 dependiendo del cultivo, la fase fenológica y la resolución espacial del sensor (Adão et al., 2017; Sishodia et al., 2020). El umbral adoptado (0,3) se ubica dentro de este rango canónico.

**Impacto cuantitativo y exploración de variaciones.** Se reportan los efectos numéricos del proceso de enmascaramiento sobre el conjunto de datos: del total de 12.583.080 píxeles del cubo original, 6.055.113 píxeles (48,12 %) resultaron válidos tras la combinación de las máscaras de no-data y de vegetación; los píxeles etiquetados y utilizados en el modelado ascendieron a 1.748.639 (13,9 % del total). Se exploró cualitativamente el efecto de variaciones del umbral: umbrales más bajos (≈0,2) incorporaban píxeles de transición con señales mixtas; umbrales más altos (≈0,4) eliminaban píxeles periféricos del dosel biológicamente activos.

**Citas bibliográficas incorporadas:** Meneses-Tovar (2011) — cita nueva añadida a la bibliografía; Adão et al. (2017) y Sishodia et al. (2020) — referencias ya presentes en el documento.

### Ubicación en el documento

- §4.1 "Máscara de vegetación" — sección completamente reescrita y ampliada.

---

## Observación 5 — Criterio de "alto bienestar" y representatividad

### Texto original del jurado

> *"es fundamental analizar si el criterio de conservar solo vegetación con 'alto bienestar' excluye a las plantas con mayores deficiencias nutricionales, afectando la representatividad del experimento."*

### Descripción de las modificaciones realizadas

Se incorporó como parte de la sección §4.1 (atendida también en la Observación 4) un análisis específico de la implicación del criterio de inclusión sobre la representatividad del experimento, organizado en tres argumentos:

1. **Rango acotado de los tratamientos.** El experimento agronómico contempló niveles de fertilización con fósforo en el rango 25 %–100 % de la dosis óptima, produciendo deficiencias relativas pero sin alcanzar escenarios de privación absoluta del nutriente. Las plantas estresadas mantuvieron, por tanto, suficiente actividad fotosintética para producir respuestas espectrales por encima del umbral establecido.

2. **Etiquetado independiente del NDVI.** El etiquetado manual fue realizado a nivel de polígono asociado a cada surco, con base en la identificación visual de las parcelas en el mosaico hiperspectral, no a partir del NDVI. La permanencia de las cuatro condiciones experimentales (T1 a T4) en el conjunto de datos final, con proporciones correspondientes a su distribución espacial original, indica que la máscara no excluyó sistemáticamente parcelas asociadas a tratamientos de mayor estrés.

3. **Granularidad de exclusión a nivel de píxel.** Los píxeles efectivamente excluidos por el filtro de NDVI corresponden mayoritariamente a zonas de suelo expuesto entre plantas, sombras y bordes de surco, no a plantas completas. La unidad de exclusión es el píxel y no el individuo vegetal, por lo que cada planta del experimento contribuyó al conjunto de datos con un número variable de píxeles válidos sin verse completamente eliminada.

Se reconoce explícitamente como limitación del enfoque que el umbral fue establecido empíricamente y no mediante una optimización sistemática frente a una métrica externa de validación, y se identifica la caracterización del impacto de variaciones controladas del umbral y la exploración de máscaras adaptativas como líneas pertinentes de trabajo futuro.

### Ubicación en el documento

- §4.1 "Máscara de vegetación" — análisis integrado en los párrafos finales de la sección reescrita.

---

## Observación 6 — Ampliación de la selección de bandas y métodos descartados

### Texto original del jurado

> *"En cuanto a la selección de bandas, es relevante ampliar la descripción de la técnica de reducción de dimensionalidad y sintetizar las razones por las cuales no se contemplaron otros métodos."*

### Descripción de las modificaciones realizadas

Se ampliaron los párrafos introductorios de §3.3 con tres bloques nuevos que documentan de manera transparente la exploración previa de técnicas clásicas de reducción de dimensionalidad y la justificación del cambio de estrategia hacia la selección de bandas adoptada en el trabajo:

**Métodos clásicos considerados.** Se describe que, antes de adoptar la estrategia finalmente implementada, se evaluó la viabilidad de aplicar técnicas clásicas (PCA, ICA, LDA y variantes basadas en kernel), referenciadas en la Sección 2.3.2. Se contextualiza el problema computacional implicado: una matriz aproximada de 12 millones de píxeles válidos por 379 bandas espectrales.

**Limitaciones operativas observadas.** Se documenta que la descomposición en valores singulares requerida por PCA estándar no resultó tratable sobre la matriz completa en el equipo de cómputo utilizado, debido a la combinación de alta dimensionalidad espectral, gran volumen de píxeles y fuerte correlación entre bandas adyacentes (correlaciones superiores a 0,99 en los segmentos VNIR, NIR y SWIR1). Se reporta el intento posterior con `IncrementalPCA` y muestreo aleatorio estratificado de píxeles (documentado en el Notebook 201 del repositorio), señalando que esta vía implicó la introducción de decisiones adicionales cuya sensibilidad sobre las componentes resultantes comprometía la reproducibilidad del procedimiento. Adicionalmente, se documenta una limitación interpretativa común a todas las técnicas mencionadas: las características generadas son combinaciones lineales o no lineales de las bandas originales, lo cual dificulta su asociación directa con procesos fisiológicos conocidos y la posterior construcción de índices de vegetación canónicos.

**Justificación de la estrategia adoptada.** Se explicita que, frente a estas limitaciones, se optó por una selección informada de bandas espectrales que conservara las bandas originales —preservando su interpretabilidad fisiológica— y aprovechara la estructura de correlación espectral característica de los datos hiperespectrales para reducir la redundancia. La estrategia se fundamenta en métricas estadísticas (relación señal-ruido y longitud de decorrelación espectral) y criterios fisiológicos, implementada de forma reproducible mediante la función `select_bands` (Notebook 202).

**Citas bibliográficas incorporadas:** S. Li et al. (2019), Licciardi et al. (2012), Prasad & Bruce (2008) — referencias ya presentes en §2.3.2 que se conectan ahora con la discusión metodológica.

### Ubicación en el documento

- §3.3 "Selección de bandas espectrales relevantes" — tres párrafos nuevos al inicio de la sección.

---

## Observación 7 — Ventaja técnica de la división validación–prueba

### Texto original del jurado

> *"También debe acentuarse la ventaja técnica de dividir los datos de validación y prueba, considerando que provienen de la misma muestra y conjunto de datos."*

### Descripción de las modificaciones realizadas

Se insertaron dos párrafos nuevos en §3.4.2.2 "Partición espacial del conjunto de datos", explicitando las funciones diferenciadas de cada subconjunto y la ventaja técnica de la separación, aun cuando los tres provienen de la misma muestra.

**Funciones diferenciadas.** Se documenta que el subconjunto de entrenamiento se utiliza exclusivamente para ajustar los parámetros internos del modelo. El subconjunto de validación cumple tres funciones complementarias: (i) selección de hiperparámetros durante la búsqueda con Optuna; (ii) monitoreo de convergencia y aplicación del criterio de detención temprana en los modelos de Deep Learning; (iii) calibración del umbral de decisión sobre las probabilidades predichas. El subconjunto de prueba queda reservado para una única evaluación final del modelo ya seleccionado.

**Ventaja técnica.** Se argumenta que la tripartición permite separar el proceso de selección y ajuste del modelo del proceso de evaluación de su capacidad de generalización. Si se prescindiera del subconjunto de validación y se utilizara directamente el de prueba para guiar la optimización de hiperparámetros o la selección del umbral, los valores finales reportados incorporarían un sesgo optimista derivado del ajuste indirecto a sus particularidades estadísticas. Este sesgo, conocido en la literatura como sobreajuste al conjunto de prueba, es particularmente relevante en escenarios con desbalance de clases y datos espacialmente correlacionados como los del presente trabajo.

### Ubicación en el documento

- §3.4.2.2 "Partición espacial del conjunto de datos" — dos párrafos nuevos.

---

## Observación 8 — Aclaración sobre la única toma aérea utilizada

### Texto original del jurado

> *"Aunque se mencionan dos momentos de captura, el texto debe reflejar con precisión que este estudio se centró en la información de una única toma aérea."*

### Descripción de las modificaciones realizadas

Se reescribió la sección §3.1.2 "Imágenes hiperespectrales e índices de vegetación" para reflejar con precisión que el trabajo se basó en una única captura hiperspectral. La fecha de adquisición se confirmó mediante la metadata embebida en el archivo HYSPEX (`acquisition time: 2021-11-23T23:16:57.0Z`).

**Modificaciones específicas realizadas:**

- Se eliminó la mención previa a dos fechas clave del ciclo del cultivo.
- Se incorporó la formulación: *"La adquisición se realizó en una única campaña de vuelo el 23 de noviembre de 2021, correspondiente a una fase fenológica del cultivo en la cual los efectos del estrés nutricional por deficiencia de fósforo son detectables a nivel de respuesta espectral. Aunque el experimento agronómico de campo contempló mediciones in situ en otras fechas (Anexo 32), el presente trabajo se basa exclusivamente en la información obtenida de esta única captura hiperspectral."*
- Se corrigió la dimensión espacial reportada del cubo (3.660 × 3.438 píxeles), consistente con los 12.583.080 píxeles totales por banda.
- Se añadieron datos disponibles en la metadata: sensor HYSPEX, resolución espacial real (2 cm/píxel), procesamiento radiométrico previo (PARGE/ATCOR) y sistema de coordenadas (EPSG:32618).

### Ubicación en el documento

- §3.1.2 "Imágenes hiperespectrales e índices de vegetación" — sección reescrita.

---

## Observación 9 — Justificación de los 12 algoritmos iniciales

### Texto original del jurado

> *"Es necesario justificar la selección de los 12 algoritmos iniciales, considerando que en el apartado 2.3.2 se mencionan más de 20 modelos. Se sugiere introducir un párrafo que identifique cuáles tuvieron un carácter exploratorio."*

### Descripción de las modificaciones realizadas

Se ampliaron los párrafos introductorios de §3.4.3.1 "Conjunto inicial de modelos evaluados" para justificar explícitamente el conjunto seleccionado y su carácter exploratorio, mediante tres criterios complementarios:

1. **Cobertura de familias de clasificadores** discutidas en §2.3.2, de modo que la comparación inicial proporcionara una visión amplia del espacio de soluciones.
2. **Homogeneidad metodológica:** restricción a algoritmos disponibles de manera estandarizada en la librería LazyPredict, permitiendo aplicar un protocolo experimental homogéneo sobre configuraciones por defecto.
3. **Pertinencia al problema:** clasificadores supervisados de uso establecido en problemas de teledetección hiperspectral.

Se discute que la diferencia numérica entre los 12 algoritmos evaluados y el conjunto más amplio descrito en §2.3.2 obedece a la distinta finalidad de cada listado, justificando explícitamente la exclusión de técnicas de reducción de dimensionalidad (no son clasificadores), métodos no supervisados (no aplican a problema con etiquetas), métodos semi-supervisados y especializados (sin implementación en LazyPredict) y arquitecturas de Deep Learning (evaluadas posteriormente con protocolo dedicado).

Se explicita finalmente que la etapa cumplió un **carácter exploratorio**: su objetivo no fue identificar el modelo óptimo final, sino caracterizar el desempeño comparativo de las principales familias de clasificadores supervisados.

### Ubicación en el documento

- §3.4.3.1 "Conjunto inicial de modelos evaluados" — dos bloques de párrafos añadidos.

---

## Observación 10 — Profundización del análisis de Random Forest

### Texto original del jurado

> *"Se sugiere introducir un párrafo que identifique cuáles tuvieron un carácter exploratorio y profundizar en el análisis de Random Forest como referente comparativo debido a su alto desempeño (Tabla 4.1)."*

### Descripción de las modificaciones realizadas

Se atendió la solicitud mediante la reescritura ampliada del párrafo de descarte en §4.3.1 "Resultados exploratorios y descarte de algoritmos". Tras revisar las métricas reportadas en la Tabla 4-1 en conjunto, se identificó que el valor de accuracy de 0,74 reportado para Random Forest, aparentemente alto, debe interpretarse en el contexto del desbalance de clases del problema. El análisis de las demás métricas reportadas (ROC-AUC = 0,50, indistinguible del azar) revela que la decisión de descarte tomada en la fase exploratoria fue metodológicamente sólida.

El nuevo análisis documenta:

- **Diagnóstico técnico:** el accuracy de 0,74 coincide exactamente con el del DummyClassifier, y el ROC-AUC de 0,50 confirma que la configuración por defecto de Random Forest no logra discriminar mejor que el azar en presencia del desbalance del problema.
- **Costo computacional contrastado:** Random Forest requirió aproximadamente 754 segundos de entrenamiento, frente a los 10–15 segundos de LightGBM o XGBoost, una diferencia de prácticamente dos órdenes de magnitud.
- **Reconocimiento de la decisión:** la combinación de un desempeño discriminativo indistinguible del azar bajo configuración por defecto y un costo computacional sustancialmente mayor justificó el descarte del algoritmo.
- **Discusión análoga** de K-Nearest Neighbors (costo prohibitivo) y AdaBoost (replicó comportamiento del DummyClassifier).

### Ubicación en el documento

- §4.3.1 "Resultados exploratorios y descarte de algoritmos" — ampliación sustancial.

---

## Observación 11 — Hiperparametrización sin cambios significativos

### Texto original del jurado

> *"Además, debe ampliarse la discusión sobre por qué la hiperparametrización no aportó cambios significativos en tres de los modelos finales de ML."*

### Descripción de las modificaciones realizadas

Tras revisar las métricas reales registradas en MLflow, se constató que la observación del jurado es precisa: la búsqueda bayesiana de hiperparámetros mediante Optuna **no produjo cambios significativos en la métrica primaria PR-AUC en cinco de los seis modelos finales evaluados**, no solo en tres como originalmente señalado. Se reescribió la discusión en §4.4 para reflejar honestamente este hallazgo:

- **Constatación cuantitativa:** las variaciones observadas en PR-AUC se mantuvieron por debajo de 0,02 puntos para Regresión Logística, SGDClassifier, XGBoost, CNN-1D y CNN-2D.
- **Caso particular de LightGBM:** único modelo en el cual la optimización modificó apreciablemente el comportamiento, manifestándose como una recalibración del punto de operación (recall de ≈0,34 a ≈0,91) más que como mejora del PR-AUC (≈0,78 a ≈0,82).
- **Explicación estructural por familia de modelos:** modelos lineales acotados por separabilidad lineal; XGBoost con defaults conservadores; arquitecturas DL ya cerca del techo estructural; LightGBM con mayor sensibilidad al ajuste de parámetros.

Adicionalmente, se ajustaron las afirmaciones de las síntesis de ML y DL para que reflejen fielmente que las configuraciones iniciales adoptadas ya se aproximaban al desempeño máximo alcanzable, y se incorporó un nuevo bullet en la síntesis ML destacando el caso de LightGBM como hallazgo notable (recall de ≈0,34 a ≈0,91 sin mejora significativa de PR-AUC).

### Ubicación en el documento

- §4.4 "Comparación entre modelos" — reescritura del párrafo final sobre HPO.
- §4.3.4 "Resultados de los modelos CNN-2D" — ajuste del párrafo sobre comparación baseline vs final.
- §4.3.4 "Síntesis de los resultados de Deep Learning" — reformulación.
- §4.3.3 "Síntesis de los resultados de Machine Learning" — bullet adicional sobre LightGBM.

---

## Observación 12 — Importancia relativa de los falsos positivos frente a los falsos negativos

### Texto original del jurado

> *"En la discusión, se debe considerar la importancia de los falsos positivos frente a los falsos negativos."*

### Descripción de las modificaciones realizadas

Se ampliaron dos párrafos nuevos en §4.6 "Discusión de resultados" para abordar la importancia relativa de los errores de clasificación en el contexto operativo del trabajo, integrados al hilo argumentativo continuo de la discusión.

**Encuadre operativo y asimetría de costos.** Bajo el alcance operativo planteado en el Capítulo 1 (detección temprana para intervención agronómica), los falsos positivos y falsos negativos no implican costos equivalentes:

- Un **falso negativo** se traduce en una microzona del cultivo que no recibe atención cuando la requiere. Considerando que la fertilización fosfórica adecuada puede incrementar el rendimiento del fríjol común hasta en un 38 % (Y. Gao et al., 2016), el costo agronómico de un FN no detectado se materializa en pérdidas de productividad difícilmente recuperables.
- Un **falso positivo** desencadena una intervención local sobre una zona que no la requería; el costo es económico y ambiental, de magnitud menor.

Se incorpora la cita de **Breure et al. (2022)** para sustentar el marco general de funciones de pérdida asimétricas en agricultura de precisión.

**Perfiles operativos diferenciados de los modelos:**
- CNN-2D optimizado: balance entre precisión (0,895) y recall (0,886); adecuado para caracterización y apoyo a decisiones.
- LightGBM optimizado: recall ≈ 0,91, precisión ≈ 0,75; orientado a maximizar sensibilidad.
- Modelos lineales: recall en el rango 0,68–0,70; referencias metodológicas.

Se concluye argumentando que la elección del modelo para despliegue operativo depende de la prioridad agronómica del usuario, y que el ajuste fino del umbral de decisión permite modular el balance FP/FN según el contexto.

**Citas bibliográficas incorporadas:** Y. Gao et al. (2016) — referencia ya presente en el Capítulo 1; Breure et al. (2022) — cita nueva añadida a la bibliografía.

### Ubicación en el documento

- §4.6 "Discusión de resultados" — dos párrafos nuevos integrados al hilo argumentativo continuo.

---

## Observación 13 — Caso práctico de parcelas con plantas mixtas

### Texto original del jurado

> *"ampliar el análisis para casos prácticos donde coexistan plantas con y sin estrés visible en una misma parcela y su posible afectación en el modelo CNN-2D."*

### Descripción de las modificaciones realizadas

Se incorporaron dos párrafos adicionales a §4.6 abordando específicamente el comportamiento del modelo CNN-2D ante parcelas con heterogeneidad intra-parcela.

**Tres elementos del diseño que anticipan comportamiento favorable:**

1. **Granularidad de predicción adecuada:** la unidad de predicción del modelo es el píxel, no la parcela. Cada decisión del CNN-2D corresponde a una microzona de aproximadamente 10 × 10 cm sobre el terreno, escala inferior a una planta individual. Ante una parcela con mosaico de plantas estresadas y sanas, el modelo produce un mapa de probabilidades de igual granularidad.

2. **Heterogeneidad ya documentada:** el análisis desagregado por genotipo (§4.5) mostró desempeños entre PR-AUC = 0,824 y 0,993, incluyendo casos de discriminación más difícil como el genotipo L17-G51018 (recall = 0,497).

3. **Decisión dominada por la firma espectral del píxel:** la prueba de ablación del píxel central (§4.5.3) demostró que la firma espectral individual aporta la mayor parte de la capacidad discriminativa (PR-AUC = 0,815), mientras que el contexto espacial local añade aproximadamente 0,15 puntos. La presencia de plantas sanas en la vecindad de plantas estresadas no debería arrastrar sistemáticamente las predicciones hacia una sola clase.

Se reconoce que la limitación más relevante para parcelas mixtas no proviene del modelo sino de la naturaleza binaria del etiquetado: las plantas con grados intermedios de estrés quedan absorbidas dentro de la clase positiva. La calibración del umbral de decisión ofrece un mecanismo operativo concreto. La extensión hacia un esquema multiclase se identifica en §5.2 como línea prioritaria de trabajo futuro.

### Ubicación en el documento

- §4.6 "Discusión de resultados" — dos párrafos adicionales integrados al texto argumentativo continuo.

---

# Bloque III — Observaciones de análisis adicional

---

## Observación 16 — Matrices de confusión en porcentajes y reducción de datos en CNN-2D

### Texto original del jurado

> *"Para facilitar la interpretación, se propone cambiar las matrices de confusión por porcentajes ya permite identificar la mejora significativa del modelo CNN-2D ya que este modelo presenta menor cantidad de datos, también es importante aclarar al lector que no esté familiarizado motivo de la reducción de datos en la matriz de confusión."*

### Descripción de las modificaciones realizadas

Se incorporó al final de §4.3.4.3 "Análisis de las matrices de confusión" un análisis comparativo en términos porcentuales de los seis modelos finales, acompañado de una nueva figura editorial con seis paneles (uno por modelo) que presenta las matrices normalizadas por fila (recall por clase).

**Justificación de la representación porcentual.** Se argumenta que la normalización por fila desacopla la lectura de la magnitud absoluta de cada clase, lo cual es recomendado en problemas con desbalance de clases.

**Lectura comparativa de los modelos:**
- Modelos lineales (LR, SGD): recall positivo 67–69 %, FPR cercano al 60 % (selectividad limitada).
- LightGBM tras HPO: recall 91,23 %, FPR 90,27 % (umbral calibrado para maximizar detección).
- XGBoost y CNN-1D: perfiles intermedios (recall ≈ 72 %, FPR 55–58 %).
- CNN-2D: balance más favorable (recall 88,61 %, especificidad 66,96 %, FPR 33,04 %, FNR 11,39 %).

**Explicación de la reducción de datos en CNN-2D.** Se aclara explícitamente que el CNN-2D evalúa 251.856 píxeles frente a 361.810 de los demás modelos (aproximadamente el 69,6 %), debido a la geometría de los parches de 5×5 píxeles requeridos por la arquitectura: los píxeles ubicados a menos de dos posiciones del borde de un surco, de una zona enmascarada o de los límites del lote no pueden constituir un parche completo. Esta reducción es una consecuencia geométrica del diseño, no una decisión metodológica de filtrado, y afecta de manera homogénea a ambas clases.

### Ubicación en el documento

- §4.3.4.3 "Análisis de las matrices de confusión" — bloque final con análisis y Figura 4-10.

---

## Observación 17 — Validación empírica de la influencia de los índices de vegetación

### Texto original del jurado

> *"validar la influencia real de los índices de vegetación calculados frente a las características utilizadas en el proceso"*

### Descripción de las modificaciones realizadas

Se incorporó al Capítulo 4 una nueva subsección **§4.3.5 "Validación empírica del aporte de los índices de vegetación"**, dedicada a cuantificar el aporte real de los cinco índices de vegetación (NDVI, NDRE, CIgreen, PRI, PSRI) frente a las 58 bandas espectrales seleccionadas. El análisis se realizó mediante ablación en tiempo de inferencia, sin reentrenamiento.

**Protocolo metodológico.** Para cada uno de los seis modelos finales sobre el conjunto de prueba, se evaluó el desempeño bajo tres configuraciones principales:
- **C0 (línea base):** todas las características disponibles (58 bandas + 5 IV).
- **C1 (todas las bandas):** los IV se sustituyen por sus promedios calculados sobre el conjunto de entrenamiento.
- **C2 (IV originales):** las 58 bandas se sustituyen por sus promedios del entrenamiento.

Adicionalmente, se evaluaron cinco subcondiciones que neutralizan un solo IV a la vez. La sustitución por promedios de entrenamiento —en lugar de ceros— evita introducir valores fuera del dominio aprendido por los modelos.

**Resultados:**

| Modelo | C0 (todas) | C1 (sin IV) | C2 (solo IV) | Δ (C0 − C1) | Δ % |
|---|---|---|---|---|---|
| Regresión Logística | 0,7902 | 0,7811 | 0,7652 | 0,0091 | 1,15 |
| SGDClassifier | 0,7885 | 0,7785 | 0,7649 | 0,0100 | 1,27 |
| LightGBM | 0,7824 | 0,7699 | 0,7441 | 0,0125 | 1,60 |
| XGBoost | 0,8236 | 0,7975 | 0,7714 | 0,0261 | 3,17 |
| CNN-1D | 0,8341 | 0,8189 | 0,7637 | 0,0152 | 1,82 |
| **CNN-2D** | **0,9635** | **0,9105** | **0,8018** | **0,0530** | **5,50** |

**Tres observaciones diferenciadas:**

1. **Para LR, SGD, LightGBM y CNN-1D** (Δ < 0,02): los IV capturan información ampliamente redundante con las 58 bandas seleccionadas.
2. **Para XGBoost** (Δ = 0,0261): aporte complementario modesto. El NDRE es el índice de mayor influencia individual.
3. **Para CNN-2D** (Δ = 0,0530, 5,50 %): los IV son informativamente significativos. En la configuración C2 (solo IV), el PR-AUC se mantiene en 0,8018. El NDRE domina la contribución individual (Δ = 0,0733), seguido de NDVI (0,0344) y CIgreen (0,0322). PRI y PSRI son marginales para todas las arquitecturas.

**Conclusión:** la ablación valida la pertinencia de incluir los IV en el vector de características, particularmente para arquitecturas con capacidad espectro-espacial, y permite priorizar NDRE, NDVI y CIgreen como los más relevantes.

### Ubicación en el documento

- §4.3.5 "Validación empírica del aporte de los índices de vegetación" — subsección nueva con Tabla 4-4 y Figuras 4-11 y 4-12.

---

## Observación 18 — Métricas de costo computacional para los modelos de Deep Learning

### Texto original del jurado

> *"Si bien se menciona el costo computacional como causa para no profundizar en él, este criterio debe discutirse más a fondo dada la naturaleza del caso de estudio y las capacidades de cómputo actuales. (...) debe (...) incluir las métricas de costo computacional para los modelos de Deep Learning, ya que como fue expuesto en el texto anteriormente es un criterio relevante para la comparación de resultados."*

### Descripción de las modificaciones realizadas

Se incorporó al Capítulo 4 una nueva subsección **§4.4.1 "Costo computacional"** dentro de la sección comparativa §4.4 "Comparación entre modelos", presentando una caracterización cuantitativa del costo computacional asociado a los seis modelos finales evaluados. Aunque la observación solicita explícitamente las métricas para los modelos de Deep Learning, se incluyeron también los cuatro modelos de Machine Learning para reforzar la comparativa.

**Estructura de la caracterización.** La caracterización combina dos fuentes de información:
- **Métricas extraídas de MLflow:** tiempos de entrenamiento del run final, tiempo total acumulado durante la búsqueda de hiperparámetros.
- **Mediciones empíricas locales:** conteo de parámetros, tamaño de los pesos serializados, memoria pico durante inferencia, latencia y rendimiento.

**Hallazgos principales:**
- En **costo de entrenamiento**, los modelos lineales y LightGBM presentan tiempos del orden de minutos. Los modelos de Deep Learning presentan los tiempos más extensos (hasta 15h 12min para CNN-1D durante HPO).
- En **costo de inferencia**, todos los modelos presentan latencias inferiores a 15 milisegundos por lote de 512 muestras y rendimientos superiores a 35.000 muestras por segundo. La extrapolación para procesar un millón de píxeles oscila entre 1,5 segundos para CNN-1D y 27,9 segundos para XGBoost.

**Discusión contextual.** Se incorpora un párrafo de cierre que argumenta que la consideración explícita del costo computacional debe interpretarse en el contexto agronómico actual, donde la disponibilidad de cómputo accesible —incluyendo aceleración por GPU en equipos de gama media— ha modificado los compromisos tradicionales entre complejidad del modelo y viabilidad práctica.

### Ubicación en el documento

- §4.4.1 "Costo computacional" — subsección nueva con Tabla 4-3 y Figura 4-14.

---

## Observación 19 — Diagrama del diseño experimental

### Texto original del jurado

> *"Se recomienda incluir un diagrama del diseño experimental."*

### Descripción de las modificaciones realizadas

Se incorporó a §3.1.1 "Origen y condiciones del experimento" un diagrama de la estructura factorial del diseño experimental. Adicionalmente, se reorganizó y consolidó la descripción del diseño experimental que aparecía dispersa en la sección.

**Figura nueva (Figura 3-2).** La figura presenta una matriz de 8 genotipos × 4 niveles de fertilización con P₂O₅ × 3 repeticiones, representando las 96 unidades experimentales nominales del experimento. Cada celda muestra los tres puntos correspondientes a las repeticiones, con código de color por nivel de fertilización.

### Ubicación en el documento

- §3.1.1 "Origen y condiciones del experimento" — texto reorganizado y Figura 3-2 nueva.

---

# Bloque IV — Observaciones de estructura

---

## Observación 14 — Reestructuración de las conclusiones

### Texto original del jurado

> *"se recomienda reestructurar las conclusiones para evidenciar el cumplimiento de cada objetivo específico. Se deben abordar las limitaciones, la justificación de la clasificación binaria frente al desbalance de clases y si contemplo solo comprar respecto a un tratamiento para evitar dicho desbalance"*

### Descripción de las modificaciones realizadas

Se reescribió completamente la sección §5.1 "Conclusiones", pasando de párrafos descriptivos a una estructura organizada en torno a cada uno de los objetivos del trabajo y cerrando con una subsección dedicada a limitaciones:

- **§5.1.1 — Cumplimiento del OE1: Base de datos hiperespectral etiquetada.**
- **§5.1.2 — Cumplimiento del OE2: Caracterización de la huella hiperespectral del estrés por fósforo.**
- **§5.1.3 — Cumplimiento del OE3: Selección de un modelo de IA supervisado.**
- **§5.1.4 — Cumplimiento del objetivo general.**
- **§5.1.5 — Limitaciones del estudio.**

La subsección de limitaciones aborda explícitamente los tres puntos solicitados por el jurado:

1. **Adopción de un enfoque de clasificación binaria.** Se documentan tres criterios que motivaron la decisión: línea base reproducible, preservación del tamaño muestral por clase, y relevancia operativa (distinción primaria entre "estado nutricional adecuado" y "presencia de algún grado de deficiencia").

2. **Comparación frente a un único tratamiento de referencia.** Se documenta que la binarización produjo la clase negativa exclusivamente con T4 (dosis óptima) y la clase positiva con T1+T2+T3, generando el desbalance del 70 % de clase positiva. Se evaluó y se descartó la alternativa de muestrear un único tratamiento de estrés frente al control T4 por dos razones: reducción del volumen de datos y restricción de la generalización a un único nivel de deficiencia.

3. **Estratificación de la partición espacial.** La estratificación únicamente por etiqueta binaria dejó algunos entries fuera del holdout. Se identifica como limitación reconocida.

Adicionalmente, se documentan dos limitaciones complementarias: captura única en una sola fase fenológica y especificidad del estrés evaluado (solo deficiencia de fósforo).

### Ubicación en el documento

- §5.1 "Conclusiones" — sección completamente reescrita con cinco subsecciones (5.1.1 a 5.1.5).

---

## Observación 15 — Trabajo futuro: validación ante otros estreses y genotipos

### Texto original del jurado

> *"respecto al trabajo futuro se identifica la necesidad de validar el modelo ante otros tipos de estrés (bióticos y abióticos) para dar respuesta a la generalidad del título propuesto. Dado que el enfoque actual se limita al estrés por fósforo, la validación futura en diversos escenarios es indispensable para respaldar la tesis general del trabajo, también integrar la información de los genotipos su afectación sobre los resultados obtenidos."*

### Descripción de las modificaciones realizadas

Se reformuló el párrafo introductorio de §5.2 "Recomendaciones y trabajo futuro" y se incorporaron tres nuevos ítems en posición prioritaria que responden directamente a los tres puntos planteados por el jurado:

1. **Validación ante otros tipos de estrés abiótico.** Propone extender el enfoque hacia deficiencias de nitrógeno (con manifestación espectral parcialmente solapada con la del fósforo), deficiencia de potasio y estrés hídrico. Argumenta que la proximidad fisiológica de algunos de estos estreses al de fósforo permitiría explorar esquemas de aprendizaje multitarea o transferencia de modelos.

2. **Validación ante estreses bióticos.** Propone evaluar la aplicabilidad a enfermedades fungosas, enfermedades virales y daños por plagas, y la extensión hacia un sistema de diagnóstico multiclase capaz de discriminar diferentes orígenes del estrés.

3. **Integración del genotipo como variable predictiva y estudios genotipo-específicos.** Propone la incorporación del genotipo como variable de entrada adicional al modelo, aprovechando las diferencias documentadas en firmas espectrales basales (Sección 4.5).

### Ubicación en el documento

- §5.2 "Recomendaciones y trabajo futuro" — párrafo introductorio reformulado y tres ítems nuevos en posición prioritaria.

---

# Bloque V — Observaciones de forma

---

## Observación 1 — Normalización de separadores decimales

### Texto original del jurado

> *"es necesario realizar ajustes para normalizar el uso de los separadores decimales en el texto, gráficas y tablas"*

### Descripción de las modificaciones realizadas

Se realizó una pasada sistemática sobre el documento completo aplicando las convenciones del español académico: coma como separador decimal, punto como separador de miles, y espacio antes de las unidades. Se aplicaron 238 cambios en total, clasificados por tipo:

- **Decimales < 1 (formato 0,X):** 138 cambios (ej. `0.963` → `0,963`).
- **Decimales pequeños (formato N,M):** 33 cambios (ej. `1.43` → `1,43`).
- **Separadores de miles para números grandes:** 28 cambios (ej. `12,583,080` → `12.583.080`).
- **Inserción de separadores en enteros sin separador:** 6 cambios (ej. `2215` → `2.215`).
- **Espacios antes del símbolo de porcentaje:** 33 cambios (ej. `78%` → `78 %`).

Las correcciones respetaron los siguientes ámbitos sin modificación: DOIs y referencias bibliográficas, identificadores técnicos, ecuaciones matemáticas, URLs y nombres de versión de software.

### Ubicación en el documento

- Todo el documento, especialmente las secciones de Resultados (§4) y los pies de tabla y figura.

---

## Observación 2 — Corrección ortográfica y de digitación

### Texto original del jurado

> *"corregir errores ortográficos y de digitación señalados en la copia del documento con comentarios"*

### Descripción de las modificaciones realizadas

Se atendieron las correcciones señaladas explícitamente en el PDF anotado del jurado, además de realizar una pasada general de uniformidad terminológica:

**Correcciones aplicadas:**

| Origen | Original | Reemplazo |
|---|---|---|
| Tabla de notación (p. 14) | `K = Nitrógeno` / `N = Potasio` | `K = Potasio` / `N = Nitrógeno` |
| Concordancia de género (p. 30) | `es requerido` | `es requerida` |

**Uniformidad terminológica:** se unificó el uso de los términos "fríjol" (con tilde, forma RAE para Colombia) y "huella espectral" (coherente con el título de la tesis), reemplazando 24 instancias de "frijol" por "fríjol" y 14 instancias de "firma espectral" por "huella espectral" en el cuerpo del texto. Las referencias bibliográficas conservaron los términos originales de las publicaciones.

### Ubicación en el documento

- Todo el documento.

---

## Observación 3 — Redundancia en la definición de imágenes hiperespectrales

### Texto original del jurado

> *"Se observa una redundancia informativa respecto al concepto de imágenes hiperespectrales en al menos tres secciones, por lo que se recomienda unificar esta definición en una única presentación."*

### Descripción de las modificaciones realizadas

Se identificaron tres apariciones donde se definía conceptualmente el término "imágenes hiperespectrales (HSI)" con distintos niveles de detalle: dos en la Justificación (§1.2) y una en el Marco teórico (§2.2). Se consolidó la definición técnica completa en §2.2 (incluyendo rangos espectrales VIS/NIR/SWIR, conceptos de reflectancia/absorbancia/emitancia y "huella dactilar única"), y se redujeron las menciones de §1.2 a una versión condensada con referencia cruzada explícita a la sección 2.2.

**Cambios realizados:**

- §1.2 (Justificación, primer párrafo): condensado a una versión breve con remisión a §2.2.
- §1.2 (Justificación, segundo párrafo): ajuste mínimo añadiendo referencia cruzada `(ver Sección 2.2)`.
- §2.2 (Marco teórico): expandido para constituir la definición canónica completa y auto-suficiente.

### Ubicación en el documento

- §1.2 "Justificación" — definiciones reducidas con remisión a §2.2.
- §2.2 "Percepción remota e imágenes hiperespectrales" — definición canónica expandida.

---

## Recursos de soporte para la verificación

Las modificaciones realizadas se acompañan de los siguientes recursos en el repositorio público del trabajo (https://dagshub.com/johnma96/thesis):

| Recurso | Ubicación | Soporta |
|---|---|---|
| Notebook 201 — Exploración inicial de PCA | `notebooks/201-jmmz-pca.ipynb` | Observación 6 |
| Notebook 202 — Selección de bandas espectrales | `notebooks/202-jmmz-band-selection.ipynb` | Observación 6 |
| Notebook 401 — Análisis por genotipo | `notebooks/401-jmmz-genotype-analysis.ipynb` | Observación 20 |
| Notebook 402 — Pruebas de ablación espectral | `notebooks/402-jmmz-ablation-tests.ipynb` | Observación 20 |
| Notebook 403 — Costo computacional | `notebooks/403-jmmz-computational-cost.ipynb` | Observación 18 |
| Notebook 404 — Matrices de confusión normalizadas | `notebooks/404-jmmz-confusion-matrices-pct.ipynb` | Observación 16 |
| Notebook 405 — Ablación de índices de vegetación | `notebooks/405-jmmz-iv-ablation.ipynb` | Observación 17 |
| Tracking experimental MLflow | https://dagshub.com/johnma96/thesis.mlflow | Todas las observaciones de modelado |

---

## Referencias bibliográficas incorporadas durante el proceso de corrección

Se relacionan a continuación las cinco referencias bibliográficas nuevas integradas a la bibliografía del trabajo durante la atención de las observaciones del jurado:

- **Breure, T. S., Haefele, S. M., Hannam, J. A., Corstanje, R., Webster, R., Moreno-Rojas, S., & Milne, A. E.** (2022). A loss function to evaluate agricultural decision-making under uncertainty: A case study of soil spectroscopy. *Precision Agriculture, 23*(5), 1942–1971. https://doi.org/10.1007/s11119-022-09887-2 — Observación 12.
- **Kattenborn, T., Schiefer, F., Frey, J., Feilhauer, H., Mahecha, M. D., & Dormann, C. F.** (2022). Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks. *ISPRS Open Journal of Photogrammetry and Remote Sensing, 5*, 100018. https://doi.org/10.1016/j.ophoto.2022.100018 — Observación 20.
- **Meneses-Tovar, C. L.** (2011). NDVI as indicator of degradation. *Unasylva, 62*(238), 39–46. https://www.fao.org/4/i2560e/i2560e07.pdf — Observación 4.
- **Okyere, F. G., Cudjoe, D., Sadeghi-Tehran, P., Virlet, N., Riche, A. B., Castle, M., Greche, L., Simms, D., Mhada, M., Mohareb, F., & Hawkesford, M. J.** (2023). Modeling the spatial-spectral characteristics of plants for nutrient status identification using hyperspectral data and deep learning methods. *Frontiers in Plant Science, 14*, 1209500. https://doi.org/10.3389/fpls.2023.1209500 — Observación 20.
- **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881 — Observación 20.

---

## Cierre

Las modificaciones documentadas en esta carta corresponden a la totalidad de las veinte observaciones del jurado evaluador planteadas el 13 de abril de 2026. El documento se entrega dentro del plazo establecido del 12 de mayo de 2026.

Quedo a disposición del jurado evaluador para resolver cualquier inquietud adicional sobre las modificaciones realizadas.

Atentamente,

**John Mario Montoya Zapata**
Estudiante de Maestría
[Programa académico]
Universidad Nacional de Colombia