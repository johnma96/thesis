# Carta de respuesta a las observaciones del jurado evaluador

**Trabajo Final de Maestría:** *Diagnóstico no invasivo del estado de salud del fríjol común (Phaseolus vulgaris L.) en Colombia: Un enfoque basado en la huella espectral y la inteligencia artificial.*

**Estudiante:** John Mario Montoya Zapata
**Director(a):** [completar]
**Jurado evaluador:** Manuel Mauricio Goez Mora — Instituto Tecnológico Metropolitano (ITM)
**Fecha de las observaciones:** 13 de abril de 2026
**Fecha de respuesta:** [completar]

---

## Presentación

Se presenta a continuación la respuesta a las observaciones formuladas por el jurado evaluador sobre el Trabajo Final de Maestría. Cada observación se transcribe literalmente y se acompaña de la descripción de las modificaciones realizadas en el documento, así como de la ubicación específica de los cambios. En todos los casos, las modificaciones se han incorporado al documento final preservando los resultados experimentales originales del trabajo, sin nuevos entrenamientos ni modificaciones de los modelos previamente registrados.

Las observaciones se han agrupado por afinidad temática y se han atendido siguiendo un orden de prioridad determinado por su impacto sobre la coherencia general del documento. Para facilitar la verificación, las referencias a secciones, figuras y tablas corresponden a la numeración del documento corregido.

Agradezco al jurado evaluador la rigurosidad y profundidad de las observaciones planteadas, las cuales han contribuido de manera sustancial al fortalecimiento metodológico y argumentativo del trabajo.

---

## Tabla de seguimiento de observaciones

| N° | Observación | Categoría | Estado | Sección modificada |
|---|---|---|---|---|
| 1 | Normalización de separadores decimales | Forma | Pendiente | Todo el documento |
| 2 | Corrección ortográfica y de digitación | Forma | Pendiente | Todo el documento |
| 3 | Redundancia en la definición de imágenes hiperespectrales | Forma | Pendiente | §1, §2, §3 |
| 4 | Justificación del umbral NDVI | Metodología | Pendiente | §3.2.x |
| 5 | Criterio de "alto bienestar" y representatividad | Metodología | Pendiente | §3.2.x |
| 6 | Ampliación de selección de bandas y métodos descartados | Metodología | **Atendida** | §3.3 |
| 7 | Ventaja técnica de la división validación–prueba | Metodología | **Atendida** | §3.4.2.2 |
| 8 | Aclaración sobre la única toma aérea | Metodología | **Atendida** | §3.1.2 |
| 9 | Justificación de los 12 algoritmos iniciales | Metodología | **Atendida** | §3.4.3.1 |
| 10 | Profundización del análisis de Random Forest | Modelado | Pendiente | §4.x |
| 11 | Hiperparametrización sin cambios significativos | Modelado | **Atendida** | §4.2 |
| 12 | Importancia relativa de FP vs FN | Discusión | **Atendida** | §4.6 |
| 13 | Caso práctico: parcelas con plantas mixtas | Discusión | Pendiente | §4.x |
| 14 | Reestructuración de conclusiones | Estructura | Pendiente | §5 |
| 15 | Trabajo futuro: validación ante otros estreses y genotipos | Estructura | Pendiente | §5 |
| 16 | Matrices de confusión en porcentajes y reducción de datos | Análisis | Pendiente | §4.x |
| 17 | Influencia real de los índices de vegetación | Análisis | Pendiente | §4.x |
| 18 | Métricas de costo computacional para modelos DL | Análisis | **Atendida** | §4.4.1 |
| 19 | Diagrama del diseño experimental | Análisis | **Atendida** | §3.1.1 |
| 20 | Riesgo de aprendizaje de estructura espacial en CNN-2D | Modelado | **Atendida** | §4.3 |

**Observaciones atendidas en esta entrega:** 10 de 20.
**Observaciones pendientes:** 10 (en proceso de atención).

---

# Bloque I — Observaciones metodológicas críticas

---

## Observación 20 — Posible aprendizaje de la estructura espacial por parte del modelo CNN-2D

### Texto original del jurado

> *"Respecto al modelado, el desempeño del modelo CNN-2D con presento un PR-AUC de 0.963 resulta excepcionalmente alto, lo que sugiere la necesidad de revisar si la red está aprendiendo la estructura espacial de las parcelas en lugar de la firma espectral del estrés, dado el etiquetado por polígonos manuales. Es fundamental mencionar si el modelo incluyó el 'Genotipo' como variable o si se evaluó el desempeño por variedad, considerando que las firmas espectrales varían naturalmente entre los 8 genotipos utilizados; en este sentido, debe aclararse si la omisión de este criterio en la conformación de los conjuntos de validación y prueba genera alguna afectación."*

### Descripción de las modificaciones realizadas

Se incorporó al Capítulo 4 una nueva sección titulada **"4.3 Análisis de robustez del modelo CNN-2D"**, dedicada exclusivamente a evaluar empíricamente la inquietud planteada por el jurado. La sección se estructura en tres análisis complementarios sobre el modelo final, ejecutados sin reentrenamiento alguno y utilizando exclusivamente el conjunto de prueba (espacialmente disjunto del conjunto de entrenamiento por construcción).

**Análisis 1 — Desempeño desagregado por genotipo.** Se aclaró explícitamente que el genotipo no fue utilizado como variable de entrada en ninguno de los modelos desarrollados; la red CNN-2D recibió únicamente el cubo hiperspectral compuesto por las 58 bandas seleccionadas y los 5 índices de vegetación (NDVI, NDRE, CIgreen, PRI, PSRI). Sobre el conjunto de prueba se desagregaron las métricas por entrada (entry) y se observó una variación de aproximadamente 0,17 puntos de PR-AUC entre genotipos (rango: 0,824 para Cargamanto hasta 0,993 para L2-G11819). Esta heterogeneidad es **incompatible con la hipótesis de memorización espacial** —que produciría desempeño uniforme entre genotipos por independencia entre la información geométrica y la varietal— y **consistente con un modelo que aprende la firma espectral del estrés**, cuya manifestación bioquímica varía entre variedades por diferencias en pigmentos, contenido de agua y arquitectura foliar. El caso del entry 6 (L17-G51018), con un recall de 0,497, refuerza esta interpretación: un modelo que hubiera memorizado la geometría no presentaría diferencias de desempeño entre variedades.

**Análisis 2 — Verificación de identificadores no documentados.** Se identificaron en los datos crudos las entradas 9 y 10, no incluidas en la Tabla 8 del Anexo 32, presentes únicamente en la clase 0 (no estresada). Su existencia se reporta de manera transparente en el documento. Estas entradas no aparecen en el conjunto de prueba y, por tanto, no afectan las métricas reportadas.

**Análisis 3 — Pruebas de ablación espectral.** Se diseñaron y ejecutaron dos pruebas para aislar la contribución del contenido espectral frente a la del contexto espacial:

- **Prueba 1 (permutación espectral):** se permutó aleatoriamente el orden de las bandas en cada parche del conjunto de prueba, preservando la estructura espacial pero destruyendo la coherencia espectral. El ROC-AUC colapsó a 0,5056, indistinguible del azar.
- **Prueba 2 (píxel central):** se evaluó el modelo replicando el espectro del píxel central en todas las posiciones del parche 5×5, conservando información espectral pero eliminando el contexto espacial. El PR-AUC obtenido (0,815) resulta prácticamente equivalente al de la arquitectura CNN-1D real (0,83), que opera por construcción sobre un vector espectral sin vecindad.

La diferencia entre el PR-AUC global del CNN-2D (0,964) y el obtenido en la Prueba 2 (0,815) cuantifica en aproximadamente 0,15 puntos la ganancia atribuible al contexto espacial local, ganancia que corresponde a información espectral aportada por los píxeles vecinos dentro del mismo surco —no a memorización de la geometría de los polígonos. Esta interpretación es coherente con la escala del campo receptivo del modelo: un parche de 5×5 píxeles representa solo 10×10 cm sobre el terreno, frente a un surco típico de 3,6 × 1 m equivalente a aproximadamente 9.000 píxeles, por lo que la red dispone de información local suficiente para integrar textura y vecindad espectral pero carece de la cobertura espacial necesaria para reconstruir la forma de los polígonos manualmente delimitados.

**Sobre el efecto de la no estratificación por genotipo en los splits.** Se reconoce explícitamente como una limitación del diseño experimental que la partición espacial fue estratificada únicamente por la etiqueta binaria y no por genotipo, motivo por el cual los entries 4, 9 y 10 no aparecen en el holdout. Esta limitación se documenta en el texto y se incorpora al apartado de trabajo futuro como una línea de mejora metodológica para estudios subsiguientes.

**Citas bibliográficas incorporadas:** Roberts et al. (2017) sobre estrategias de validación cruzada para datos con estructura espacial; Kattenborn et al. (2022) sobre la inflación del desempeño en CNN cuando hay autocorrelación espacial entre muestras de entrenamiento y validación; Okyere et al. (2023) sobre la complementariedad espectro-espacial en modelos profundos para identificación del estado nutricional con HSI.

### Ubicación en el documento

- §4.3 "Análisis de robustez del modelo CNN-2D" (sección nueva).
- Subsecciones: 4.3.1 Desempeño desagregado por genotipo; 4.3.2 Verificación de identificadores no documentados; 4.3.3 Pruebas de ablación espectral.
- Notebooks de soporte en el repositorio: `401-jmmz-genotype-analysis.ipynb` y `402-jmmz-ablation-tests.ipynb`.

---

# Bloque II — Observaciones metodológicas

---

## Observación 6 — Ampliación de la selección de bandas y métodos descartados

### Texto original del jurado

> *"En cuanto a la selección de bandas, es relevante ampliar la descripción de la técnica de reducción de dimensionalidad y sintetizar las razones por las cuales no se contemplaron otros métodos."*

### Descripción de las modificaciones realizadas

Se ampliaron los párrafos introductorios de §3.3 con tres bloques nuevos que documentan de manera transparente la exploración previa de técnicas clásicas de reducción de dimensionalidad y la justificación del cambio de estrategia hacia la selección de bandas adoptada en el trabajo:

**Primer bloque — Métodos clásicos considerados.** Se describe que, antes de adoptar la estrategia finalmente implementada, se evaluó la viabilidad de aplicar técnicas clásicas (PCA, ICA, LDA y variantes basadas en kernel), referenciadas en la Sección 2.3.2. Se contextualiza el problema computacional implicado: una matriz aproximada de 12 millones de píxeles válidos por 379 bandas espectrales.

**Segundo bloque — Limitaciones operativas observadas.** Se documenta que la descomposición en valores singulares requerida por PCA estándar no resultó tratable sobre la matriz completa en el equipo de cómputo utilizado, debido a la combinación de alta dimensionalidad espectral, gran volumen de píxeles y fuerte correlación entre bandas adyacentes. Se reporta el intento posterior con `IncrementalPCA` y muestreo aleatorio estratificado de píxeles (documentado en el Notebook 201 del repositorio), señalando que esta vía implicó la introducción de decisiones adicionales (tamaño de muestra, tamaño de lote, estrategia de exclusión de ventanas de absorción de agua) cuya sensibilidad sobre las componentes resultantes comprometía la reproducibilidad del procedimiento. Adicionalmente, se documenta una limitación interpretativa común a todas las técnicas mencionadas: las características generadas son combinaciones lineales o no lineales de las bandas originales, lo cual dificulta su asociación directa con procesos fisiológicos conocidos y la posterior construcción de índices de vegetación canónicos (NDVI, NDRE, PRI), elementos clave para conectar los hallazgos con el conocimiento agronómico del fenómeno.

**Tercer bloque — Justificación de la estrategia adoptada.** Se explicita que, frente a estas limitaciones, se optó por una selección informada de bandas espectrales que conservara las bandas originales —preservando su interpretabilidad fisiológica— y aprovechara la estructura de correlación espectral característica de los datos hiperespectrales para reducir la redundancia. La estrategia se fundamenta en métricas estadísticas (relación señal-ruido y longitud de decorrelación espectral) y criterios fisiológicos (anclas en regiones asociadas a procesos de absorción y reflexión vegetal conocidos), implementada de forma reproducible mediante la función `select_bands` (Notebook 202).

**Citas bibliográficas incorporadas:** S. Li et al. (2019), Licciardi et al. (2012), Prasad & Bruce (2008) — referencias ya presentes en §2.3.2 que se conectan ahora con la discusión metodológica.

### Ubicación en el documento

- §3.3 "Selección de bandas espectrales relevantes" — tres párrafos nuevos al inicio de la sección.

---

## Observación 7 — Ventaja técnica de la división validación–prueba

### Texto original del jurado

> *"También debe acentuarse la ventaja técnica de dividir los datos de validación y prueba, considerando que provienen de la misma muestra y conjunto de datos."*

### Descripción de las modificaciones realizadas

Se insertaron dos párrafos nuevos en §3.4.2.2 "Partición espacial del conjunto de datos", explicitando las funciones diferenciadas de cada subconjunto y la ventaja técnica de la separación, aun cuando los tres provienen de la misma muestra:

**Primer párrafo — Funciones diferenciadas.** Se documenta que el subconjunto de entrenamiento se utiliza exclusivamente para ajustar los parámetros internos del modelo. El subconjunto de validación cumple tres funciones complementarias: (i) selección de hiperparámetros durante la búsqueda con Optuna (sección 3.4.4); (ii) monitoreo de convergencia y aplicación del criterio de detención temprana (early stopping) en los modelos de Deep Learning (sección 3.4.5); (iii) calibración del umbral de decisión sobre las probabilidades predichas. El subconjunto de prueba queda reservado para una única evaluación final del modelo ya seleccionado y configurado, garantizando que la métrica reportada no esté contaminada por decisiones de modelado informadas por su contenido.

**Segundo párrafo — Ventaja técnica.** Se argumenta que la tripartición permite separar el proceso de selección y ajuste del modelo del proceso de evaluación de su capacidad de generalización. Se discute que, si se prescindiera del subconjunto de validación y se utilizara directamente el de prueba para guiar la optimización de hiperparámetros o la selección del umbral, los valores finales reportados incorporarían un sesgo optimista derivado del ajuste indirecto a sus particularidades estadísticas. Este sesgo, conocido en la literatura como sobreajuste al conjunto de prueba, es particularmente relevante en escenarios con desbalance de clases y datos espacialmente correlacionados como los del presente trabajo.

Adicionalmente, se ajustó el párrafo final de §3.4.2.2 para integrar la idea de que la combinación entre partición espacial y separación funcional entre validación y prueba protege simultáneamente la validez externa (sin fuga espacial) y la validez interna (sin contaminación entre selección y evaluación) de los resultados reportados.

### Ubicación en el documento

- §3.4.2.2 "Partición espacial del conjunto de datos" — dos párrafos nuevos insertados después de la definición de las proporciones globales (60/20/20) y antes del proceso secuencial de partición. Ajuste menor al párrafo final de la subsección.

---

## Observación 8 — Aclaración sobre la única toma aérea utilizada

### Texto original del jurado

> *"Aunque se mencionan dos momentos de captura, el texto debe reflejar con precisión que este estudio se centró en la información de una única toma aérea."*

### Descripción de las modificaciones realizadas

Se reescribió la sección §3.1.2 "Imágenes hiperespectrales e índices de vegetación" para reflejar con precisión que el trabajo se basó en una única captura hiperspectral. La fecha de adquisición se confirmó mediante la metadata embebida en el archivo HYSPEX (`acquisition time: 2021-11-23T23:16:57.0Z`) y es consistente con el reporte del Anexo 41 (Figura 28 del anexo).

**Modificaciones específicas realizadas:**

- Se eliminó la mención previa a "*dos fechas clave del ciclo del cultivo (3 y 19 de noviembre de 2021)*" en el texto principal.
- Se incorporó la formulación: *"La adquisición se realizó en una única campaña de vuelo el 23 de noviembre de 2021, correspondiente a una fase fenológica del cultivo en la cual los efectos del estrés nutricional por deficiencia de fósforo son detectables a nivel de respuesta espectral. Aunque el experimento agronómico de campo contempló mediciones in situ en otras fechas (Anexo 32), el presente trabajo se basa exclusivamente en la información obtenida de esta única captura hiperspectral."*
- Se aprovechó la modificación para corregir la dimensión espacial reportada del cubo (3660 × 3438 píxeles, no 3660 × 3238 como aparecía en una redacción previa), consistente con los 12.583.080 píxeles totales por banda.
- Se añadieron datos de mayor precisión disponibles en la metadata: modelo del sensor (HYSPEX Mjolnir VS-620), resolución espacial real (2 cm/píxel), procesamiento radiométrico previo (PARGE/ATCOR) y sistema de coordenadas (WGS-84/UTM 18N, EPSG:32618).

### Ubicación en el documento

- §3.1.2 "Imágenes hiperespectrales e índices de vegetación" — sección reescrita.
- Eliminación de la oración correspondiente al cubo hiperspectral en §3.1.1 (información ahora consolidada en §3.1.2).

---

## Observación 9 — Justificación de los 12 algoritmos iniciales

### Texto original del jurado

> *"Es necesario justificar la selección de los 12 algoritmos iniciales, considerando que en el apartado 2.3.2 se mencionan más de 20 modelos. Se sugiere introducir un párrafo que identifique cuáles tuvieron un carácter exploratorio."*

### Descripción de las modificaciones realizadas

Se ampliaron los párrafos introductorios de §3.4.3.1 "Conjunto inicial de modelos evaluados" para justificar explícitamente el conjunto seleccionado y su carácter exploratorio:

**Primer bloque añadido — Tres criterios de selección:**
1. **Cobertura de familias de clasificadores** discutidas en §2.3.2 (modelos lineales, métodos basados en distancia, métodos probabilísticos, árboles de decisión, ensambles, métodos de boosting basados en gradiente), de modo que la comparación inicial proporcionara una visión amplia del espacio de soluciones.
2. **Homogeneidad metodológica:** restricción a algoritmos disponibles de manera estandarizada en la librería LazyPredict, permitiendo aplicar un protocolo experimental homogéneo (misma partición de datos, mismas métricas, mismas condiciones de hardware) sobre configuraciones por defecto, eliminando el sesgo derivado de implementaciones ad hoc.
3. **Pertinencia al problema:** clasificadores supervisados de uso establecido en problemas de teledetección hiperspectral, dada la disponibilidad de etiquetas a nivel de píxel.

**Segundo bloque añadido — Reconciliación con §2.3.2 y carácter exploratorio:**

Se discute que la diferencia numérica entre los 12 algoritmos evaluados y el conjunto más amplio descrito en §2.3.2 obedece a la distinta finalidad de cada listado. Mientras que §2.3.2 proporciona una panorámica del estado del arte en aplicación de ML/DL sobre datos hiperespectrales, la etapa exploratoria se circunscribe específicamente a clasificadores supervisados implementables bajo un protocolo estandarizado. Se justifica explícitamente:

- Las **técnicas de reducción de dimensionalidad** (PCA, ICA, LDA, Kernel PCA, t-SNE) no constituyen clasificadores sino métodos de transformación; su consideración como alternativas para la preparación de los datos se discute en §3.3.
- Los **métodos no supervisados** (k-means, SOM, ISODATA) quedan fuera del alcance del problema, dado que se cuenta con etiquetas a nivel de píxel.
- Los **métodos semi-supervisados** (TSVM, LapSVM) y **especializados** (ELM, MLC) no disponen de implementaciones estandarizadas en LazyPredict; su comparación habría requerido implementaciones ad hoc que introducirían heterogeneidad en el protocolo experimental.
- Las **arquitecturas de Deep Learning** fueron evaluadas posteriormente bajo un protocolo dedicado (§3.4.5), dado que su configuración exige consideraciones específicas que las hacen inadecuadas para una comparación rápida bajo configuraciones por defecto.

Se explicita finalmente que la etapa cumplió un **carácter exploratorio**: su objetivo no fue identificar el modelo óptimo final, sino caracterizar el desempeño comparativo de las principales familias de clasificadores supervisados sobre el problema, identificando qué tipos de modelos exhibían potencial suficiente para justificar la inversión de esfuerzo en una segunda etapa de optimización rigurosa de hiperparámetros (§3.4.4).

### Ubicación en el documento

- §3.4.3.1 "Conjunto inicial de modelos evaluados" — dos bloques de párrafos añadidos al inicio y al final de la subsección.

---

## Observación 11 — Hiperparametrización sin cambios significativos

### Texto original del jurado

> *"Además, debe ampliarse la discusión sobre por qué la hiperparametrización no aportó cambios significativos en tres de los modelos finales de ML."*

### Descripción de las modificaciones realizadas

Tras revisar las métricas reales registradas en MLflow y representadas en las Figuras 4-4 y 4-6 del documento, se constató que la observación del jurado es precisa: la búsqueda bayesiana de hiperparámetros mediante Optuna **no produjo cambios significativos en la métrica primaria PR-AUC en cinco de los seis modelos finales evaluados**, no solo en tres como originalmente señalado. En consecuencia, se realizaron las siguientes modificaciones para que el documento refleje fielmente este hallazgo y lo contextualice metodológicamente:

**Modificación 1 — Reescritura del párrafo de cierre de §4.2 "Comparación entre modelos".** El texto original afirmaba que "*la optimización de hiperparámetros mediante Optuna tuvo un impacto positivo en todos los algoritmos*", afirmación que no se sostiene cuantitativamente. Se reemplazó por una discusión extendida en tres párrafos que documenta de manera honesta:

- La constatación cuantitativa: las variaciones observadas en PR-AUC se mantuvieron por debajo de 0,02 puntos para Regresión Logística, SGDClassifier, XGBoost, CNN-1D y CNN-2D, magnitudes comparables al rango de variación esperable entre corridas.
- El caso particular de **LightGBM**, único modelo en el cual la optimización modificó de manera apreciable el comportamiento, aunque dicho cambio se manifestó principalmente como una recalibración del punto de operación: incremento sustancial en recall (de aproximadamente 0,34 a 0,91) y cambio correspondiente en el umbral óptimo de decisión, más que como una mejora del PR-AUC (que pasó de aproximadamente 0,78 a 0,82).
- La explicación estructural por familia de modelos:
  - Modelos lineales (LR, SGD): espacio de hiperparámetros con efecto material reducido; desempeño máximo acotado por la separabilidad lineal intrínseca del problema.
  - XGBoost: valores predeterminados conservadores y bien calibrados para problemas de clasificación tabular; la regularización explícita y el early stopping ya controlan el sobreajuste sin requerir ajuste fino adicional.
  - Arquitecturas de DL: desempeño base cercano al techo estructural alcanzable con la información disponible.
  - LightGBM: mayor sensibilidad de su punto de operación a la combinación de número de hojas, tasa de aprendizaje y mecanismos de regularización.

**Modificación 2 — Ajuste del párrafo de resultados CNN-2D.** Se reemplazó la afirmación de que "*la comparación entre el modelo base y el modelo final optimizado muestra mejoras sistemáticas en prácticamente todas las métricas*" por una formulación que indica fielmente que el desempeño en términos de PR-AUC se mantiene prácticamente constante (≈0,96 en ambos casos), señalando que la búsqueda bayesiana cumplió un rol de **confirmación de la robustez** de la arquitectura propuesta más que de mejora del desempeño predictivo.

**Modificación 3 — Ajuste de la Síntesis de Deep Learning.** Se reemplazó el bullet point que afirmaba que "*los modelos finales optimizados superaron consistentemente a sus versiones baseline*" por una formulación que reporta los valores reales (CNN-1D: 0,824 vs 0,834; CNN-2D: ≈0,96 en ambos casos) y precisa que las configuraciones iniciales adoptadas a partir de prácticas comunes de la literatura ya se aproximaban al desempeño máximo alcanzable.

**Modificación 4 — Documentación del caso de LightGBM en la Síntesis ML.** Se incorporó un nuevo bullet point que destaca el caso de LightGBM como hallazgo notable: aunque el PR-AUC se modificó solo en aproximadamente 0,04 puntos, el recall pasó de aproximadamente 0,34 a 0,91, evidenciando que la búsqueda bayesiana actuó principalmente sobre el punto de operación del modelo —vía el ajuste del umbral óptimo y de los parámetros que regulan el balance entre clases— más que sobre su capacidad discriminativa global.

**Modificación 5 — Coherencia con el cierre de §4.2.** Se ajustó la frase final de la sección para que la conclusión global sea coherente con la nueva discusión: las configuraciones finales adoptadas representan un punto de operación robusto y reproducible, independientemente de la magnitud de las ganancias por optimización, manteniendo la validez de la metodología por su capacidad de garantizar reproducibilidad y trazabilidad, no por una supuesta mejora uniforme.

### Cierre argumentativo

La heterogeneidad de las ganancias por optimización bayesiana entre los seis algoritmos no debe interpretarse como una limitación del proceso experimental, sino como evidencia de que la sensibilidad del desempeño a los hiperparámetros depende fuertemente de la familia algorítmica y del régimen de datos. Para los modelos cuyo desempeño base ya se encontraba cerca del techo estructural de su familia, ganancias adicionales en capacidad discriminativa requerirían modificaciones más profundas —cambios en la representación de las características o uso de arquitecturas con mayor capacidad expresiva—, motivación adicional para la incorporación de modelos de Deep Learning en el flujo experimental.

### Ubicación en el documento

- §4.2 "Comparación entre modelos" — reescritura del párrafo final sobre HPO (tres párrafos nuevos).
- §4.1.2 "Resultados de los modelos CNN-2D" — ajuste del párrafo sobre comparación baseline vs final.
- §4.1.x "Síntesis de los resultados de Deep Learning" — reformulación del bullet sobre HPO.
- §4.1.x "Síntesis de los resultados de Machine Learning" — bullet adicional sobre LightGBM.

---

## Observación 12 — Importancia relativa de los falsos positivos frente a los falsos negativos

### Texto original del jurado

> *"En la discusión, se debe considerar la importancia de los falsos positivos frente a los falsos negativos."*

### Descripción de las modificaciones realizadas

Se ampliaron dos párrafos nuevos en §4.6 "Discusión de resultados" para abordar la importancia relativa de los errores de clasificación en el contexto operativo del trabajo. La extensión se integra al hilo argumentativo continuo de la discusión —preservando el formato narrativo de la sección— inmediatamente después del párrafo existente sobre PR-AUC y manejo del desbalance de clases (que ya introducía la idea brevemente vía la cita de Saini et al., 2025) y antes del párrafo sobre ausencia de sobreajuste.

**Primer párrafo añadido — Encuadre operativo y asimetría de costos.**

Se contextualiza la discusión recordando el alcance operativo planteado en el Capítulo 1: el sistema desarrollado se concibe como una herramienta de apoyo a la toma de decisiones para la **detección temprana** del estrés por deficiencia de fósforo, con miras a posibilitar una intervención oportuna a través de una gestión nutricional ajustada. Bajo este encuadre, se argumenta que los falsos positivos y falsos negativos no implican costos equivalentes:

- Un **falso negativo** —planta efectivamente estresada que el modelo clasifica como sana— se traduce en una microzona del cultivo que no recibe atención cuando la requiere, prolongando la condición de estrés y comprometiendo la eficiencia fotosintética y el rendimiento. Considerando que la fertilización fosfórica adecuada puede incrementar el rendimiento del fríjol común hasta en un 38 % frente a manejos deficientes (Y. Gao et al., 2016), el costo agronómico de un FN no detectado se materializa en pérdidas de productividad difícilmente recuperables en la misma campaña.
- Un **falso positivo**, en contraste, desencadena típicamente una intervención local sobre una planta o zona que no la requería; el costo asociado es principalmente económico (uso adicional de fertilizante) y ambiental (potencial lixiviación de nutrientes), de magnitud claramente menor a la pérdida por estrés no atendido.

Se incorpora la cita de **Breure et al. (2022)** para sustentar el marco general: los autores argumentan que las funciones de pérdida asociadas a errores de subestimación y sobreestimación de nutrientes son típicamente asimétricas en agricultura de precisión, y deben tratarse como tales en el diseño de sistemas de apoyo a la decisión.

**Segundo párrafo añadido — Perfiles operativos diferenciados de los modelos evaluados.**

A la luz de esta asimetría, se caracterizan los seis modelos evaluados por su perfil operativo, conectando los resultados con la implicación práctica de su despliegue:

- **CNN-2D optimizado:** balance entre precisión (0,895) y recall (0,886); adecuado tanto para tareas de caracterización del campo como para apoyo a decisiones de intervención.
- **LightGBM optimizado:** recall ≈ 0,91 con precisión menor (≈ 0,75); perfil consistente con un sistema orientado a maximizar la sensibilidad de detección, prefiriendo asumir el costo de algunos falsos positivos para reducir el riesgo de pérdidas por estrés no detectado.
- **Modelos lineales (LR, SGD):** recall en el rango 0,68–0,70; adecuados como referencias metodológicas más que como herramientas operativas en un escenario donde la sensibilidad sea el criterio dominante.

Se concluye argumentando que la elección del modelo para un despliegue operativo no depende exclusivamente del PR-AUC global sino de la prioridad agronómica del usuario final, y que el ajuste fino del umbral de decisión aporta un grado adicional de control para modular el balance FP/FN según el contexto de aplicación.

### Verificación de coherencia con el Capítulo 1

Se verificó que el caso operativo planteado (detección temprana para intervenir) es coherente con las menciones del Capítulo 1 al respecto, sin contradicción interna: el alcance operativo aparece consistentemente declarado en cinco pasajes de la introducción y la justificación del problema, todos ellos enfatizando "detección temprana", "intervención temprana", "gestión nutricional" y "optimización del uso de fertilizantes".

**Citas bibliográficas incorporadas:** Y. Gao et al. (2016) — referencia ya presente en el Capítulo 1; Breure et al. (2022) — cita nueva añadida a la bibliografía.

### Ubicación en el documento

- §4.6 "Discusión de resultados" — dos párrafos nuevos integrados al hilo argumentativo continuo, después del párrafo sobre PR-AUC y desbalance de clases.

---

# Bloque III — Observaciones de análisis adicional

---

## Observación 18 — Métricas de costo computacional para los modelos de Deep Learning

### Texto original del jurado

> *"Si bien se menciona el costo computacional como causa para no profundizar en él, este criterio debe discutirse más a fondo dada la naturaleza del caso de estudio y las capacidades de cómputo actuales. (...) debe (...) incluir las métricas de costo computacional para los modelos de Deep Learning, ya que como fue expuesto en el texto anteriormente es un criterio relevante para la comparación de resultados."*

### Descripción de las modificaciones realizadas

Se incorporó al Capítulo 4 una nueva subsección **§4.2.1 "Costo computacional"** dentro de la sección comparativa §4.2 "Comparación entre modelos", presentando una caracterización cuantitativa del costo computacional asociado a los seis modelos finales evaluados.

**Decisión de alcance.** Aunque la observación del jurado solicita explícitamente las métricas para los modelos de Deep Learning, se decidió incluir en la tabla comparativa también los cuatro modelos finales de Machine Learning (Regresión Logística, SGDClassifier, LightGBM y XGBoost). Esta decisión se justifica porque la observación adicional del jurado solicita "*discutir más a fondo el costo computacional como criterio dada la naturaleza del caso de estudio*", y una tabla comparativa ML vs DL refuerza este argumento. Adicionalmente, esta decisión es coherente con que el texto original de la tesis ya invocaba el costo computacional como criterio para la selección de los algoritmos finales de ML.

**Estructura de la caracterización.** La caracterización combina dos fuentes de información:

- **Métricas extraídas de MLflow (M):** tiempos de entrenamiento del run final, tiempo total acumulado durante la búsqueda de hiperparámetros, hardware reportado, hiperparámetros finales.
- **Mediciones empíricas locales (E):** conteo de parámetros o estructura interna, tamaño de los pesos serializados en disco, memoria pico durante la inferencia, latencia y rendimiento (throughput) de inferencia.

La medición de inferencia se realizó sobre lotes de 512 muestras, con 100 lotes de calentamiento previo descartados, sobre el mismo conjunto de prueba utilizado en las secciones anteriores.

**Consideraciones metodológicas explicitadas:**

1. Las mediciones de tamaño en disco están condicionadas por las librerías de serialización empleadas (joblib para scikit-learn, formato nativo de LightGBM y XGBoost, `state_dict` de PyTorch para los modelos CNN). Las diferencias absolutas reflejan no solo la complejidad estructural sino también las convenciones de cada librería.
2. Las mediciones de inferencia se realizaron sobre el hardware nativo de cada familia: CPU para los modelos de ML y GPU NVIDIA RTX 3050 Laptop para los modelos de DL. Esta condición es consistente con el contexto típico de despliegue, no con una comparación de eficiencia algorítmica en igualdad de hardware.

**Hallazgos principales reportados:**

- En **costo de entrenamiento**, los modelos lineales y LightGBM presentan tiempos del orden de minutos. XGBoost incrementa significativamente el costo durante la fase de Optuna, alcanzando tiempos comparables a los de las arquitecturas de Deep Learning. Los modelos de Deep Learning presentan los tiempos más extensos (hasta 15h 12min para CNN-1D durante HPO).
- En **costo de inferencia**, todos los modelos presentan latencias inferiores a 15 milisegundos por lote de 512 muestras y rendimientos superiores a 35.000 muestras por segundo. La extrapolación del tiempo necesario para procesar un millón de píxeles oscila entre 1,5 segundos para CNN-1D y 27,9 segundos para XGBoost.

**Discusión contextual.** Se incorpora un párrafo de cierre que argumenta que la consideración explícita del costo computacional debe interpretarse en el contexto agronómico actual, donde la disponibilidad de cómputo accesible —incluyendo aceleración por GPU en equipos de gama media— ha modificado los compromisos tradicionales entre complejidad del modelo y viabilidad práctica. En este sentido, el costo computacional no actúa en el trabajo como criterio excluyente sino como una dimensión de comparación complementaria al desempeño predictivo.

**Figura adicional.** Se incorporó la Figura 4-X que presenta gráficamente la comparativa de eficiencia de inferencia (rendimiento y latencia) entre los seis modelos.

### Ubicación en el documento

- §4.2.1 "Costo computacional" — subsección nueva dentro de la sección comparativa.
- Tabla 4-Z y Figura 4-X dentro de la subsección.
- Notebook de soporte: `402-jmmz-computational-cost.ipynb`.
- Módulo del repositorio: `spectralcrop/performance/computational_cost.py`.

---

## Observación 19 — Diagrama del diseño experimental

### Texto original del jurado

> *"Se recomienda incluir un diagrama del diseño experimental."*

### Descripción de las modificaciones realizadas

Se incorporó a §3.1.1 "Origen y condiciones del experimento" un diagrama de la estructura factorial del diseño experimental. Adicionalmente, se reorganizó y consolidó la descripción del diseño experimental que aparecía dispersa en la sección, eliminando una redundancia entre dos párrafos que describían el arreglo de bloques completos al azar y la estructura factorial.

**Figura nueva incorporada.** La figura presenta una matriz de 8 genotipos × 4 niveles de fertilización con P₂O₅ × 3 repeticiones, representando las 96 unidades experimentales nominales del experimento. Cada celda muestra los tres puntos correspondientes a las repeticiones, con código de color por nivel de fertilización (T1=25%, T2=50%, T3=75%, T4=100%).

**Texto introductorio de la figura.** Se redactó un párrafo que articula las dos perspectivas del diseño:

- La **estructura factorial conceptual**, ilustrada en la nueva Figura 3-2.
- La **realización física sobre el lote experimental**, referenciada hacia la Figura existente en §3.2.4 (NDVI con polígonos de etiquetado), que ya cumple esa función de manera satisfactoria en el documento original.

**Decisión de no incluir un tercer mapa.** Durante la elaboración se generó adicionalmente una figura de distribución espacial basada en los centroides de los polígonos. Tras evaluar su valor informativo, se concluyó que aportaba información redundante respecto a la Figura existente de NDVI + polígonos en §3.2.4 y se descartó su inclusión en el documento. El archivo correspondiente queda disponible en el repositorio como evidencia del trabajo realizado, sin referencia desde el documento.

### Ubicación en el documento

- §3.1.1 "Origen y condiciones del experimento" — texto reorganizado y figura nueva insertada.
- Notebook de soporte: `403-jmmz-experimental-design-diagrams.ipynb`.

---

# Observaciones pendientes de atención

Las siguientes observaciones se encuentran en proceso de atención y serán abordadas en la siguiente entrega:

| N° | Observación | Categoría |
|---|---|---|
| 1 | Normalización de separadores decimales | Forma |
| 2 | Corrección ortográfica y de digitación | Forma |
| 3 | Redundancia en la definición de imágenes hiperespectrales | Forma |
| 4 | Justificación del umbral NDVI | Metodología |
| 5 | Criterio de "alto bienestar" y representatividad | Metodología |
| 10 | Profundización del análisis de Random Forest | Modelado |
| 13 | Caso práctico: parcelas con plantas mixtas | Discusión |
| 14 | Reestructuración de conclusiones | Estructura |
| 15 | Trabajo futuro: validación ante otros estreses y genotipos | Estructura |
| 16 | Matrices de confusión en porcentajes y reducción de datos | Análisis |
| 17 | Influencia real de los índices de vegetación | Análisis |

---

## Recursos de soporte para la verificación

Las modificaciones realizadas se acompañan de los siguientes recursos en el repositorio público del trabajo (https://dagshub.com/johnma96/thesis):

| Recurso | Ubicación | Soporta |
|---|---|---|
| Notebook 201 — Exploración inicial de PCA | `notebooks/201-jmmz-pca.ipynb` | Observación 6 |
| Notebook 202 — Selección de bandas espectrales | `notebooks/202-jmmz-band-selection.ipynb` | Observación 6 |
| Notebook 401 — Análisis por genotipo | `notebooks/401-jmmz-genotype-analysis.ipynb` | Observación 20 |
| Notebook 402 — Pruebas de ablación y costo computacional | `notebooks/402-jmmz-ablation-tests.ipynb`, `notebooks/402-jmmz-computational-cost.ipynb` | Observaciones 18 y 20 |
| Notebook 403 — Diagramas de diseño experimental | `notebooks/403-jmmz-experimental-design-diagrams.ipynb` | Observación 19 |
| Módulo de costo computacional | `spectralcrop/performance/computational_cost.py` | Observación 18 |
| Módulo de visualización experimental | `spectralcrop/visualization/experimental_design.py` | Observación 19 |
| Tracking experimental MLflow | https://dagshub.com/johnma96/thesis.mlflow | Todas las observaciones de modelado |

---

## Referencias bibliográficas incorporadas durante el proceso de corrección

Se relacionan a continuación las referencias bibliográficas nuevas integradas a la bibliografía del trabajo durante la atención de las observaciones del jurado:

- **Breure, T. S., Haefele, S. M., Hannam, J. A., Corstanje, R., Webster, R., Moreno-Rojas, S., & Milne, A. E.** (2022). A loss function to evaluate agricultural decision-making under uncertainty: A case study of soil spectroscopy. *Precision Agriculture, 23*(5), 1942–1971. https://doi.org/10.1007/s11119-022-09887-2 — Observación 12.
- **Kattenborn, T., Schiefer, F., Frey, J., Feilhauer, H., Mahecha, M. D., & Dormann, C. F.** (2022). Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks. *ISPRS Open Journal of Photogrammetry and Remote Sensing, 5*, 100018. https://doi.org/10.1016/j.ophoto.2022.100018 — Observación 20.
- **Okyere, F. G., Cudjoe, D., Sadeghi-Tehran, P., Virlet, N., Riche, A. B., Castle, M., Greche, L., Simms, D., Mhada, M., Mohareb, F., & Hawkesford, M. J.** (2023). Modeling the spatial-spectral characteristics of plants for nutrient status identification using hyperspectral data and deep learning methods. *Frontiers in Plant Science, 14*, 1209500. https://doi.org/10.3389/fpls.2023.1209500 — Observación 20.
- **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881 — Observación 20.

---

## Cierre

Las modificaciones documentadas en esta carta corresponden a las diez observaciones del jurado evaluador atendidas hasta la fecha. La totalidad de las observaciones restantes (diez) están en proceso de atención bajo el cronograma establecido para la entrega final, con fecha límite del 12 de mayo de 2026.

Quedo a disposición del jurado evaluador para resolver cualquier inquietud adicional sobre las modificaciones realizadas o sobre el avance del trabajo en su conjunto.

Atentamente,

**John Mario Montoya Zapata**
Estudiante de Maestría
[Programa académico]
Universidad Nacional de Colombia

---

*Documento elaborado el [completar fecha]. Versión preliminar — sujeta a actualización conforme se completen las observaciones pendientes.*