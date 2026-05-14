# Propuesta de despliegue en Google Cloud Platform (GCP)

**Proyecto:** spectralcrop — diagnóstico no invasivo de estrés por P en frijol  
**Modelo:** CNN-2D (`bean_stress_classifier` v1, PR-AUC = 0.9635)  
**Fecha de propuesta:** Mayo 2026

---

## Contexto y objetivos

El sistema debe soportar dos flujos de trabajo distintos con requisitos muy diferentes:

| Flujo | Frecuencia | Latencia aceptable | Carga de datos |
|---|---|---|---|
| **Predicción por imagen completa** (nueva captura UAV → mapa de estrés) | Esporádico (tras vuelo UAV, ~ días/semanas) | Minutos–horas | 9.5 GB por imagen |
| **Predicción por parche** (consulta puntual, API REST) | Frecuente | < 500 ms | 5×5×63 floats ≈ 63 KB |

Una arquitectura que sirva ambos casos usa dos rutas separadas.

---

## Arquitectura propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIOS                                  │
│   Investigador/Agrónomo          Aplicación / Script            │
└─────────┬──────────────────────────────────┬────────────────────┘
          │ Subir imagen ENVI                │ POST /predict (parche)
          ▼                                  ▼
┌─────────────────┐              ┌────────────────────────┐
│   Cloud Storage  │              │    Cloud Run (API)      │
│   (GCS)          │              │    FastAPI + CNN-2D     │
│   Input bucket   │              │    Siempre activo       │
└─────────┬────────┘              └────────────────────────┘
          │ Evento: archivo nuevo
          ▼
┌─────────────────────────┐
│   Cloud Run Jobs         │
│   (Batch processing)     │
│   preprocess → predict   │
│   → guardar resultados   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────┐    ┌──────────────────────┐
│   Cloud Storage  │    │  Artifact Registry   │
│   Output bucket  │    │  (imagen Docker)      │
│   GeoTIFFs       │    └──────────────────────┘
└─────────────────┘
          │
          ▼
┌─────────────────────────┐
│   Looker Studio /        │
│   GEE / webapp           │
│   (visualización)        │
└─────────────────────────┘
```

---

## Componentes

### 1. Cloud Storage (GCS)

Dos buckets:

```
gs://spectralcrop-inputs/
  raw/YYYY-MM-DD/        ← imagen ENVI subida por el usuario
  models/                ← pesos del modelo (cnn2d_final_model_weights.pt, scaler)

gs://spectralcrop-outputs/
  predictions/YYYY-MM-DD/
    prediction_proba.tif
    prediction_class.tif
    metadata.json        ← fecha, modelo usado, PR-AUC reportado
```

**Por qué GCS:** los archivos ENVI de 9.5 GB no pueden pasar por una API HTTP.
La arquitectura event-driven (evento al subir el archivo) desacopla la ingesta del procesamiento.

### 2. Cloud Run Jobs — procesamiento batch

Tarea asíncrona que se dispara automáticamente cuando aparece un nuevo archivo en el bucket de inputs:

```
Eventarc (GCS trigger)
  → Cloud Run Job: spectralcrop-batch
     1. Descargar imagen ENVI desde GCS
     2. uv run python main.py preprocess --envi <path>
     3. uv run python main.py predict --zarr <zarr>
     4. Subir GeoTIFFs a gs://spectralcrop-outputs/
     5. Publicar notificación (Pub/Sub → email / webhook)
```

**Especificación del Job:**

```yaml
# job.yaml (Cloud Run Job)
apiVersion: run.googleapis.com/v1
kind: Job
spec:
  template:
    spec:
      containers:
        - image: REGION-docker.pkg.dev/PROJECT/spectralcrop/batch:latest
          resources:
            limits:
              cpu: "8"
              memory: "32Gi"    # necesario para cargar el cubo + parches en memoria
          env:
            - name: GCS_INPUT_BUCKET
              value: spectralcrop-inputs
            - name: GCS_OUTPUT_BUCKET
              value: spectralcrop-outputs
      timeoutSeconds: 7200     # 2 horas máximo por imagen
```

**GPU opcional:** si se agrega una GPU (NVIDIA T4 en Cloud Run Jobs), el tiempo de inferencia baja de ~2 min a ~20 s. El costo de la GPU es ~$0.35/h, justificado si se procesan varias imágenes por semana.

### 3. Cloud Run Service — API REST para parches

Para el caso de uso de integración con aplicaciones web o sistemas de decisión agrícola que consultan píxeles individuales:

```
POST https://spectralcrop-api-HASH-uc.a.run.app/predict
Body: {"patch": [[[...]]]}   # (63, 5, 5) float32
Response: {"label": 1, "probability_stressed": 0.87, ...}
```

El modelo se carga una vez al arrancar el contenedor (lifespan handler en `app/main.py` ya implementado). Cloud Run escala a cero cuando no hay tráfico.

**Especificación del Service:**

```yaml
# service.yaml
resources:
  limits:
    cpu: "2"
    memory: "4Gi"   # CNN-2D pesa ~6 MB, sin GPU
min-instances: 0    # escala a cero (ahorro de costos)
max-instances: 10
```

### 4. Artifact Registry

Almacena las imágenes Docker del batch y de la API. El `Dockerfile.example` que ya existe en `app/` es el punto de partida para la imagen de la API.

Para el batch:

```dockerfile
# Dockerfile.batch
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY spectralcrop/ spectralcrop/
COPY main.py .
# CPU-only torch inside the container
RUN uv sync --no-dev --extra pytorch-cpu --frozen
CMD ["uv", "run", "python", "scripts/batch_predict.py"]
```

### 5. Secret Manager

Las credenciales sensibles (DagsHub token, MLflow URI, GCS service account) se almacenan en Secret Manager y se inyectan como variables de entorno en tiempo de ejecución — nunca en el código ni en la imagen Docker.

---

## Infraestructura como código (Terraform)

```hcl
# main.tf
resource "google_storage_bucket" "inputs" {
  name     = "spectralcrop-inputs"
  location = "US-CENTRAL1"
  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }   # borrar inputs después de 90 días
  }
}

resource "google_cloud_run_v2_job" "batch" {
  name     = "spectralcrop-batch"
  location = "us-central1"
  template {
    template {
      containers {
        image = "us-central1-docker.pkg.dev/${var.project}/spectralcrop/batch:latest"
        resources { limits = { cpu = "8", memory = "32Gi" } }
      }
      max_retries     = 1
      timeout         = "7200s"
    }
  }
}
```

---

## Estimación de costos (referencia)

| Componente | Uso estimado | Costo/mes aprox. |
|---|---|---|
| GCS (input + output + models) | ~50 GB almacenamiento | $1–2 |
| Cloud Run Job (batch) | 4 imágenes/mes × 30 min × 32 GB RAM | ~$15–25 |
| Cloud Run Service (API) | 1000 req/mes, escala a 0 | < $1 |
| Artifact Registry | 2 imágenes Docker ~3 GB | < $1 |
| **Total estimado** | | **~$18–30 / mes** |

Con GPU (T4): agregar ~$12/imagen procesada → evaluar si el volumen lo justifica.

---

## Plan de implementación (4 sprints)

| Sprint | Duración | Tareas |
|---|---|---|
| **1 — Contenedores** | 1 semana | Dockerfile.batch funcional, CI que pushea imagen a Artifact Registry, prueba local con `docker run` |
| **2 — GCS + batch job** | 1 semana | Bucket, Cloud Run Job, script `batch_predict.py`, prueba end-to-end con imagen real |
| **3 — API + Eventarc** | 1 semana | Cloud Run Service con `app/`, trigger automático al subir archivo a GCS |
| **4 — IaC + seguridad** | 1 semana | Terraform, Secret Manager, IAM mínimo, dominio personalizado |

---

## Prerrequisitos

1. Cuenta GCP con proyecto creado (`spectralcrop` o similar)
2. `gcloud` CLI configurado
3. APIs habilitadas: Cloud Run, Cloud Storage, Artifact Registry, Eventarc, Secret Manager
4. Service Account con roles: `Storage Admin`, `Cloud Run Developer`, `Artifact Registry Writer`
5. Completar `app/services/model_loader.py` para cargar modelo desde GCS/MLflow en lugar de ruta local

---

## Próximo paso concreto

Completar el `app/services/model_loader.py` para cargar el modelo desde GCS:

```python
# Reemplazar la carga desde archivo local por:
import google.cloud.storage as gcs

def load_from_gcs(bucket: str, blob_path: str, local_path: Path) -> None:
    client = gcs.Client()
    bucket_obj = client.bucket(bucket)
    bucket_obj.blob(blob_path).download_to_filename(str(local_path))

# En ModelRegistry.load():
load_from_gcs("spectralcrop-inputs", "models/cnn2d_final_model_weights.pt", weights_path)
load_from_gcs("spectralcrop-inputs", "models/robust_scaler.pkl", scaler_path)
```
