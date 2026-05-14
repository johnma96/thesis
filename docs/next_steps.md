# Next steps — spectralcrop roadmap

Last updated: 2026-05-14 | Current version: v1.3.0 | Active branch: `develop`

---

## BLOQUEADO — Hacer primero en GCP (pasos manuales, ~1 h)

Seguir **`docs/gcp_setup.md`** en orden:

```
1. gcloud projects create spectralcrop-prod
2. Habilitar APIs (Artifact Registry, Cloud Run, Storage, IAM)
3. Crear repositorio AR: us-central1-docker.pkg.dev/spectralcrop-prod/spectralcrop
4. Crear service account: github-actions@spectralcrop-prod.iam.gserviceaccount.com
5. Configurar Workload Identity Federation (WIF) para GitHub Actions
6. Agregar 3 secrets en GitHub:
     GCP_PROJECT_ID              → spectralcrop-prod
     GCP_WORKLOAD_IDENTITY_PROVIDER → (output del paso 5)
     GCP_SERVICE_ACCOUNT         → github-actions@spectralcrop-prod.iam.gserviceaccount.com
7. Crear buckets GCS y subir model weights:
     gs://spectralcrop-inputs/models/cnn2d_final_model_weights.pt
     gs://spectralcrop-inputs/models/cnn2d_final_model_info.json
     gs://spectralcrop-inputs/models/robust_scaler.pkl
```

Una vez completado: hacer PR `develop` → `main` en GitHub.
El CI construirá y publicará `Dockerfile.batch` automáticamente.

---

## Sprint 2 — Cloud Run Job + trigger automático (1 semana)

Prerequisito: Sprint 1 (GCP setup) completado.

- [ ] Crear el Cloud Run Job en GCP apuntando a la imagen de Artifact Registry
- [ ] Configurar Eventarc: trigger automático cuando aparece un `.hdr` en `gs://spectralcrop-inputs/raw/`
- [ ] Probar end-to-end: subir imagen real → recibir TIFs en `gs://spectralcrop-outputs/`
- [ ] Configurar notificación al finalizar (Pub/Sub → email o webhook)

---

## Sprint 3 — Cloud Run Service (API REST) (1 semana)

- [ ] Crear `Dockerfile.api` basado en `app/Dockerfile.example`
- [ ] Completar `app/services/model_loader.py` para cargar el modelo desde GCS en lugar de ruta local
- [ ] Desplegar `app/` como Cloud Run Service (escala a cero cuando no hay tráfico)
- [ ] Probar endpoint: `POST /predict` con un parche 5×5×63 real
- [ ] Agregar autenticación básica (API key via Cloud Endpoints o header)

---

## Sprint 4 — Infraestructura como código + seguridad (1 semana)

- [ ] Escribir `terraform/main.tf` para reproducir toda la infraestructura GCP desde cero
- [ ] Mover secretos a Secret Manager (tokens DagsHub, credenciales GCP)
- [ ] Configurar dominio personalizado para la API (opcional)
- [ ] IAM mínimo: revisar que ningún componente tenga permisos más amplios de lo necesario
- [ ] Documentar arquitectura final en `docs/gcp_deployment_proposal.md` (actualizar con lo real)

---

## Mejoras de código pendientes (sin prerequisito de GCP)

Estas se pueden hacer en cualquier momento en `develop`:

- [ ] `main.py evaluate` — ya escribe métricas JSON, pero falta test de integración que corra `dvc repro` y verifique PR-AUC = 0.9635
- [ ] `spectralcrop/data/labeling.py` y `split.py` — cobertura de tests = 0%. Crear GeoPackage sintético con proyección UTM para tests unitarios
- [ ] `app/` — completar `model_loader.py` para cargar desde GCS (prereq Sprint 3)
- [ ] `spectralcrop/performance/` y `spectralcrop/visualization/` — módulos vacíos; rellenar o eliminar
- [ ] Pre-commit hooks: `uv run pre-commit install` con ruff (auto-format antes de cada commit)
- [ ] Multiclass classification: los datos originales tienen 4 niveles de P; el modelo binario fue una simplificación; explorar modelo multiclase como trabajo futuro

---

## Versiones y tags

| Tag | Estado |
|---|---|
| `v1.3.0` | ✅ Actual — vectorised inference, DVC pipeline, tests 40% |
| `v1.4.0` | Próximo — GCP Sprint 1 mergeado a main |
| `v1.5.0` | Cloud Run Job funcionando end-to-end |
| `v2.0.0` | API REST + batch job en producción |
