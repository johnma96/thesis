# GCP setup guide — one-time configuration

Before the `docker-publish.yml` CI workflow can push images to Artifact Registry,
complete these steps **once** in your GCP project.

---

## 1. Create the GCP project

```bash
gcloud projects create spectralcrop-prod --name="spectralcrop"
gcloud config set project spectralcrop-prod
gcloud billing projects link spectralcrop-prod --billing-account=<BILLING_ACCOUNT_ID>
```

## 2. Enable required APIs

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com
```

## 3. Create Artifact Registry repository

```bash
gcloud artifacts repositories create spectralcrop \
  --repository-format=docker \
  --location=us-central1 \
  --description="spectralcrop Docker images"
```

## 4. Create a service account for CI

```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions CI"

# Grant permission to push to Artifact Registry
gcloud projects add-iam-policy-binding spectralcrop-prod \
  --member="serviceAccount:github-actions@spectralcrop-prod.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

## 5. Configure Workload Identity Federation (keyless auth — no JSON keys)

```bash
# Create the pool
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Actions pool"

# Create the provider
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --workload-identity-pool=github-pool \
  --location=global \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='johnma96/thesis'"

# Allow the GitHub repo to impersonate the service account
POOL_ID=$(gcloud iam workload-identity-pools describe github-pool \
  --location=global --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding \
  github-actions@spectralcrop-prod.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/johnma96/thesis"
```

## 6. Add GitHub repository secrets

Go to **GitHub → johnma96/thesis → Settings → Secrets → Actions** and add:

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | `spectralcrop-prod` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Output of: `gcloud iam workload-identity-pools providers describe github-provider --workload-identity-pool=github-pool --location=global --format="value(name)"` |
| `GCP_SERVICE_ACCOUNT` | `github-actions@spectralcrop-prod.iam.gserviceaccount.com` |

## 7. Create GCS buckets

```bash
gcloud storage buckets create gs://spectralcrop-inputs  --location=US-CENTRAL1
gcloud storage buckets create gs://spectralcrop-outputs --location=US-CENTRAL1

# Upload model weights and scaler (from local models/ directory)
gcloud storage cp models/cnn2d_final_model_weights.pt  gs://spectralcrop-inputs/models/
gcloud storage cp models/cnn2d_final_model_info.json   gs://spectralcrop-inputs/models/
gcloud storage cp models/robust_scaler.pkl             gs://spectralcrop-inputs/models/
```

## 8. Verify: trigger the CI workflow

Push any change to `main` that touches `Dockerfile.batch`, `spectralcrop/`,
`scripts/batch_predict.py`, or `main.py`. The `docker-publish.yml` workflow
will build and push the image automatically.

Check progress at: **GitHub → johnma96/thesis → Actions → Build and push Docker image**

---

## Local test (without GCP credentials)

```bash
# Build the image locally
docker build -f Dockerfile.batch -t spectralcrop-batch:local .

# Smoke test (no GCS connection needed)
docker run --rm spectralcrop-batch:local python -c "import spectralcrop; import torch; print('OK')"
```
