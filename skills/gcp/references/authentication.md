# GCP Authentication and IAM

Scope: service accounts for training jobs, Workload Identity Federation for CI, and Docker registry authentication.

## Service Account Setup

```bash
# Create a service account for training jobs
gcloud iam service-accounts create ml-trainer \
    --display-name="ML Training Service Account"

# Grant required roles (least privilege)
SA_EMAIL="ml-trainer@my-project.iam.gserviceaccount.com"
for ROLE in aiplatform.user storage.objectAdmin artifactregistry.reader; do
    gcloud projects add-iam-policy-binding my-project \
        --member="serviceAccount:${SA_EMAIL}" --role="roles/${ROLE}"
done
```

## Workload Identity Federation for CI

```bash
# Create a workload identity pool for GitHub Actions
gcloud iam workload-identity-pools create "github-pool" \
    --location="global" \
    --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
    --location="global" \
    --workload-identity-pool="github-pool" \
    --display-name="GitHub Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --issuer-uri="https://token.actions.githubusercontent.com"

# Allow the GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/OWNER/REPO"
```

## Docker Authentication

```bash
# Local: gcloud auth configure-docker us-central1-docker.pkg.dev
# CI (prefer WIF over keys): cat key.json | docker login -u _json_key \
#   --password-stdin https://us-central1-docker.pkg.dev
```
