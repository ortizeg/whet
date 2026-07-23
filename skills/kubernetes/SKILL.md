---
name: kubernetes
description: >
  Use this skill when deploying ML inference or training services to Kubernetes —
  Deployments, GPU scheduling, Services and Ingress, HPA autoscaling, ConfigMaps and
  Secrets, persistent volumes for model storage, health probes, Helm charts, and
  Kustomize overlays for dev/staging/prod. Reach for it any time you'd otherwise hand-write
  k8s manifests or a Helm chart for a model server, even if the user just says "deploy the
  model to the cluster" or "scale up the endpoint". For building the container image the
  cluster runs, see docker-cv.
---

# Kubernetes Skill

Deployment patterns for ML inference and training services: GPU scheduling, Helm charts,
autoscaling, and environment-specific overlays. This page is the index — the essential
manifest shape is inline, everything else lives in `references/` and should be read only
when the task calls for it.

## Is Kubernetes the Right Target?

Reach for K8s only when the load and operational maturity justify it.

```
Where should this run?
├── Internal/team use → single instance + Docker Compose
├── Production API (< 100 RPS) → Cloud Run / App Runner / ECS Fargate
├── Production API (> 100 RPS) → Kubernetes with autoscaling  ← this skill
├── Batch inference → Vertex AI Batch / SageMaker Batch Transform
└── Training at scale → Vertex AI / SageMaker training jobs (see gcp)
```

## The Core Inference Deployment

Nearly every ML deployment starts here: a Deployment with explicit GPU requests, all three
probe types, and a ClusterIP Service in front of it.

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  selector:
    matchLabels: { app: model-server }
  template:
    metadata:
      labels: { app: model-server }
    spec:
      containers:
        - name: model-server
          image: registry.example.com/ml-images/inference:v1.2.0  # never :latest
          ports:
            - { containerPort: 8000, name: http }
          resources:  # GPU requests and limits must be equal
            requests: { cpu: "2", memory: "4Gi", nvidia.com/gpu: "1" }
            limits: { cpu: "4", memory: "8Gi", nvidia.com/gpu: "1" }
          envFrom:
            - configMapRef: { name: model-config }
            - secretRef: { name: model-secrets }
          startupProbe:  # ~5 min budget for model load — do not omit
            httpGet: { path: /health, port: http }
            periodSeconds: 10
            failureThreshold: 30
          livenessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 15
          readinessProbe:  # 503 until weights are loaded
            httpGet: { path: /health/ready, port: http }
            periodSeconds: 10
      tolerations:
        - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  type: ClusterIP
  ports:
    - { port: 80, targetPort: http, protocol: TCP, name: http }
  selector: { app: model-server }
```

Apply with `kubectl apply -k k8s/overlays/dev/` (Kustomize) or `helm install` (Helm) —
never `kubectl apply -f` against hand-edited per-environment copies.

## Conventions

- Set GPU, CPU, and memory **requests and limits** on every pod; GPU requests and limits must match exactly.
- Use **all three probes** on ML containers — startup for load time, liveness for hangs, readiness to gate traffic.
- **Pin image tags** to a version or Git SHA; `latest` breaks rollbacks.
- Keep model weights **out of images** — PVC or init container.
- One **namespace per environment** (`ml-dev`, `ml-staging`, `ml-prod`, `ml-training`), each with a ResourceQuota.
- Express environment differences with **Kustomize overlays or Helm values**, never duplicated YAML.
- Run **as non-root** (`runAsNonRoot: true`) and terminate TLS at the Ingress.
- Set a **PodDisruptionBudget** (`minAvailable`) so node upgrades cannot drain every replica.
- Scale the HPA on **request rate or GPU utilization**, not CPU alone.

## Anti-Patterns to Avoid

- Do not omit GPU resource limits -- without explicit `nvidia.com/gpu` limits, pods will not be scheduled on GPU nodes and will silently fall back to CPU.
- Do not use `latest` image tags in production -- always pin to a specific version tag or Git SHA for reproducibility and safe rollbacks.
- Do not skip startup probes for ML containers -- model loading can take minutes; without a startup probe, Kubernetes will kill pods before they are ready.
- Do not store model weights inside container images -- images become multi-gigabyte and slow to pull; use PVCs or init containers to download models.
- Do not set CPU requests too low for GPU pods -- GPU inference still requires CPU for preprocessing; under-provisioned CPU starves the pipeline.
- Do not hardcode environment-specific values in base manifests -- use Kustomize overlays or Helm values files for dev/staging/prod differences.
- Do not run pods as root -- always set `runAsNonRoot: true` in the pod security context.
- Do not skip resource quotas on shared clusters -- without quotas, a single team can monopolize all GPU resources.
- Do not expose inference services without TLS -- always terminate TLS at the Ingress or use a service mesh.
- Do not ignore pod disruption budgets -- set `minAvailable` to prevent all replicas from being evicted during node upgrades.

## Deep dives

- `references/manifests-and-services.md` — read when writing the full Deployment, Service, or Ingress, or wiring up liveness/readiness/startup probes and their FastAPI endpoints.
- `references/gpu-scheduling.md` — read when requesting GPUs, installing the NVIDIA device plugin, or targeting specific accelerator nodes with tolerations and nodeSelectors.
- `references/autoscaling.md` — read when configuring an HPA, tuning scale-up/scale-down behavior, or scaling on custom metrics like request rate.
- `references/config-and-secrets.md` — read when injecting model configuration or credentials, or organizing namespaces and per-namespace GPU quotas.
- `references/storage-and-volumes.md` — read when deciding how model weights reach the pod: PersistentVolumeClaims versus an init container download.
- `references/helm-and-kustomize.md` — read when packaging the service as a Helm chart or building dev/staging/prod Kustomize overlays.
- `references/training-jobs.md` — read when running training on the cluster with a `batch/v1` Job rather than serving inference.
