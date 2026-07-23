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

Deployment patterns for ML inference and training services: GPU scheduling, Helm charts, autoscaling, and environment-specific overlays.

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

## Deployment Manifests for ML Services

Define Deployments with explicit resource requests and GPU scheduling for inference workloads.

### Inference Deployment

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  labels:
    app: model-server
    component: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
        component: inference
    spec:
      containers:
        - name: model-server
          image: registry.example.com/ml-images/inference:v1.2.0
          ports:
            - containerPort: 8000
              name: http
          resources:
            requests: { cpu: "2", memory: "4Gi", nvidia.com/gpu: "1" }
            limits: { cpu: "4", memory: "8Gi", nvidia.com/gpu: "1" }
          envFrom:
            - configMapRef: { name: model-config }
            - secretRef: { name: model-secrets }
          volumeMounts:
            - { name: model-storage, mountPath: /models, readOnly: true }
          livenessProbe:
            httpGet: { path: /health, port: http }
            initialDelaySeconds: 30
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet: { path: /health/ready, port: http }
            initialDelaySeconds: 60
            periodSeconds: 10
            failureThreshold: 5
          startupProbe:  # allows ~5 min for model load
            httpGet: { path: /health, port: http }
            initialDelaySeconds: 10
            periodSeconds: 10
            failureThreshold: 30
      volumes:
        - name: model-storage
          persistentVolumeClaim: { claimName: model-pvc }
      tolerations:
        - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

## GPU Resource Requests

Always set both `requests` and `limits` for `nvidia.com/gpu`, and they must be equal — GPU scheduling requires exact counts (fractional GPUs are not natively supported). Multi-GPU pods just raise the count alongside CPU/memory:

```yaml
resources:
  requests: { cpu: "8", memory: "32Gi", nvidia.com/gpu: "4" }
  limits: { cpu: "16", memory: "64Gi", nvidia.com/gpu: "4" }
```

### NVIDIA Device Plugin

GPU scheduling requires the NVIDIA device plugin deployed in-cluster:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
    --namespace kube-system --set runtimeClassName=nvidia
```

## Service and Ingress Configuration

A `ClusterIP` Service exposes the pods internally:

```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  type: ClusterIP
  ports:
    - { port: 80, targetPort: http, protocol: TCP, name: http }
  selector:
    app: model-server
```

### Ingress

TLS-terminated Ingress (nginx + cert-manager); raise `proxy-body-size` for large image uploads:

```yaml
# k8s/base/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-server
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts: [api.ml.example.com]
      secretName: model-server-tls
  rules:
    - host: api.ml.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: model-server
                port: { name: http }
```

## Horizontal Pod Autoscaler

Scale inference pods based on CPU, memory, or custom metrics such as request latency or GPU utilization.

```yaml
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 10
  behavior:  # scale up fast (60s window), down slow (300s window) to avoid flapping
    scaleUp:
      stabilizationWindowSeconds: 60
      policies: [{ type: Pods, value: 2, periodSeconds: 60 }]
    scaleDown:
      stabilizationWindowSeconds: 300
      policies: [{ type: Pods, value: 1, periodSeconds: 120 }]
  metrics:  # add a matching memory Resource metric (averageUtilization: 80) as needed
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods  # custom metric — scale on request rate, not just CPU
      pods:
        metric:
          name: inference_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

## ConfigMaps and Secrets for Model Config

### ConfigMap

```yaml
# k8s/base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_NAME: "resnet50"
  MODEL_PATH: "/models/resnet50_v1.2.0.onnx"
  BATCH_SIZE: "8"
  NUM_WORKERS: "4"
  LOG_LEVEL: "INFO"
  CONFIDENCE_THRESHOLD: "0.5"
```

### Secret

Never commit plaintext secrets to a `Secret` manifest — create them at deploy time (or use sealed-secrets):

```bash
kubectl create secret generic model-secrets \
    --from-literal=WANDB_API_KEY="${WANDB_API_KEY}" \
    --from-literal=S3_ACCESS_KEY="${S3_ACCESS_KEY}" \
    --namespace=ml-inference
```

## Persistent Volumes for Model Storage

Use PersistentVolumeClaims to store large model weights independently of pod lifecycle.

```yaml
# k8s/base/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes: [ReadOnlyMany]  # many pods mount read-only; use ReadWriteOnce for checkpoints
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
```

### Init Container to Download Models

```yaml
spec:
  initContainers:
    - name: model-downloader
      image: google/cloud-sdk:slim
      command: ["gsutil", "cp", "gs://my-ml-bucket/models/resnet50.onnx", "/models/resnet50.onnx"]
      volumeMounts:
        - { name: model-storage, mountPath: /models }
  containers:
    - name: model-server
      image: registry.example.com/ml-images/inference:v1.2.0
      volumeMounts:
        - { name: model-storage, mountPath: /models, readOnly: true }
  volumes:
    - name: model-storage
      emptyDir: { sizeLimit: 50Gi }
```

## Helm Chart Patterns

Organize manifests into a Helm chart (`helm/model-server/`) with `Chart.yaml`, a base `values.yaml` plus per-env `values-{dev,staging,prod}.yaml`, and a `templates/` dir holding templated copies of each manifest (deployment, service, ingress, hpa, configmap, secret, pvc) plus `_helpers.tpl` and `NOTES.txt`.

### values.yaml

```yaml
# helm/model-server/values.yaml
replicaCount: 2

image:
  repository: registry.example.com/ml-images/inference
  tag: "v1.2.0"
  pullPolicy: IfNotPresent

model:
  name: resnet50
  version: v1.2.0
  path: /models/resnet50_v1.2.0.onnx
  storageSizeGi: 50

resources:
  requests: { cpu: "2", memory: "4Gi", nvidia.com/gpu: "1" }
  limits: { cpu: "4", memory: "8Gi", nvidia.com/gpu: "1" }

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70

ingress:
  enabled: true
  host: api.ml.example.com
  tls: true

# probes: liveness/readiness/startup paths + timings (see Deployment manifest)

nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-t4

tolerations:
  - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
```

### Helm Deployment Commands

```bash
helm install model-server ./helm/model-server \
    --namespace ml-inference --create-namespace \
    --values helm/model-server/values-prod.yaml

helm upgrade model-server ./helm/model-server --set image.tag=v1.3.0
helm rollback model-server 1 --namespace ml-inference
helm template model-server ./helm/model-server \
    --values helm/model-server/values-staging.yaml --debug  # preview
```

## Health Checks (Liveness, Readiness, and Startup Probes)

ML containers need long startup times for model loading, so use all three probe types (see the Deployment manifest above for the full YAML): a `startupProbe` (`failureThreshold: 30`, `periodSeconds: 10` allows ~5 min to load), a `livenessProbe` to restart hung processes, and a `readinessProbe` on `/health/ready` to gate traffic until the model is loaded.

### FastAPI Health Endpoints

`/health` (liveness) returns alive as soon as the process is up; `/health/ready` (readiness) must return 503 until the model is loaded so traffic is not routed early:

```python
from fastapi import FastAPI, Response, status

app = FastAPI()
model_loaded = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(response: Response) -> dict[str, str]:
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "model loading"}
    return {"status": "ready"}
```

## Namespace Organization

Separate environments and workload types using namespaces (`ml-dev`, `ml-staging`, `ml-prod`, `ml-training`) via `kubectl create namespace <name>`, each with a resource quota.

### Resource Quotas per Namespace

```yaml
# k8s/namespaces/ml-prod-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-prod
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.cpu: "32"
    requests.memory: "128Gi"
    persistentvolumeclaims: "20"
```

## Kustomize Overlays for Dev/Staging/Prod

Use Kustomize to manage environment-specific variations without duplicating manifests. Layout: `k8s/base/` holds the shared manifests plus a `kustomization.yaml`; `k8s/overlays/{dev,staging,prod}/` each hold a `kustomization.yaml` and a `patches/` directory.

### Base Kustomization

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources: [deployment.yaml, service.yaml, ingress.yaml, hpa.yaml, configmap.yaml, pvc.yaml]
commonLabels:
  app.kubernetes.io/name: model-server
  app.kubernetes.io/managed-by: kustomize
```

### Overlay Example

Each overlay references `../../base`, sets a `namespace`, pins the image `newTag`, and applies patches. Only the patch values differ per environment.

```yaml
# k8s/overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ml-dev
resources:
  - ../../base
patches:
  - path: patches/deployment-patch.yaml
  - path: patches/hpa-patch.yaml
images:
  - name: registry.example.com/ml-images/inference
    newTag: dev-latest
```

```yaml
# k8s/overlays/dev/patches/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: model-server
          resources:
            requests: { cpu: "1", memory: "2Gi", nvidia.com/gpu: "1" }
            limits: { cpu: "2", memory: "4Gi", nvidia.com/gpu: "1" }
```

The `hpa-patch.yaml` in the same overlay just overrides `minReplicas`/`maxReplicas` (e.g. `1`/`2` for dev). Staging and prod use the same structure with scaled-up values: prod pins a real version tag (e.g. `newTag: v1.2.0`), sets `namespace: ml-prod`, `replicas: 3`, and larger CPU/memory requests (e.g. `cpu: "4"`, `memory: "8Gi"`) with `minReplicas` raised accordingly.

```bash
kubectl apply -k k8s/overlays/dev/        # or staging/ , prod/
kubectl kustomize k8s/overlays/prod/       # preview rendered manifests without applying
```

## Training Jobs with Kubernetes

Use Kubernetes Jobs for one-off training runs.

```yaml
# k8s/jobs/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-resnet50
  namespace: ml-training
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: trainer
          image: registry.example.com/ml-images/training:v1.2.0
          command: ["python", "-m", "my_project.train"]
          args: ["--config=configs/train.yaml", "--epochs=100"]
          resources:
            requests: { cpu: "8", memory: "32Gi", nvidia.com/gpu: "4" }
            limits: { cpu: "16", memory: "64Gi", nvidia.com/gpu: "4" }
          volumeMounts:
            - { name: data, mountPath: /data, readOnly: true }
            - { name: checkpoints, mountPath: /checkpoints }
          envFrom:
            - secretRef: { name: training-secrets }
      volumes:
        - name: data
          persistentVolumeClaim: { claimName: training-data-pvc }
        - name: checkpoints
          persistentVolumeClaim: { claimName: checkpoints-pvc }
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
```

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

## Best Practices

The positive form of the anti-patterns above: always set GPU/CPU/memory requests and limits; use all three probe types; separate environments by namespace with resource quotas; manage env differences with Kustomize/Helm (never duplicated YAML); pin image tags; run as non-root; store model weights on PVCs (not in images); set pod disruption budgets; scale HPA on request rate or GPU utilization; and pull weights via init containers.
