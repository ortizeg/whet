# Helm Charts and Kustomize Overlays

Packaging ML manifests and managing dev/staging/prod differences without duplicating YAML.

## Contents

- [Helm Chart Patterns](#helm-chart-patterns)
- [values.yaml](#valuesyaml)
- [Helm Deployment Commands](#helm-deployment-commands)
- [Kustomize Overlays for Dev/Staging/Prod](#kustomize-overlays-for-devstagingprod)
- [Base Kustomization](#base-kustomization)
- [Overlay Example](#overlay-example)

## Helm Chart Patterns

Organize manifests into a Helm chart (`helm/model-server/`) with `Chart.yaml`, a base `values.yaml` plus per-env `values-{dev,staging,prod}.yaml`, and a `templates/` dir holding templated copies of each manifest (deployment, service, ingress, hpa, configmap, secret, pvc) plus `_helpers.tpl` and `NOTES.txt`.

## values.yaml

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

## Helm Deployment Commands

```bash
helm install model-server ./helm/model-server \
    --namespace ml-inference --create-namespace \
    --values helm/model-server/values-prod.yaml

helm upgrade model-server ./helm/model-server --set image.tag=v1.3.0
helm rollback model-server 1 --namespace ml-inference
helm template model-server ./helm/model-server \
    --values helm/model-server/values-staging.yaml --debug  # preview
```

## Kustomize Overlays for Dev/Staging/Prod

Use Kustomize to manage environment-specific variations without duplicating manifests. Layout: `k8s/base/` holds the shared manifests plus a `kustomization.yaml`; `k8s/overlays/{dev,staging,prod}/` each hold a `kustomization.yaml` and a `patches/` directory.

## Base Kustomization

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources: [deployment.yaml, service.yaml, ingress.yaml, hpa.yaml, configmap.yaml, pvc.yaml]
commonLabels:
  app.kubernetes.io/name: model-server
  app.kubernetes.io/managed-by: kustomize
```

## Overlay Example

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
