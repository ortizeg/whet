# ConfigMaps, Secrets, and Namespace Organization

How to inject model configuration and credentials into pods, and how to partition a cluster by environment.

## ConfigMap

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

## Secret

Never commit plaintext secrets to a `Secret` manifest — create them at deploy time (or use sealed-secrets):

```bash
kubectl create secret generic model-secrets \
    --from-literal=WANDB_API_KEY="${WANDB_API_KEY}" \
    --from-literal=S3_ACCESS_KEY="${S3_ACCESS_KEY}" \
    --namespace=ml-inference
```

Both are consumed by the Deployment via `envFrom`:

```yaml
          envFrom:
            - configMapRef: { name: model-config }
            - secretRef: { name: model-secrets }
```

## Namespace Organization

Separate environments and workload types using namespaces (`ml-dev`, `ml-staging`, `ml-prod`, `ml-training`) via `kubectl create namespace <name>`, each with a resource quota.

## Resource Quotas per Namespace

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

Without quotas on a shared cluster, a single team can monopolize all GPU resources.
