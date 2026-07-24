# Persistent Volumes and Model Storage

Getting large model weights onto pods without baking them into container images.

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

## Init Container to Download Models

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

## Choosing Between the Two

- **PVC (`ReadOnlyMany`)** — weights are written once and shared read-only by many replicas; survives pod restarts and avoids repeated downloads.
- **Init container + `emptyDir`** — no shared storage class required; each pod pulls its own copy at startup. Simpler, but slower to scale up and repeats egress cost per pod.
- Never store model weights inside container images — images become multi-gigabyte and slow to pull.
