# GPU Scheduling on Kubernetes

How to request GPUs for ML pods, and the cluster prerequisites that make GPU scheduling work.

## GPU Resource Requests

Always set both `requests` and `limits` for `nvidia.com/gpu`, and they must be equal — GPU scheduling requires exact counts (fractional GPUs are not natively supported). Multi-GPU pods just raise the count alongside CPU/memory:

```yaml
resources:
  requests: { cpu: "8", memory: "32Gi", nvidia.com/gpu: "4" }
  limits: { cpu: "16", memory: "64Gi", nvidia.com/gpu: "4" }
```

Do not set CPU requests too low for GPU pods — GPU inference still requires CPU for preprocessing, and under-provisioned CPU starves the pipeline.

## NVIDIA Device Plugin

GPU scheduling requires the NVIDIA device plugin deployed in-cluster:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
    --namespace kube-system --set runtimeClassName=nvidia
```

## Node Selection and Tolerations

GPU nodes are normally tainted, so GPU pods must tolerate the taint and select the accelerator type they need:

```yaml
      tolerations:
        - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

Training workloads typically select a larger accelerator class, e.g.:

```yaml
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
```
