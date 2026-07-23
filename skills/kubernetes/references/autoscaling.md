# Autoscaling ML Inference Pods

HorizontalPodAutoscaler configuration for model servers, including custom-metric scaling.

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

## Notes

- CPU utilization alone is a poor proxy for GPU inference load — prefer a request-rate or GPU-utilization custom metric where a metrics adapter is available.
- Asymmetric `behavior` windows (fast up, slow down) prevent replica flapping under bursty traffic.
- Pair the HPA with a PodDisruptionBudget (`minAvailable`) so node upgrades cannot evict every replica at once.
