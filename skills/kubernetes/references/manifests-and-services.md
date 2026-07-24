# Kubernetes Manifests, Services, and Health Probes

Full Deployment manifest for an ML inference service, plus the Service, Ingress, and health-probe patterns that expose it.

## Contents

- [Inference Deployment](#inference-deployment)
- [Service](#service)
- [Ingress](#ingress)
- [Health Checks (Liveness, Readiness, and Startup Probes)](#health-checks-liveness-readiness-and-startup-probes)
- [FastAPI Health Endpoints](#fastapi-health-endpoints)

## Inference Deployment

Define Deployments with explicit resource requests and GPU scheduling for inference workloads.

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

## Service

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

## Ingress

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

## Health Checks (Liveness, Readiness, and Startup Probes)

ML containers need long startup times for model loading, so use all three probe types (see the Deployment manifest above for the full YAML): a `startupProbe` (`failureThreshold: 30`, `periodSeconds: 10` allows ~5 min to load), a `livenessProbe` to restart hung processes, and a `readinessProbe` on `/health/ready` to gate traffic until the model is loaded.

## FastAPI Health Endpoints

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
