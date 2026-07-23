# Training Jobs on Kubernetes

Running one-off or scheduled model training on a cluster with `batch/v1` Jobs.

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

## Notes

- `backoffLimit` caps retries so a systematically failing run does not burn GPU hours indefinitely.
- `ttlSecondsAfterFinished` garbage-collects completed Jobs (86400 = 1 day) so the namespace does not fill with finished pods.
- `restartPolicy: OnFailure` restarts the container in place; combine it with checkpoint resume so a restart does not lose progress.
- Mount data read-only and checkpoints read-write on separate PVCs — the training data should never be mutable from the trainer.
- Run training in its own namespace (`ml-training`) with its own GPU quota so training cannot starve production inference.
