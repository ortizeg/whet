# Batch Transform

Offline inference over an S3 dataset, without standing up a persistent endpoint.

## Running a Transform Job

Offline inference over an S3 dataset — reuses the same `PyTorchModel`/inference script as endpoints:

```python
from sagemaker.pytorch import PyTorchModel


def run_batch_transform(
    model_data_s3: str, input_s3_uri: str, output_s3_uri: str, role: str,
    instance_type: str = "ml.g5.xlarge",
) -> None:
    model = PyTorchModel(
        model_data=model_data_s3, role=role,
        framework_version="2.1.0", py_version="py310",
        entry_point="inference.py", source_dir="src/inference",
    )
    transformer = model.transformer(
        instance_count=1, instance_type=instance_type,
        output_path=output_s3_uri, strategy="MultiRecord", max_payload=6,
    )
    transformer.transform(data=input_s3_uri, content_type="application/json", split_type="Line")
```

## When to Use It

Batch transform beats a real-time endpoint whenever there is no interactive consumer: nightly
scoring, backfilling predictions over a historical dataset, or evaluating a candidate model over
a full validation set. The instances are provisioned for the job and torn down afterwards, so
you pay only for the runtime instead of an always-on endpoint.

## Parameters That Matter

- **`strategy="MultiRecord"`** batches as many records as fit within `max_payload` into a single
  invocation — far faster than `SingleRecord`, which calls the handler once per line. Use
  `SingleRecord` only when the inference script cannot handle batched input.
- **`max_payload=6`** is megabytes per request. Raise it for larger batches, but the handler
  must be able to hold that much decoded data in memory.
- **`split_type="Line"`** tells SageMaker how to split the input objects into records; use
  `"Line"` for JSON-Lines input, `"None"` when each S3 object is one record (e.g. one image
  per file, with `content_type="application/x-image"`).
- **`instance_count`** scales horizontally by sharding the input objects — increase it for large
  datasets rather than reaching for a bigger single instance.

## Output Layout

Results land in `output_path` as one `.out` object per input object, in the same relative
prefix structure. To correlate predictions with inputs, either preserve an id field inside each
record or rely on the ordering within each file — SageMaker does not add one for you.

The inference script is identical to the endpoint one (`model_fn`, `input_fn`, `predict_fn`,
`output_fn`), which is the main reason to keep it free of endpoint-specific assumptions.
