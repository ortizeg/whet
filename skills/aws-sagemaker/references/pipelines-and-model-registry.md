# SageMaker Pipelines and Model Registry

Wiring preprocessing, training, and conditional model registration into a single reproducible pipeline definition.

## Contents

- [Pipeline Definition](#pipeline-definition)
- [How Steps Pass Data](#how-steps-pass-data)
- [Conditional Registration](#conditional-registration)
- [Running a Pipeline](#running-a-pipeline)

## Pipeline Definition

End-to-end preprocess → train → conditional-register. Steps pass data via `.properties` references; registration only runs if accuracy clears a threshold.

```python
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def create_pipeline(role: str, pipeline_name: str = "cv-training-pipeline") -> Pipeline:
    session = sagemaker.Session()
    input_data = ParameterString(name="InputData")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.90)

    processor = ScriptProcessor(
        role=role,
        image_uri=session.sagemaker_client.describe_image("pytorch-training")["ImageUri"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        command=["python3"],
    )
    preprocess = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="val", source="/opt/ml/processing/output/val"),
        ],
        code="src/processing/preprocess.py",
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role=role,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
    )
    outs = preprocess.properties.ProcessingOutputConfig.Outputs
    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={"train": outs["train"].S3Output.S3Uri, "validation": outs["val"].S3Output.S3Uri},
    )
    register = ModelStep(
        name="RegisterModel",
        step_args=estimator.register(
            content_types=["application/json"],
            response_types=["application/json"],
            model_package_group_name="cv-models",
            approval_status="PendingManualApproval",
        ),
    )
    condition = ConditionStep(
        name="CheckAccuracy",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=train_step.name, property_file="metrics", json_path="val_accuracy"),
            right=accuracy_threshold,
        )],
        if_steps=[register],
        else_steps=[],
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[input_data, accuracy_threshold],
        steps=[preprocess, train_step, condition],
        sagemaker_session=session,
    )
```

## How Steps Pass Data

- `preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri` is a
  **symbolic reference**, not a value — it resolves at execution time. Referencing it in
  `train_step` is also what creates the dependency edge, so the steps run in the right order
  without an explicit `depends_on`.
- Never interpolate these references into an f-string; they are objects, and stringifying them
  produces a placeholder rather than the real URI.
- `ParameterString` / `ParameterFloat` are the pipeline's inputs. Supply them at start time so
  one definition serves dev and prod without editing code — this is what replaces hardcoded S3
  paths.
- `ProcessingStep` inputs/outputs map S3 prefixes to container paths under
  `/opt/ml/processing/`; the processing script reads and writes those local paths only.

## Conditional Registration

`JsonGet(step_name=..., property_file="metrics", json_path="val_accuracy")` reads a value out of
a property file the training job emitted. The training script must write that JSON into
`SM_OUTPUT_DATA_DIR` and the step must declare the property file for this to resolve.

`approval_status="PendingManualApproval"` puts the new model package in the registry without
promoting it — a human (or a separate approval pipeline) flips it to `Approved` before
deployment. Using `Approved` directly removes the only human gate between training and
production; do it only when downstream deployment has its own gate.

`model_package_group_name` groups versions of the same logical model. Each pipeline run that
clears the threshold appends a new version, which is what gives you rollback.

## Running a Pipeline

```python
pipeline = create_pipeline(role=role)
pipeline.upsert(role_arn=role)
execution = pipeline.start(parameters={"InputData": "s3://bucket/raw/", "AccuracyThreshold": 0.92})
```

`upsert` creates or updates the definition; `start` launches an execution. Poll with
`execution.describe()` rather than blocking — pipeline runs take hours.
