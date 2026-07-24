# Config Groups and Composition

Scope: the `conf/` layout, the YAML files that make up each config group, the defaults list, `@package` directives, and recursive/experiment configs.

## Contents

- [Directory structure](#directory-structure)
- [conf/config.yaml](#confconfigyaml)
- [Training group](#training-group)
- [Data group](#data-group)
- [Model group](#model-group)
- [Augmentation group](#augmentation-group)
- [Config groups with package directive](#config-groups-with-package-directive)
- [Recursive defaults and experiment configs](#recursive-defaults-and-experiment-configs)

## Directory structure

Organize configuration files by concern. Each subdirectory represents a config group that can be swapped via the defaults list or CLI overrides.

```
conf/
├── config.yaml          # Default config (top-level)
├── training/
│   ├── default.yaml
│   ├── fast.yaml        # Quick training for debugging
│   └── full.yaml        # Full training run
├── data/
│   ├── coco.yaml
│   ├── imagenet.yaml
│   └── custom.yaml
├── model/
│   ├── resnet50.yaml
│   ├── efficientnet.yaml
│   └── yolov8.yaml
├── augmentation/
│   ├── basic.yaml
│   ├── heavy.yaml
│   └── none.yaml
└── experiment/
    ├── debug.yaml
    └── production.yaml
```

## conf/config.yaml

The top-level config uses the `defaults` list to compose from config groups. The `_self_` entry controls where this file's values are placed relative to the defaults.

```yaml
defaults:
  - training: default
  - data: coco
  - model: resnet50
  - augmentation: basic
  - _self_

seed: 42
experiment_name: "default_experiment"

# Interpolation example
output_dir: "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
```

## Training group

### conf/training/default.yaml

```yaml
learning_rate: 1e-3
batch_size: 32
epochs: 100
optimizer: adamw
weight_decay: 0.01
scheduler: cosine
warmup_epochs: 5
gradient_clip_val: 1.0
```

### conf/training/fast.yaml

```yaml
learning_rate: 1e-3
batch_size: 64
epochs: 10
optimizer: adam
weight_decay: 0.0
scheduler: none
warmup_epochs: 0
gradient_clip_val: null
```

### conf/training/full.yaml

```yaml
learning_rate: 3e-4
batch_size: 16
epochs: 300
optimizer: adamw
weight_decay: 0.05
scheduler: cosine
warmup_epochs: 10
gradient_clip_val: 1.0
```

## Data group

### conf/data/coco.yaml

```yaml
data_dir: "/data/coco"
train_split: 0.8
val_split: 0.1
test_split: 0.1
num_workers: 8
pin_memory: true
image_size: 640
format: "coco"
```

### conf/data/imagenet.yaml

```yaml
data_dir: "/data/imagenet"
train_split: 0.9
val_split: 0.05
test_split: 0.05
num_workers: 16
pin_memory: true
image_size: 224
format: "imagefolder"
```

## Model group

### conf/model/resnet50.yaml

```yaml
name: resnet50
pretrained: true
num_classes: 80
dropout: 0.1
freeze_backbone: false
backbone_lr_factor: 0.1
```

## Augmentation group

### conf/augmentation/heavy.yaml

```yaml
horizontal_flip: 0.5
vertical_flip: 0.0
rotation_limit: 30
brightness_limit: 0.3
contrast_limit: 0.3
hue_shift_limit: 20
mosaic_prob: 0.5
mixup_prob: 0.3
cutout_prob: 0.2
```

## Config groups with package directive

When you need to mount a config group at a specific path in the config tree, use the `@package` directive.

```yaml
# conf/server/apache.yaml
# @package _group_
host: localhost
port: 8080
```

## Recursive defaults and experiment configs

Compose configs that themselves reference other defaults.

```yaml
# conf/experiment/production.yaml
defaults:
  - /training: full
  - /data: imagenet
  - /model: efficientnet
  - /augmentation: heavy

seed: 0
experiment_name: "production_run"
```

Then run with:

```bash
python train.py +experiment=production
```

Keep named experiment configs (e.g. `experiment/ablation_v3.yaml`) checked in so a specific combination of config groups can be reproduced by name.
