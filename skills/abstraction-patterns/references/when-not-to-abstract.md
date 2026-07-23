# When NOT to Abstract

Scope: the three failure modes that produce over-engineered code, each with a bad/good pair.

## Do not abstract single-use logic

```python
# BAD: Unnecessary abstraction for one-off logic
class ImagePreprocessor:
    def __init__(self, size):
        self.size = size

    def process(self, image):
        return cv2.resize(image, self.size)

# GOOD: Just use the function directly
resized = cv2.resize(image, (224, 224))
```

## Do not hide critical details

```python
# BAD: Hides GPU memory management
class AutoBatcher:
    def auto_batch(self, items):
        # Magically figures out batch size based on GPU memory
        # Developer has no idea what's happening
        ...

# GOOD: Be explicit about batch size
for batch in DataLoader(dataset, batch_size=32):
    ...
```

## Do not create classes for single functions

```python
# BAD: A class with one method is just a function
class NMSProcessor:
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, boxes, scores):
        return nms(boxes, scores, self.threshold)

# GOOD: Use functools.partial or just pass the argument
from functools import partial

apply_nms = partial(nms, iou_threshold=0.5)
filtered = apply_nms(boxes, scores)
```
