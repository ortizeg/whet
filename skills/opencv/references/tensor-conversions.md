# NumPy / PyTorch Conversions

Scope: converting between NumPy HWC uint8 images and PyTorch CHW/BCHW float32 tensors.

## NumPy / PyTorch Conversions

```python
import torch


def numpy_to_torch(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 tensor."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """CHW float32 tensor -> HWC uint8."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def batch_to_numpy(batch: torch.Tensor) -> list[np.ndarray]:
    """BCHW tensor -> list of HWC uint8 arrays."""
    return [torch_to_numpy(batch[i]) for i in range(batch.shape[0])]
```
