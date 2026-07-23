# Parametrized Tests

Using `@pytest.mark.parametrize` to cover shapes, formats, and configurations without duplicating test bodies.

## Usage

Use `@pytest.mark.parametrize` (stack decorators for a cross product) and `ids=` for readable case names.

```python
import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("image_size", [224, 320, 640])
def test_model_handles_various_sizes(batch_size, image_size):
    """Model should handle different batch sizes and resolutions."""
    model = MyModel(num_classes=10)
    x = torch.randn(batch_size, 3, image_size, image_size)
    output = model(x)
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize(
    "bbox_format,expected",
    [
        ("xyxy", [10, 20, 110, 220]),
        ("xywh", [10, 20, 100, 200]),
        ("cxcywh", [60, 120, 100, 200]),
    ],
)
def test_bbox_conversion(bbox_format, expected):
    """Test bounding box format conversions."""
    result = convert_bbox([10, 20, 110, 220], from_format="xyxy", to_format=bbox_format)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "num_classes,input_shape",
    [(10, (1, 3, 224, 224)), (1000, (1, 3, 384, 384))],
    ids=["10cls-single", "1000cls-highres"],
)
def test_classifier_output(num_classes, input_shape):
    """Test classifier with various class counts and inputs."""
    model = Classifier(num_classes=num_classes)
    x = torch.randn(*input_shape)
    out = model(x)
    assert out.shape == (input_shape[0], num_classes)
```

## Notes

- Stacked `parametrize` decorators produce the full cross product — three batch sizes × three image sizes is nine test cases.
- Supply `ids=` whenever the parameters are tuples or objects; the default generated ids are unreadable in CI output.
- A parametrized fixture (`@pytest.fixture(params=[...])`) is the right tool when the variation belongs to a shared setup rather than to one test.
