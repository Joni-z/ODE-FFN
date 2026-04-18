import torch

from src.engine_jit import _prepare_batch


def test_prepare_batch_normalizes_uint8_once():
    x = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
    labels = torch.tensor([7], dtype=torch.long)

    x_prepared, labels_prepared = _prepare_batch(x, labels, torch.device("cpu"))

    assert x_prepared.dtype == torch.float32
    assert torch.allclose(x_prepared, torch.tensor([[[[-1.0, 1.0]]]]))
    assert torch.equal(labels_prepared, labels)


def test_prepare_batch_keeps_normalized_float_unchanged():
    x = torch.tensor([[[[-1.0, 1.0]]]], dtype=torch.float32)
    labels = torch.tensor([3], dtype=torch.long)

    x_prepared, labels_prepared = _prepare_batch(x, labels, torch.device("cpu"))

    assert torch.allclose(x_prepared, x)
    assert torch.equal(labels_prepared, labels)
