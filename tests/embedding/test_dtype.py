import pytest
import torch

from egaudio.embedding.model import resolve_dtype


def test_bf16_and_fp32_allowed():
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32


@pytest.mark.parametrize("name", ["float16", "fp16", "half"])
def test_fp16_forbidden(name):
    with pytest.raises(ValueError):
        resolve_dtype(name)
