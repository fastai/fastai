"""Tests for fastai.torch_core module.

Covers TensorBase, tensor creation, utility functions, one_hot encoding,
device management, and other core tensor operations.
"""
import sys
import os
import pytest
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.torch_core import (
    TensorBase, TensorImage, TensorImageBW, TensorMask, TensorCategory,
    TensorMultiCategory, TitledTensorScalar,
    tensor, set_seed, unsqueeze, unsqueeze_,
    to_detach, to_half, to_float, to_cpu, to_np, to_concat, to_device,
    default_device, one_hot, one_hot_decode, concat, flatten_check,
    apply, logit, make_cross_image, Chunks, params, trainable_params,
    get_random_states, no_random,
)


# ============================================================
# Tests for `tensor` function
# ============================================================

class TestTensorCreation:
    """Tests for the `tensor()` function."""

    def test_tensor_from_list(self):
        t = tensor([1, 2, 3])
        assert isinstance(t, Tensor)
        assert t.shape == (3,)
        assert t.tolist() == [1, 2, 3]

    def test_tensor_from_tuple(self):
        t = tensor((4.0, 5.0, 6.0))
        assert isinstance(t, Tensor)
        assert t.dtype == torch.float32

    def test_tensor_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = tensor(arr)
        assert isinstance(t, Tensor)
        assert t.dtype == torch.float32
        np.testing.assert_allclose(t.numpy(), arr)

    def test_tensor_from_numpy_float64_converts_to_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        t = tensor(arr)
        # float64 should be downcast to float32
        assert t.dtype == torch.float32

    def test_tensor_from_existing_tensor(self):
        original = torch.tensor([1, 2, 3])
        t = tensor(original)
        assert t is original  # should return same tensor

    def test_tensor_multiple_args(self):
        t = tensor(1, 2, 3)
        assert t.tolist() == [1, 2, 3]

    def test_tensor_from_scalar(self):
        t = tensor(5)
        assert t.item() == 5

    def test_tensor_from_numpy_uint16(self):
        arr = np.array([1, 2, 3], dtype=np.uint16)
        t = tensor(arr)
        # uint16 should be converted to float32
        assert t.dtype == torch.float32


# ============================================================
# Tests for TensorBase
# ============================================================

class TestTensorBase:
    """Tests for the TensorBase class and its subclasses."""

    def test_creation_from_list(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        assert isinstance(tb, TensorBase)
        assert isinstance(tb, Tensor)
        assert tb.shape == (3,)

    def test_creation_from_tensor(self):
        t = torch.randn(3, 4)
        tb = TensorBase(t)
        assert isinstance(tb, TensorBase)
        assert tb.shape == (3, 4)
        assert torch.equal(tb, t)

    def test_repr_uses_class_name(self):
        tb = TensorBase([1.0, 2.0])
        assert 'TensorBase' in repr(tb)

    def test_subclass_preserves_type_after_operations(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        result = tb + 1
        assert isinstance(result, TensorBase)

    def test_requires_grad_workaround(self):
        """Test the requires_grad_ workaround for pytorch#50219."""
        tb = TensorBase([1.0, 2.0, 3.0])
        result = tb.requires_grad_(True)
        assert result.requires_grad is True
        assert result is tb

    def test_requires_grad_false(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        tb.requires_grad_(True)
        tb.requires_grad_(False)
        assert tb.requires_grad is False

    def test_clone_preserves_type(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        cloned = tb.clone()
        assert isinstance(cloned, TensorBase)
        assert torch.equal(tb, cloned)

    def test_new_ones_preserves_type(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        new = tb.new_ones((2, 3))
        assert isinstance(new, TensorBase)
        assert new.shape == (2, 3)
        assert torch.all(new == 1.0)

    def test_new_tensor_preserves_type(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        new = tb.new_tensor([4.0, 5.0])
        assert isinstance(new, TensorBase)
        assert new.tolist() == [4.0, 5.0]

    def test_metadata_preserved(self):
        tb = TensorBase([1.0, 2.0, 3.0])
        tb.custom_attr = 'test_value'
        result = tb * 2
        assert hasattr(result, 'custom_attr')
        assert result.custom_attr == 'test_value'


class TestTensorSubclasses:
    """Tests for TensorImage, TensorMask, TensorCategory, etc."""

    def test_tensor_image_creation(self):
        ti = TensorImage(torch.randn(3, 32, 32))
        assert isinstance(ti, TensorImage)
        assert isinstance(ti, TensorBase)

    def test_tensor_image_bw_creation(self):
        ti = TensorImageBW(torch.randn(1, 32, 32))
        assert isinstance(ti, TensorImageBW)

    def test_tensor_mask_creation(self):
        tm = TensorMask(torch.zeros(32, 32))
        assert isinstance(tm, TensorMask)

    def test_tensor_category_creation(self):
        tc = TensorCategory(torch.tensor([0, 1, 2]))
        assert isinstance(tc, TensorCategory)

    def test_tensor_multi_category_creation(self):
        tmc = TensorMultiCategory(torch.tensor([0, 1, 0, 1]))
        assert isinstance(tmc, TensorMultiCategory)

    def test_titled_tensor_scalar(self):
        tts = TitledTensorScalar(torch.tensor(3.14))
        assert isinstance(tts, TitledTensorScalar)
        assert abs(tts.item() - 3.14) < 1e-5


# ============================================================
# Tests for utility functions
# ============================================================

class TestUtilityFunctions:
    """Tests for various utility functions in torch_core."""

    def test_unsqueeze_single(self):
        t = torch.randn(3, 4)
        result = unsqueeze(t, dim=-1, n=1)
        assert result.shape == (3, 4, 1)

    def test_unsqueeze_multiple(self):
        t = torch.randn(3, 4)
        result = unsqueeze(t, dim=-1, n=3)
        assert result.shape == (3, 4, 1, 1, 1)

    def test_unsqueeze_dim_zero(self):
        t = torch.randn(3, 4)
        result = unsqueeze(t, dim=0, n=2)
        assert result.shape == (1, 1, 3, 4)

    def test_unsqueeze_inplace(self):
        t = torch.randn(3, 4)
        result = unsqueeze_(t, dim=-1, n=2)
        assert result.shape == (3, 4, 1, 1)

    def test_to_detach_tensor(self):
        t = torch.randn(3, 4, requires_grad=True)
        result = to_detach(t)
        assert not result.requires_grad
        assert result.device.type == 'cpu'

    def test_to_detach_list(self):
        tensors = [torch.randn(2, 3, requires_grad=True) for _ in range(3)]
        results = to_detach(tensors)
        for r in results:
            assert not r.requires_grad

    def test_to_half(self):
        t = torch.randn(3, 4)
        result = to_half(t)
        assert result.dtype == torch.float16

    def test_to_half_integer_unchanged(self):
        t = torch.tensor([1, 2, 3])
        result = to_half(t)
        assert result.dtype == torch.int64  # integers stay as-is

    def test_to_float(self):
        t = torch.randn(3, 4).half()
        result = to_float(t)
        assert result.dtype == torch.float32

    def test_to_cpu(self):
        t = torch.randn(3, 4)
        result = to_cpu(t)
        assert result.device.type == 'cpu'

    def test_to_np(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_np(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_to_concat_tensors(self):
        xs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = to_concat(xs)
        assert result.tolist() == [1, 2, 3, 4]

    def test_to_concat_empty(self):
        result = to_concat([])
        assert result == []

    def test_apply_function(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = apply(lambda x: x * 2, t)
        assert torch.equal(result, torch.tensor([2.0, 4.0, 6.0]))

    def test_apply_to_list(self):
        tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
        results = apply(lambda x: x * 3, tensors)
        assert len(results) == 2
        assert results[0].item() == 3.0
        assert results[1].item() == 6.0

    def test_apply_to_dict(self):
        d = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        results = apply(lambda x: x + 1, d)
        assert results['a'].item() == 2.0
        assert results['b'].item() == 3.0


# ============================================================
# Tests for one_hot encoding
# ============================================================

class TestOneHot:
    """Tests for one_hot and one_hot_decode."""

    def test_one_hot_single_class(self):
        result = one_hot(0, 5)
        expected = torch.tensor([1, 0, 0, 0, 0], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_one_hot_last_class(self):
        result = one_hot(4, 5)
        expected = torch.tensor([0, 0, 0, 0, 1], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_one_hot_multiple_classes(self):
        result = one_hot([0, 2, 4], 5)
        expected = torch.tensor([1, 0, 1, 0, 1], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_one_hot_tensor_input(self):
        result = one_hot(torch.tensor([1, 3]), 5)
        expected = torch.tensor([0, 1, 0, 1, 0], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_one_hot_decode_basic(self):
        encoded = torch.tensor([1, 0, 1, 0, 1], dtype=torch.uint8)
        result = one_hot_decode(encoded)
        assert list(result) == [0, 2, 4]

    def test_one_hot_decode_with_vocab(self):
        encoded = torch.tensor([1, 0, 1, 0, 0], dtype=torch.uint8)
        vocab = ['cat', 'dog', 'bird', 'fish', 'snake']
        result = one_hot_decode(encoded, vocab)
        assert list(result) == ['cat', 'bird']


# ============================================================
# Tests for set_seed and randomness control
# ============================================================

class TestRandomness:
    """Tests for set_seed and no_random context manager."""

    def test_set_seed_reproducibility(self):
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        assert torch.equal(t1, t2)

    def test_set_seed_different_seeds(self):
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(123)
        t2 = torch.randn(5)
        assert not torch.equal(t1, t2)

    def test_no_random_context_manager(self):
        with no_random(seed=42):
            t1 = torch.randn(5)
        with no_random(seed=42):
            t2 = torch.randn(5)
        assert torch.equal(t1, t2)

    def test_no_random_restores_state(self):
        set_seed(100)
        before = torch.randn(3)
        set_seed(100)
        # Use no_random in between, which should restore state after
        with no_random(seed=999):
            _ = torch.randn(10)
        # After no_random, we should be back to original state
        after = torch.randn(3)
        assert torch.equal(before, after)

    def test_get_random_states_returns_dict(self):
        states = get_random_states()
        assert 'random_state' in states
        assert 'numpy_state' in states
        assert 'torch_state' in states


# ============================================================
# Tests for concat
# ============================================================

class TestConcat:
    """Tests for the concat function."""

    def test_concat_tensors(self):
        a = torch.tensor([1, 2])
        b = torch.tensor([3, 4])
        result = concat(a, b)
        assert result.tolist() == [1, 2, 3, 4]

    def test_concat_numpy_arrays(self):
        a = np.array([1, 2])
        b = np.array([3, 4])
        result = concat(a, b)
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_concat_lists(self):
        a = [1, 2]
        b = [3, 4]
        result = concat(a, b)
        assert list(result) == [1, 2, 3, 4]

    def test_concat_empty(self):
        result = concat()
        assert result == []


# ============================================================
# Tests for flatten_check
# ============================================================

class TestFlattenCheck:
    """Tests for the flatten_check function."""

    def test_flatten_check_same_shape(self):
        inp = torch.randn(2, 3)
        targ = torch.randn(2, 3)
        flat_inp, flat_targ = flatten_check(inp, targ)
        assert flat_inp.shape == (6,)
        assert flat_targ.shape == (6,)

    def test_flatten_check_1d(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([4.0, 5.0, 6.0])
        flat_inp, flat_targ = flatten_check(inp, targ)
        assert flat_inp.shape == (3,)
        assert flat_targ.shape == (3,)


# ============================================================
# Tests for logit
# ============================================================

class TestLogit:
    """Tests for the logit function."""

    def test_logit_half(self):
        t = torch.tensor([0.5])
        result = logit(t)
        assert abs(result.item()) < 1e-5  # logit(0.5) = 0

    def test_logit_near_one(self):
        t = torch.tensor([0.9])
        result = logit(t)
        assert result.item() > 0  # logit > 0 for p > 0.5

    def test_logit_near_zero(self):
        t = torch.tensor([0.1])
        result = logit(t)
        assert result.item() < 0  # logit < 0 for p < 0.5

    def test_logit_clamping(self):
        """Verify that logit handles extreme values without inf."""
        t = torch.tensor([0.0, 1.0])
        result = logit(t)
        assert torch.all(torch.isfinite(result))


# ============================================================
# Tests for make_cross_image
# ============================================================

class TestMakeCrossImage:
    """Tests for the make_cross_image function."""

    def test_bw_cross(self):
        im = make_cross_image(bw=True)
        assert im.shape == (5, 5)
        # Center row and col should be 1
        assert im[2, 0] == 1.0
        assert im[0, 2] == 1.0
        # Corners should be 0
        assert im[0, 0] == 0.0
        assert im[4, 4] == 0.0

    def test_color_cross(self):
        im = make_cross_image(bw=False)
        assert im.shape == (3, 5, 5)
        # Red channel: row 2 is 1
        assert im[0, 2, 0] == 1.0
        # Green channel: col 2 is 1
        assert im[1, 0, 2] == 1.0


# ============================================================
# Tests for params and trainable_params
# ============================================================

class TestParams:
    """Tests for params and trainable_params utility functions."""

    def test_params_returns_all(self):
        model = torch.nn.Linear(10, 5)
        p = params(model)
        assert len(p) == 2  # weight and bias

    def test_trainable_params(self):
        model = torch.nn.Linear(10, 5)
        tp = trainable_params(model)
        assert len(tp) == 2

    def test_trainable_params_frozen(self):
        model = torch.nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        tp = trainable_params(model)
        assert len(tp) == 0


# ============================================================
# Tests for Chunks
# ============================================================

class TestChunks:
    """Tests for the Chunks class."""

    def test_chunks_getitem(self):
        chunks = Chunks([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])])
        assert chunks[0].item() == 1
        assert chunks[3].item() == 4
        assert chunks[5].item() == 6

    def test_chunks_negative_indexing(self):
        chunks = Chunks([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])])
        assert chunks[-1].item() == 6

    def test_chunks_totlen(self):
        chunks = Chunks([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        assert chunks.totlen == 5


# ============================================================
# Tests for default_device and to_device
# ============================================================

class TestDevice:
    """Tests for device-related functions."""

    def test_default_device_cpu(self):
        device = default_device(use=False)
        assert device == torch.device('cpu')

    def test_to_device_cpu(self):
        t = torch.randn(3, 4)
        result = to_device(t, device='cpu')
        assert result.device.type == 'cpu'

    def test_to_device_nested_list(self):
        tensors = [torch.randn(2), torch.randn(3)]
        results = to_device(tensors, device='cpu')
        for r in results:
            assert r.device.type == 'cpu'
