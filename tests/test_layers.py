"""Tests for fastai.layers module.

Covers custom layers: Identity, Lambda, Flatten, SigmoidRange,
ConvLayer, AdaptiveConcatPool, BatchNorm, LinBnDrop, ParameterModule, etc.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.layers import (
    Identity, Lambda, PartialLambda, Flatten, View, ResizeBatch,
    sigmoid_range, SigmoidRange,
    AdaptiveConcatPool1d, AdaptiveConcatPool2d,
    PoolType, adaptive_pool, PoolFlatten,
    NormType, BatchNorm, InstanceNorm, BatchNorm1dFlat,
    LinBnDrop, ConvLayer, AdaptiveAvgPool, MaxPool, AvgPool,
    trunc_normal_, Embedding, sigmoid, sigmoid_,
    vleaky_relu, init_default,
    ParameterModule, children_and_parameters, has_children, flatten_model,
)


# ============================================================
# Tests for Identity layer
# ============================================================

class TestIdentity:
    """Tests for the Identity layer."""

    def test_forward(self):
        layer = Identity()
        x = torch.randn(2, 3)
        result = layer(x)
        assert torch.equal(result, x)

    def test_no_parameters(self):
        layer = Identity()
        assert len(list(layer.parameters())) == 0


# ============================================================
# Tests for Lambda layer
# ============================================================

class TestLambda:
    """Tests for the Lambda layer."""

    def test_forward_with_function(self):
        layer = Lambda(torch.relu)
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = layer(x)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.equal(result, expected)

    def test_forward_with_custom_function(self):
        layer = Lambda(lambda x: x * 2)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = layer(x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.equal(result, expected)


# ============================================================
# Tests for PartialLambda layer
# ============================================================

class TestPartialLambda:
    """Tests for the PartialLambda layer."""

    def test_forward_with_kwargs(self):
        def scale(x, factor=1.0):
            return x * factor

        layer = PartialLambda(scale, factor=3.0)
        x = torch.tensor([1.0, 2.0])
        result = layer(x)
        expected = torch.tensor([3.0, 6.0])
        assert torch.equal(result, expected)

    def test_repr(self):
        def my_func(x, a=1):
            return x + a

        layer = PartialLambda(my_func, a=5)
        assert 'my_func' in repr(layer)


# ============================================================
# Tests for Flatten layer
# ============================================================

class TestFlatten:
    """Tests for the Flatten layer."""

    def test_flatten_batch(self):
        layer = Flatten()
        x = torch.randn(4, 3, 8, 8)
        result = layer(x)
        assert result.shape == (4, 3 * 8 * 8)

    def test_flatten_full(self):
        layer = Flatten(full=True)
        x = torch.randn(4, 3, 8, 8)
        result = layer(x)
        assert result.shape == (4 * 3 * 8 * 8,)


# ============================================================
# Tests for View and ResizeBatch
# ============================================================

class TestViewAndResizeBatch:
    """Tests for View and ResizeBatch layers."""

    def test_view(self):
        layer = View(2, 6)
        x = torch.randn(12)
        result = layer(x)
        assert result.shape == (2, 6)

    def test_resize_batch(self):
        layer = ResizeBatch(3, 4)
        x = torch.randn(2, 12)
        result = layer(x)
        assert result.shape == (2, 3, 4)


# ============================================================
# Tests for sigmoid_range and SigmoidRange
# ============================================================

class TestSigmoidRange:
    """Tests for sigmoid_range function and SigmoidRange module."""

    def test_sigmoid_range_bounds(self):
        x = torch.tensor([0.0])
        result = sigmoid_range(x, -1.0, 1.0)
        # sigmoid(0) = 0.5, so result should be 0.5*(1-(-1)) + (-1) = 0
        assert abs(result.item()) < 1e-5

    def test_sigmoid_range_low_high(self):
        # Very negative input -> sigmoid approaches 0 -> result approaches low
        x = torch.tensor([-100.0])
        result = sigmoid_range(x, 2.0, 5.0)
        assert abs(result.item() - 2.0) < 1e-3

        # Very positive input -> sigmoid approaches 1 -> result approaches high
        x = torch.tensor([100.0])
        result = sigmoid_range(x, 2.0, 5.0)
        assert abs(result.item() - 5.0) < 1e-3

    def test_sigmoid_range_module(self):
        layer = SigmoidRange(low=-2.0, high=2.0)
        x = torch.tensor([0.0])
        result = layer(x)
        assert abs(result.item()) < 1e-5

    def test_sigmoid_range_batch(self):
        layer = SigmoidRange(low=0.0, high=10.0)
        x = torch.randn(8)
        result = layer(x)
        assert result.shape == (8,)
        # All values should be in (0, 10)
        assert torch.all(result > 0)
        assert torch.all(result < 10)


# ============================================================
# Tests for AdaptiveConcatPool
# ============================================================

class TestAdaptiveConcatPool:
    """Tests for AdaptiveConcatPool layers."""

    def test_adaptive_concat_pool_2d(self):
        layer = AdaptiveConcatPool2d(size=1)
        x = torch.randn(2, 16, 8, 8)
        result = layer(x)
        # Output channels should be doubled (max + avg)
        assert result.shape == (2, 32, 1, 1)

    def test_adaptive_concat_pool_1d(self):
        layer = AdaptiveConcatPool1d(size=1)
        x = torch.randn(2, 16, 100)
        result = layer(x)
        assert result.shape == (2, 32, 1)

    def test_adaptive_pool_avg(self):
        pool_cls = adaptive_pool(PoolType.Avg)
        assert pool_cls == nn.AdaptiveAvgPool2d

    def test_adaptive_pool_max(self):
        pool_cls = adaptive_pool(PoolType.Max)
        assert pool_cls == nn.AdaptiveMaxPool2d

    def test_adaptive_pool_cat(self):
        pool_cls = adaptive_pool(PoolType.Cat)
        assert pool_cls == AdaptiveConcatPool2d


# ============================================================
# Tests for BatchNorm
# ============================================================

class TestBatchNorm:
    """Tests for BatchNorm and InstanceNorm factory functions."""

    def test_batch_norm_2d(self):
        bn = BatchNorm(16, ndim=2)
        assert isinstance(bn, nn.BatchNorm2d)

    def test_batch_norm_1d(self):
        bn = BatchNorm(16, ndim=1)
        assert isinstance(bn, nn.BatchNorm1d)

    def test_batch_norm_zero(self):
        bn = BatchNorm(16, ndim=2, norm_type=NormType.BatchZero)
        # Weight should be initialized to 0
        assert torch.all(bn.weight == 0.0)

    def test_instance_norm(self):
        inn = InstanceNorm(16, ndim=2)
        assert isinstance(inn, nn.InstanceNorm2d)

    def test_batch_norm_1d_flat(self):
        bn = BatchNorm1dFlat(8)
        x = torch.randn(4, 3, 8)  # 3D input
        result = bn(x)
        assert result.shape == (4, 3, 8)


# ============================================================
# Tests for LinBnDrop
# ============================================================

class TestLinBnDrop:
    """Tests for the LinBnDrop module."""

    def test_basic(self):
        layer = LinBnDrop(10, 5)
        x = torch.randn(4, 10)
        result = layer(x)
        assert result.shape == (4, 5)

    def test_with_dropout(self):
        layer = LinBnDrop(10, 5, p=0.5)
        x = torch.randn(4, 10)
        result = layer(x)
        assert result.shape == (4, 5)

    def test_no_bn(self):
        layer = LinBnDrop(10, 5, bn=False)
        x = torch.randn(4, 10)
        result = layer(x)
        assert result.shape == (4, 5)

    def test_with_activation(self):
        layer = LinBnDrop(10, 5, act=nn.ReLU())
        x = torch.randn(4, 10)
        result = layer(x)
        assert result.shape == (4, 5)


# ============================================================
# Tests for ConvLayer
# ============================================================

class TestConvLayer:
    """Tests for the ConvLayer module."""

    def test_basic_conv(self):
        layer = ConvLayer(3, 16, ks=3)
        x = torch.randn(1, 3, 32, 32)
        result = layer(x)
        # With padding=1 (default for ks=3), spatial dims preserved
        assert result.shape == (1, 16, 32, 32)

    def test_conv_with_stride(self):
        layer = ConvLayer(3, 16, ks=3, stride=2)
        x = torch.randn(1, 3, 32, 32)
        result = layer(x)
        assert result.shape == (1, 16, 16, 16)

    def test_conv_1d(self):
        layer = ConvLayer(3, 16, ks=3, ndim=1)
        x = torch.randn(1, 3, 100)
        result = layer(x)
        assert result.shape == (1, 16, 100)

    def test_conv_no_activation(self):
        layer = ConvLayer(3, 16, ks=3, act_cls=None)
        x = torch.randn(1, 3, 32, 32)
        result = layer(x)
        assert result.shape == (1, 16, 32, 32)

    def test_conv_spectral_norm(self):
        layer = ConvLayer(3, 16, ks=3, norm_type=NormType.Spectral)
        x = torch.randn(1, 3, 32, 32)
        result = layer(x)
        assert result.shape == (1, 16, 32, 32)


# ============================================================
# Tests for Pool layers
# ============================================================

class TestPoolLayers:
    """Tests for AdaptiveAvgPool, MaxPool, AvgPool."""

    def test_adaptive_avg_pool(self):
        pool = AdaptiveAvgPool(sz=1, ndim=2)
        x = torch.randn(2, 16, 8, 8)
        result = pool(x)
        assert result.shape == (2, 16, 1, 1)

    def test_max_pool(self):
        pool = MaxPool(ks=2, ndim=2)
        x = torch.randn(2, 16, 8, 8)
        result = pool(x)
        assert result.shape == (2, 16, 4, 4)

    def test_avg_pool(self):
        pool = AvgPool(ks=2, ndim=2)
        x = torch.randn(2, 16, 8, 8)
        result = pool(x)
        assert result.shape == (2, 16, 4, 4)

    def test_pool_flatten(self):
        layer = PoolFlatten()
        x = torch.randn(2, 16, 8, 8)
        result = layer(x)
        assert result.shape == (2, 16)


# ============================================================
# Tests for Embedding with truncated normal init
# ============================================================

class TestEmbedding:
    """Tests for the custom Embedding layer."""

    def test_shape(self):
        emb = Embedding(100, 32)
        x = torch.tensor([0, 5, 99])
        result = emb(x)
        assert result.shape == (3, 32)

    def test_truncated_normal_init(self):
        emb = Embedding(1000, 64, std=0.01)
        # With truncated normal at std=0.01, values should be small
        assert emb.weight.data.abs().max() < 0.5


# ============================================================
# Tests for sigmoid and vleaky_relu
# ============================================================

class TestActivations:
    """Tests for custom activation functions."""

    def test_sigmoid_clamped(self):
        x = torch.tensor([-100.0, 0.0, 100.0])
        result = sigmoid(x)
        # Should be clamped to (eps, 1-eps)
        assert result[0] > 0
        assert result[2] < 1
        assert abs(result[1].item() - 0.5) < 1e-5

    def test_sigmoid_inplace_clamped(self):
        x = torch.tensor([-100.0, 0.0, 100.0])
        result = sigmoid_(x)
        assert result[0] > 0
        assert result[2] < 1

    def test_vleaky_relu(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        result = vleaky_relu(x, inplace=False)
        # negative slope is 0.3
        assert abs(result[0].item() - (-0.3)) < 1e-5
        assert result[1].item() == 0.0
        assert result[2].item() == 1.0


# ============================================================
# Tests for init_default
# ============================================================

class TestInitDefault:
    """Tests for the init_default function."""

    def test_init_default_zeros_bias(self):
        m = nn.Linear(10, 5)
        m.bias.data.fill_(1.0)  # set bias to non-zero
        init_default(m)
        assert torch.all(m.bias == 0.0)


# ============================================================
# Tests for ParameterModule
# ============================================================

class TestParameterModule:
    """Tests for ParameterModule."""

    def test_creation(self):
        p = nn.Parameter(torch.randn(3, 4))
        pm = ParameterModule(p)
        assert isinstance(pm, nn.Module)

    def test_forward_passthrough(self):
        """ParameterModule.forward just passes the input through."""
        p = nn.Parameter(torch.randn(3, 4))
        pm = ParameterModule(p)
        x = torch.randn(2, 5)
        result = pm(x)
        assert torch.equal(result, x)

    def test_stores_val(self):
        p = nn.Parameter(torch.randn(3, 4))
        pm = ParameterModule(p)
        assert hasattr(pm, 'val')
        assert torch.equal(pm.val, p)


# ============================================================
# Tests for model utility functions
# ============================================================

class TestModelUtils:
    """Tests for children_and_parameters, has_children, flatten_model."""

    def test_has_children_true(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        assert has_children(model) is True

    def test_has_children_false(self):
        layer = nn.Linear(10, 5)
        assert has_children(layer) is False

    def test_flatten_model(self):
        model = nn.Sequential(
            nn.Sequential(nn.Linear(10, 5), nn.ReLU()),
            nn.Linear(5, 2)
        )
        flat = flatten_model(model)
        assert len(flat) == 3  # Linear, ReLU, Linear

    def test_children_and_parameters(self):
        model = nn.Sequential(nn.Linear(10, 5))
        cp = children_and_parameters(model)
        # Should return children (the Linear layer)
        assert len(cp) >= 1


# ============================================================
# Tests for trunc_normal_
# ============================================================

class TestTruncNormal:
    """Tests for truncated normal initialization."""

    def test_trunc_normal_values(self):
        x = torch.empty(1000)
        trunc_normal_(x, mean=0.0, std=1.0)
        # Values should be bounded by fmod(2) -> max magnitude ~2
        assert x.abs().max() <= 2.0 + 1e-5

    def test_trunc_normal_mean(self):
        x = torch.empty(10000)
        trunc_normal_(x, mean=5.0, std=0.1)
        # Mean should be approximately 5.0
        assert abs(x.mean().item() - 5.0) < 0.1
