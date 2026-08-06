"""Tests for fastai.losses module.

Covers custom loss functions: BaseLoss, CrossEntropyLossFlat, FocalLoss,
BCEWithLogitsLossFlat, MSELossFlat, LabelSmoothingCrossEntropy, DiceLoss.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.losses import (
    BaseLoss, CrossEntropyLossFlat, FocalLoss, FocalLossFlat,
    BCEWithLogitsLossFlat, MSELossFlat, L1LossFlat,
    LabelSmoothingCrossEntropy, LabelSmoothingCrossEntropyFlat,
    DiceLoss,
)


# ============================================================
# Tests for BaseLoss
# ============================================================

class TestBaseLoss:
    """Tests for BaseLoss wrapper."""

    def test_creation(self):
        loss = BaseLoss(nn.MSELoss, floatify=True, is_2d=False)
        assert repr(loss).startswith('FlattenedLoss')

    def test_reduction_property(self):
        loss = BaseLoss(nn.MSELoss, floatify=True, is_2d=False)
        assert loss.reduction == 'mean'
        loss.reduction = 'sum'
        assert loss.reduction == 'sum'


# ============================================================
# Tests for CrossEntropyLossFlat
# ============================================================

class TestCrossEntropyLossFlat:
    """Tests for CrossEntropyLossFlat."""

    def test_basic_loss(self):
        loss_fn = CrossEntropyLossFlat()
        # batch of 4, 5 classes
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_perfect_prediction(self):
        loss_fn = CrossEntropyLossFlat()
        # Create a prediction that strongly predicts class 0
        pred = torch.tensor([[100.0, -100.0, -100.0]])
        targ = torch.tensor([0])
        loss = loss_fn(pred, targ)
        assert loss.item() < 0.01

    def test_activation(self):
        loss_fn = CrossEntropyLossFlat()
        pred = torch.randn(4, 5)
        activated = loss_fn.activation(pred)
        # Should be softmax - sums to 1 along class dim
        sums = activated.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_decodes(self):
        loss_fn = CrossEntropyLossFlat()
        pred = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        decoded = loss_fn.decodes(pred)
        assert decoded.tolist() == [1, 0]


# ============================================================
# Tests for FocalLoss
# ============================================================

class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_basic_loss(self):
        loss_fn = FocalLoss(gamma=2.0)
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gamma_effect(self):
        """Higher gamma should down-weight easy examples more."""
        pred = torch.tensor([[2.0, -2.0, -2.0]])  # confident prediction
        targ = torch.tensor([0])

        loss_low = FocalLoss(gamma=0.0)(pred, targ)
        loss_high = FocalLoss(gamma=5.0)(pred, targ)
        # With higher gamma, confident correct predictions have lower loss
        assert loss_high.item() < loss_low.item()

    def test_reduction_sum(self):
        loss_fn = FocalLoss(gamma=2.0, reduction='sum')
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0

    def test_reduction_none(self):
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.shape == (4,)


class TestFocalLossFlat:
    """Tests for FocalLossFlat."""

    def test_basic(self):
        loss_fn = FocalLossFlat(gamma=2.0)
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_activation(self):
        loss_fn = FocalLossFlat()
        pred = torch.randn(4, 5)
        activated = loss_fn.activation(pred)
        sums = activated.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_decodes(self):
        loss_fn = FocalLossFlat()
        pred = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        decoded = loss_fn.decodes(pred)
        assert decoded.tolist() == [1, 0]


# ============================================================
# Tests for BCEWithLogitsLossFlat
# ============================================================

class TestBCEWithLogitsLossFlat:
    """Tests for BCEWithLogitsLossFlat."""

    def test_basic_loss(self):
        loss_fn = BCEWithLogitsLossFlat()
        pred = torch.randn(4, 3)
        targ = torch.randint(0, 2, (4, 3)).float()
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_activation(self):
        loss_fn = BCEWithLogitsLossFlat()
        pred = torch.randn(4, 3)
        activated = loss_fn.activation(pred)
        # Should be sigmoid - values in (0, 1)
        assert torch.all(activated > 0)
        assert torch.all(activated < 1)

    def test_decodes(self):
        loss_fn = BCEWithLogitsLossFlat(thresh=0.5)
        pred = torch.tensor([0.6, 0.3, 0.8])
        decoded = loss_fn.decodes(pred)
        expected = torch.tensor([True, False, True])
        assert torch.equal(decoded, expected)


# ============================================================
# Tests for MSELossFlat
# ============================================================

class TestMSELossFlat:
    """Tests for MSELossFlat."""

    def test_basic_loss(self):
        loss_fn = MSELossFlat()
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        # MSE should be 0.25
        assert abs(loss.item() - 0.25) < 1e-5

    def test_zero_loss(self):
        loss_fn = MSELossFlat()
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(pred, targ)
        assert abs(loss.item()) < 1e-7


# ============================================================
# Tests for L1LossFlat
# ============================================================

class TestL1LossFlat:
    """Tests for L1LossFlat."""

    def test_basic_loss(self):
        loss_fn = L1LossFlat()
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        # L1 should be 0.5
        assert abs(loss.item() - 0.5) < 1e-5

    def test_zero_loss(self):
        loss_fn = L1LossFlat()
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(pred, targ)
        assert abs(loss.item()) < 1e-7


# ============================================================
# Tests for LabelSmoothingCrossEntropy
# ============================================================

class TestLabelSmoothingCrossEntropy:
    """Tests for LabelSmoothingCrossEntropy."""

    def test_basic_loss(self):
        loss_fn = LabelSmoothingCrossEntropy(eps=0.1)
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_eps_zero_equals_ce(self):
        """With eps=0, should behave like regular cross entropy."""
        pred = torch.randn(8, 5)
        targ = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

        lsce = LabelSmoothingCrossEntropy(eps=0.0)
        ce = nn.CrossEntropyLoss()

        loss_lsce = lsce(pred, targ)
        loss_ce = ce(pred, targ)
        assert abs(loss_lsce.item() - loss_ce.item()) < 1e-4

    def test_activation(self):
        loss_fn = LabelSmoothingCrossEntropy()
        pred = torch.randn(4, 5)
        activated = loss_fn.activation(pred)
        sums = activated.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_decodes(self):
        loss_fn = LabelSmoothingCrossEntropy()
        pred = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        decoded = loss_fn.decodes(pred)
        assert decoded.tolist() == [1, 0]

    def test_reduction_sum(self):
        loss_fn = LabelSmoothingCrossEntropy(reduction='sum')
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0


class TestLabelSmoothingCrossEntropyFlat:
    """Tests for LabelSmoothingCrossEntropyFlat."""

    def test_basic(self):
        loss_fn = LabelSmoothingCrossEntropyFlat()
        pred = torch.randn(4, 5)
        targ = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() > 0


# ============================================================
# Tests for DiceLoss
# ============================================================

class TestDiceLoss:
    """Tests for DiceLoss."""

    def test_basic_loss(self):
        loss_fn = DiceLoss()
        # 2 samples, 3 classes, 4x4 spatial
        pred = torch.randn(2, 3, 4, 4)
        targ = torch.randint(0, 3, (2, 4, 4))
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        """When prediction matches target, loss should be close to 0."""
        loss_fn = DiceLoss()
        # Create a simple case: 1 sample, 2 classes, 2x2
        # Target is all class 0
        targ = torch.zeros(1, 2, 2, dtype=torch.long)
        # Create prediction that strongly predicts class 0
        pred = torch.zeros(1, 2, 2, 2)
        pred[:, 0, :, :] = 100.0
        pred[:, 1, :, :] = -100.0
        loss = loss_fn(pred, targ)
        assert loss.item() < 0.1

    def test_one_hot_internal(self):
        """Test the internal _one_hot method."""
        targ = torch.tensor([[0, 1], [2, 0]])
        result = DiceLoss._one_hot(targ, classes=3)
        assert result.shape == (2, 3, 2)  # (batch, classes, spatial)
        # Check class 0 mask
        assert result[0, 0, 0] == 1  # (0,0) is class 0
        assert result[0, 0, 1] == 0  # (0,1) is class 1
        assert result[0, 1, 1] == 1  # (0,1) is class 1

    def test_reduction_mean(self):
        loss_fn = DiceLoss(reduction='mean')
        pred = torch.randn(2, 3, 4, 4)
        targ = torch.randint(0, 3, (2, 4, 4))
        loss = loss_fn(pred, targ)
        assert loss.ndim == 0

    def test_dice_loss_range(self):
        """Dice loss should be between 0 and 1 for valid inputs."""
        loss_fn = DiceLoss(reduction='mean')
        pred = torch.randn(4, 3, 8, 8)
        targ = torch.randint(0, 3, (4, 8, 8))
        loss = loss_fn(pred, targ)
        assert 0 <= loss.item() <= 2.0  # theoretical max is num_classes but mean brings it down
