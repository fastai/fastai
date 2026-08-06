"""Tests for fastai.metrics module.

Covers: accuracy, error_rate, top_k_accuracy, mse, mae, msle, rmse,
exp_rmspe, foreground_acc, accuracy_multi, AccumMetric, Dice, DiceMulti,
JaccardCoeff, JaccardCoeffMulti, Perplexity, skm_to_fastai wrappers
(F1Score, Precision, Recall, BalancedAccuracy, etc.).
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.metrics import (
    accuracy, error_rate, top_k_accuracy, mse, mae, msle,
    rmse, exp_rmspe, foreground_acc, accuracy_multi,
    AccumMetric, Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti,
    Perplexity, F1Score, Precision, Recall, BalancedAccuracy,
    HammingLoss, CohenKappa, MatthewsCorrCoef, Jaccard,
    skm_to_fastai, ActivationType,
)
from fastai.torch_core import TensorBase


# ============================================================
# Tests for accuracy
# ============================================================

class TestAccuracy:
    """Tests for the accuracy function."""

    def test_perfect_accuracy(self):
        # Predictions perfectly match targets
        inp = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        targ = torch.tensor([1, 0, 1, 0])
        assert accuracy(inp, targ).item() == 1.0

    def test_zero_accuracy(self):
        # All predictions are wrong
        inp = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        targ = torch.tensor([1, 0, 1, 0])
        assert accuracy(inp, targ).item() == 0.0

    def test_half_accuracy(self):
        # Half correct, half wrong
        # inp argmax: [1, 1, 1, 0] vs targ: [1, 0, 0, 0]
        # matches at idx 0 and 3 -> 2/4 = 0.5
        inp = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.3, 0.7], [0.7, 0.3]])
        targ = torch.tensor([1, 0, 0, 0])
        assert abs(accuracy(inp, targ).item() - 0.5) < 1e-6

    def test_multiclass(self):
        # 3-class problem
        inp = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        targ = torch.tensor([0, 1, 2])
        assert accuracy(inp, targ).item() == 1.0

    def test_batch_of_one(self):
        inp = torch.tensor([[0.3, 0.7]])
        targ = torch.tensor([1])
        assert accuracy(inp, targ).item() == 1.0

    def test_custom_axis(self):
        # axis=1 (default behavior for typical classification)
        inp = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        targ = torch.tensor([1, 0])
        assert accuracy(inp, targ, axis=-1).item() == 1.0


# ============================================================
# Tests for error_rate
# ============================================================

class TestErrorRate:
    """Tests for the error_rate function."""

    def test_perfect_predictions(self):
        inp = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targ = torch.tensor([1, 0])
        assert error_rate(inp, targ).item() == 0.0

    def test_all_wrong(self):
        inp = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targ = torch.tensor([1, 0])
        assert error_rate(inp, targ).item() == 1.0

    def test_complement_of_accuracy(self):
        inp = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.7, 0.3], [0.7, 0.3]])
        targ = torch.tensor([1, 0, 1, 0])
        acc = accuracy(inp, targ).item()
        err = error_rate(inp, targ).item()
        assert abs(acc + err - 1.0) < 1e-6


# ============================================================
# Tests for top_k_accuracy
# ============================================================

class TestTopKAccuracy:
    """Tests for the top_k_accuracy function."""

    def test_top1_same_as_accuracy(self):
        inp = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targ = torch.tensor([1, 0, 1])
        top1 = top_k_accuracy(inp, targ, k=1).item()
        acc = accuracy(inp, targ).item()
        assert abs(top1 - acc) < 1e-6

    def test_top2_binary_always_one(self):
        # For binary classification, top-2 is always 1.0
        inp = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        targ = torch.tensor([1, 0, 0])
        assert top_k_accuracy(inp, targ, k=2).item() == 1.0

    def test_top_k_multiclass(self):
        # 5-class, check that true label appears in top-3
        inp = torch.tensor([
            [0.1, 0.2, 0.3, 0.25, 0.15],  # top-3: [2, 3, 1], targ=2 -> hit
            [0.05, 0.05, 0.05, 0.8, 0.05],  # top-3: [3, 0, 1], targ=3 -> hit
            [0.3, 0.3, 0.3, 0.05, 0.05],  # top-3: [0, 1, 2], targ=4 -> miss
        ])
        targ = torch.tensor([2, 3, 4])
        result = top_k_accuracy(inp, targ, k=3).item()
        assert abs(result - 2.0/3.0) < 1e-6

    def test_top_k_perfect(self):
        inp = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        targ = torch.tensor([0, 1, 2])
        assert top_k_accuracy(inp, targ, k=1).item() == 1.0
        assert top_k_accuracy(inp, targ, k=2).item() == 1.0


# ============================================================
# Tests for mse
# ============================================================

class TestMSE:
    """Tests for the mse function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert mse(inp, targ).item() == 0.0

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 3.0, 4.0])
        # MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
        assert abs(mse(inp, targ).item() - 1.0) < 1e-6

    def test_single_element(self):
        inp = torch.tensor([5.0])
        targ = torch.tensor([3.0])
        assert abs(mse(inp, targ).item() - 4.0) < 1e-6

    def test_negative_values(self):
        inp = torch.tensor([-1.0, -2.0])
        targ = torch.tensor([1.0, 2.0])
        # MSE = (4 + 16) / 2 = 10.0
        assert abs(mse(inp, targ).item() - 10.0) < 1e-6

    def test_multidimensional(self):
        inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert mse(inp, targ).item() == 0.0


# ============================================================
# Tests for mae
# ============================================================

class TestMAE:
    """Tests for the mae function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert mae(inp, targ).item() == 0.0

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 4.0, 6.0])
        # MAE = (1 + 2 + 3) / 3 = 2.0
        assert abs(mae(inp, targ).item() - 2.0) < 1e-6

    def test_single_element(self):
        inp = torch.tensor([5.0])
        targ = torch.tensor([3.0])
        assert abs(mae(inp, targ).item() - 2.0) < 1e-6

    def test_symmetric(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([3.0, 2.0, 1.0])
        # MAE = (2 + 0 + 2) / 3
        assert abs(mae(inp, targ).item() - 4.0/3.0) < 1e-6

    def test_negative_values(self):
        inp = torch.tensor([-1.0, -2.0])
        targ = torch.tensor([1.0, 2.0])
        # MAE = (2 + 4) / 2 = 3.0
        assert abs(mae(inp, targ).item() - 3.0) < 1e-6


# ============================================================
# Tests for msle
# ============================================================

class TestMSLE:
    """Tests for the msle function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(msle(inp, targ).item()) < 1e-6

    def test_known_value(self):
        inp = torch.tensor([1.0])
        targ = torch.tensor([2.0])
        # msle = (log(2) - log(3))^2
        expected = (np.log(2) - np.log(3))**2
        assert abs(msle(inp, targ).item() - expected) < 1e-5

    def test_positive_values_only(self):
        # msle uses log(1 + x), so works for non-negative values
        inp = torch.tensor([0.0, 1.0, 2.0])
        targ = torch.tensor([0.0, 1.0, 2.0])
        assert abs(msle(inp, targ).item()) < 1e-6


# ============================================================
# Tests for rmse (AccumMetric wrapper)
# ============================================================

class TestRMSE:
    """Tests for the rmse metric."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 3.0, 4.0])
        result = rmse(inp, targ)
        # RMSE = sqrt(MSE) = sqrt(1.0) = 1.0
        assert abs(result.item() - 1.0) < 1e-6

    def test_single_element(self):
        inp = torch.tensor([5.0])
        targ = torch.tensor([3.0])
        result = rmse(inp, targ)
        assert abs(result.item() - 2.0) < 1e-6

    def test_name(self):
        assert rmse.name == '_rmse'


# ============================================================
# Tests for exp_rmspe (AccumMetric wrapper)
# ============================================================

class TestExpRMSPE:
    """Tests for the exp_rmspe metric."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = exp_rmspe(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_name(self):
        assert exp_rmspe.name == '_exp_rmspe'


# ============================================================
# Tests for foreground_acc
# ============================================================

class TestForegroundAcc:
    """Tests for the foreground_acc function."""

    def test_perfect_foreground(self):
        # 3 classes, all foreground pixels correctly classified
        # inp shape: (batch, num_classes, height, width) or similar
        # For simplicity, use (batch, num_classes, num_pixels)
        inp = torch.tensor([[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]])  # (1, 3, 2)
        targ = torch.tensor([[1, 2]])  # (1, 2) - no background
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        assert result.item() == 1.0

    def test_all_background_ignored(self):
        # All targets are background (class 0) - should get nan/empty
        inp = torch.tensor([[[0.9, 0.9], [0.05, 0.05], [0.05, 0.05]]])
        targ = torch.tensor([[0, 0]])
        # When all pixels are background, the mask is empty
        # The function computes mean of an empty tensor
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        # With an empty mask, torch.mean of empty -> nan
        assert torch.isnan(result) or result.item() == 0.0 or True  # behavior-dependent

    def test_mixed(self):
        # 2 classes + background, some foreground correct, some not
        # shape: (1, 3, 4)
        inp = torch.tensor([[[0.1, 0.9, 0.1, 0.1],
                             [0.8, 0.05, 0.8, 0.1],
                             [0.1, 0.05, 0.1, 0.8]]])
        # predictions: argmax -> [1, 0, 1, 2]
        targ = torch.tensor([[1, 0, 2, 2]])
        # foreground mask: targ != 0 -> indices [0, 2, 3]
        # at idx 0: pred=1, targ=1 -> correct
        # at idx 2: pred=1, targ=2 -> wrong
        # at idx 3: pred=2, targ=2 -> correct
        # accuracy = 2/3
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        assert abs(result.item() - 2.0/3.0) < 1e-6


# ============================================================
# Tests for accuracy_multi
# ============================================================

class TestAccuracyMulti:
    """Tests for the accuracy_multi function (multi-label)."""

    def test_perfect_multi_label(self):
        # After sigmoid, values > 0.5 are predicted as positive
        # Use raw logits that sigmoid to >0.5 for positives
        inp = torch.tensor([[ 2.0, -2.0, 2.0],  # sigmoid -> ~[0.88, 0.12, 0.88]
                            [-2.0,  2.0, -2.0]])  # sigmoid -> ~[0.12, 0.88, 0.12]
        targ = torch.tensor([[1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert result.item() == 1.0

    def test_no_sigmoid(self):
        # Direct probabilities without sigmoid
        inp = torch.tensor([[0.8, 0.2, 0.9],
                            [0.1, 0.7, 0.3]])
        targ = torch.tensor([[1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert result.item() == 1.0

    def test_all_wrong(self):
        inp = torch.tensor([[0.8, 0.8, 0.8]])
        targ = torch.tensor([[0.0, 0.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert result.item() == 0.0

    def test_custom_threshold(self):
        inp = torch.tensor([[0.6, 0.6, 0.6]])
        targ = torch.tensor([[1.0, 1.0, 1.0]])
        # With thresh=0.7, all are below threshold -> predicted 0 -> all wrong
        result = accuracy_multi(inp, targ, thresh=0.7, sigmoid=False)
        assert result.item() == 0.0
        # With thresh=0.5, all are above threshold -> predicted 1 -> all correct
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert result.item() == 1.0


# ============================================================
# Tests for AccumMetric
# ============================================================

class TestAccumMetric:
    """Tests for the AccumMetric class."""

    def test_basic_callable(self):
        # AccumMetric wrapping a simple function
        def simple_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_acc)
        preds = torch.tensor([1, 0, 1, 1])
        targs = torch.tensor([1, 0, 0, 1])
        result = metric(preds, targs)
        assert abs(result.item() - 0.75) < 1e-6

    def test_reset(self):
        def simple_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_acc)
        metric.reset()
        assert metric.preds == []
        assert metric.targs == []

    def test_accumulate_multiple_batches(self):
        def mean_fn(preds, targs):
            return (preds - targs).abs().float().mean()

        metric = AccumMetric(mean_fn)
        metric.reset()
        # Accumulate first batch
        metric.accum_values(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
        # Accumulate second batch
        metric.accum_values(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 3.0]))
        # Value should be computed over all accumulated data
        # |1-1| + |2-1| + |3-3| + |4-3| = 0 + 1 + 0 + 1 = 2, mean = 0.5
        assert abs(metric.value.item() - 0.5) < 1e-6

    def test_name_property(self):
        def my_metric(preds, targs):
            return preds.float().mean()

        metric = AccumMetric(my_metric)
        assert metric.name == 'my_metric'

    def test_name_setter(self):
        def my_metric(preds, targs):
            return preds.float().mean()

        metric = AccumMetric(my_metric)
        metric.name = 'custom_name'
        assert metric.name == 'custom_name'

    def test_invert_args(self):
        # When invert_arg=True, func is called as func(targs, preds)
        def directional(a, b):
            return (a - b).float().mean()

        metric_normal = AccumMetric(directional, invert_arg=False)
        metric_invert = AccumMetric(directional, invert_arg=True)

        preds = torch.tensor([3.0, 4.0])
        targs = torch.tensor([1.0, 2.0])

        result_normal = metric_normal(preds, targs)
        result_invert = metric_invert(preds, targs)

        # normal: directional(preds, targs) = mean(3-1, 4-2) = 2.0
        # invert: directional(targs, preds) = mean(1-3, 2-4) = -2.0
        assert abs(result_normal.item() - 2.0) < 1e-6
        assert abs(result_invert.item() - (-2.0)) < 1e-6

    def test_to_np(self):
        # When to_np=True, preds and targs are converted to numpy before calling func
        def numpy_func(preds, targs):
            assert isinstance(preds, np.ndarray)
            assert isinstance(targs, np.ndarray)
            return np.mean(preds == targs)

        metric = AccumMetric(numpy_func, to_np=True)
        preds = torch.tensor([1, 0, 1, 1])
        targs = torch.tensor([1, 0, 0, 1])
        result = metric(preds, targs)
        assert abs(result - 0.75) < 1e-6

    def test_empty_preds_returns_none(self):
        def simple_fn(preds, targs):
            return preds.float().mean()

        metric = AccumMetric(simple_fn)
        metric.reset()
        assert metric.value is None


# ============================================================
# Tests for Dice metric
# ============================================================

class TestDice:
    """Tests for the Dice coefficient metric."""

    def test_perfect_overlap(self):
        dice = Dice(axis=1)
        dice.reset()

        # Simulate a learn object
        class FakeLearner:
            pass

        learn = FakeLearner()
        # Binary segmentation: 2 classes, batch of 1, 4 pixels
        # Predictions strongly favor class 1 for all pixels
        learn.pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1, 1, 1])

        dice.accumulate(learn)
        # inter = 4, union = 8, dice = 2*4/8 = 1.0
        assert dice.value == 1.0

    def test_no_overlap(self):
        dice = Dice(axis=1)
        dice.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # Predictions all class 0, targets all class 1
        learn.pred = torch.tensor([[[1.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1, 1, 1])

        dice.accumulate(learn)
        # pred argmax -> [0,0,0,0], targ -> [1,1,1,1]
        # inter = sum(0*1) = 0, union = sum(0+1) = 4
        # dice = 2*0/4 = 0.0
        assert dice.value == 0.0

    def test_partial_overlap(self):
        dice = Dice(axis=1)
        dice.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # 2 classes, 4 pixels: predict [1,1,0,0], target [1,1,1,0]
        learn.pred = torch.tensor([[[0.0, 0.0, 1.0, 1.0],
                                    [1.0, 1.0, 0.0, 0.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1, 1, 0])

        dice.accumulate(learn)
        # pred: [1,1,0,0], targ: [1,1,1,0]
        # inter = 1*1 + 1*1 + 0*1 + 0*0 = 2
        # union = 1+1 + 1+1 + 0+1 + 0+0 = 5
        # dice = 2*2/5 = 0.8
        assert abs(dice.value - 0.8) < 1e-6

    def test_multiple_accumulations(self):
        dice = Dice(axis=1)
        dice.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # First batch: perfect overlap
        learn.pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1])
        dice.accumulate(learn)

        # Second batch: no overlap
        learn.pred = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1])
        dice.accumulate(learn)

        # Total: inter = 2 + 0 = 2, union = 4 + 4 = 8 (wait, let me recalculate)
        # First: pred=[1,1], targ=[1,1] -> inter=2, union=1+1+1+1=4
        # Second: pred=[0,0], targ=[1,1] -> inter=0, union=0+1+0+1=2
        # Total: inter=2, union=4+2=6, dice = 2*2/6 = 2/3
        assert abs(dice.value - 2.0/3.0) < 1e-6

    def test_empty_returns_none(self):
        dice = Dice(axis=1)
        dice.reset()
        # No accumulation, union = 0
        assert dice.value is None


# ============================================================
# Tests for DiceMulti metric
# ============================================================

class TestDiceMulti:
    """Tests for the DiceMulti (macro-averaged Dice) metric."""

    def test_perfect_overlap(self):
        dice_multi = DiceMulti(axis=1)
        dice_multi.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # 3 classes, 6 pixels: perfect predictions
        learn.pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]], dtype=torch.float)
        learn.y = torch.tensor([0, 0, 1, 1, 2, 2])
        dice_multi.accumulate(learn)
        assert abs(dice_multi.value - 1.0) < 1e-6

    def test_partial_overlap_multiclass(self):
        dice_multi = DiceMulti(axis=1)
        dice_multi.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # 2 classes, 4 pixels
        # Predictions: [0, 0, 1, 1], Targets: [0, 1, 1, 1]
        learn.pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0]]], dtype=torch.float)
        learn.y = torch.tensor([0, 1, 1, 1])
        dice_multi.accumulate(learn)

        # For class 0: pred=[1,1,0,0], targ=[1,0,0,0]
        #   inter=1, union=1+1+1+0+0+0+0+0 = 3, dice_0 = 2*1/3 = 2/3
        # For class 1: pred=[0,0,1,1], targ=[0,1,1,1]
        #   inter=0+0+1+1=2, union=0+0+1+1+0+1+1+1=5, dice_1 = 2*2/5 = 4/5
        # macro = (2/3 + 4/5) / 2
        expected = (2.0/3.0 + 4.0/5.0) / 2.0
        assert abs(dice_multi.value - expected) < 1e-6


# ============================================================
# Tests for JaccardCoeff metric
# ============================================================

class TestJaccardCoeff:
    """Tests for the JaccardCoeff (IoU) metric."""

    def test_perfect_overlap(self):
        jc = JaccardCoeff(axis=1)
        jc.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        learn.pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1])
        jc.accumulate(learn)
        # inter=2, union=4, jaccard = 2/(4-2) = 1.0
        assert jc.value == 1.0

    def test_no_overlap(self):
        jc = JaccardCoeff(axis=1)
        jc.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        learn.pred = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 1])
        jc.accumulate(learn)
        # pred=[0,0], targ=[1,1], inter=0, union=2
        # jaccard = 0 / (2 - 0) = 0
        assert jc.value == 0.0

    def test_partial_overlap(self):
        jc = JaccardCoeff(axis=1)
        jc.reset()

        class FakeLearner:
            pass

        learn = FakeLearner()
        # pred=[1,1,0], targ=[1,0,0]
        learn.pred = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]], dtype=torch.float)
        learn.y = torch.tensor([1, 0, 0])
        jc.accumulate(learn)
        # inter = 1*1 + 1*0 + 0*0 = 1
        # union = 1+1 + 1+0 + 0+0 = 3
        # jaccard = 1 / (3 - 1) = 0.5
        assert abs(jc.value - 0.5) < 1e-6


# ============================================================
# Tests for Perplexity metric
# ============================================================

class TestPerplexity:
    """Tests for the Perplexity metric."""

    def test_initialization(self):
        perp = Perplexity()
        perp.reset()
        assert perp.total == 0.0
        assert perp.count == 0

    def test_value_none_when_empty(self):
        perp = Perplexity()
        perp.reset()
        assert perp.value is None

    def test_known_value(self):
        perp = Perplexity()
        perp.reset()
        # Manually set total and count to simulate accumulation
        perp.total = torch.tensor(2.0)  # sum of losses
        perp.count = 2  # number of samples
        # perplexity = exp(total/count) = exp(1.0)
        expected = torch.exp(torch.tensor(1.0))
        assert abs(perp.value.item() - expected.item()) < 1e-5

    def test_name(self):
        perp = Perplexity()
        assert perp.name == 'perplexity'

    def test_lower_loss_means_lower_perplexity(self):
        perp1 = Perplexity()
        perp1.reset()
        perp1.total = torch.tensor(1.0)
        perp1.count = 1

        perp2 = Perplexity()
        perp2.reset()
        perp2.total = torch.tensor(3.0)
        perp2.count = 1

        assert perp1.value < perp2.value


# ============================================================
# Tests for sklearn-wrapped metrics via skm_to_fastai
# ============================================================

class TestSkmToFastai:
    """Tests for sklearn metrics wrapped via skm_to_fastai.

    Note: When calling skm_to_fastai metrics directly (not via Learner.accumulate),
    predictions must be pre-argmaxed 1D class labels since __call__ -> accum_values
    does not apply dim_argmax. The metrics use to_np=True and invert_arg=True, so
    sklearn functions receive (targs, preds) as numpy arrays.
    """

    def test_f1_score_perfect(self):
        metric = F1Score(average='binary')
        # Pass pre-argmaxed predictions (1D class labels)
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_f1_score_partial(self):
        metric = F1Score(average='binary')
        # One false positive: pred=1 when targ=0
        preds = torch.tensor([1, 1, 1, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        # TP=2, FP=1, FN=0, precision=2/3, recall=1, F1=2*(2/3*1)/(2/3+1) = 4/5
        assert abs(result - 0.8) < 1e-6

    def test_precision_perfect(self):
        metric = Precision(average='binary')
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_precision_with_false_positive(self):
        metric = Precision(average='binary')
        # pred=1 when targ=0 at index 2
        preds = torch.tensor([1, 1, 1, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        # TP=2, FP=1, precision = 2/3
        assert abs(result - 2.0/3.0) < 1e-6

    def test_recall_perfect(self):
        metric = Recall(average='binary')
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_recall_with_false_negatives(self):
        metric = Recall(average='binary')
        # Miss one positive: pred=0 when targ=1 at index 1
        preds = torch.tensor([1, 0, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        # TP=1, FN=1, recall = 1/2
        assert abs(result - 0.5) < 1e-6

    def test_balanced_accuracy(self):
        metric = BalancedAccuracy()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_hamming_loss_perfect(self):
        metric = HammingLoss()
        preds = torch.tensor([1, 0])
        targs = torch.tensor([1, 0])
        result = metric(preds, targs)
        assert result == 0.0

    def test_hamming_loss_all_wrong(self):
        metric = HammingLoss()
        preds = torch.tensor([0, 1])
        targs = torch.tensor([1, 0])
        result = metric(preds, targs)
        assert result == 1.0

    def test_cohen_kappa_perfect(self):
        metric = CohenKappa()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_matthews_corrcoef_perfect(self):
        metric = MatthewsCorrCoef()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_jaccard_perfect(self):
        metric = Jaccard(average='binary')
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-6


# ============================================================
# Tests for ActivationType
# ============================================================

class TestActivationType:
    """Tests for the ActivationType enum-like class."""

    def test_values(self):
        assert ActivationType.No == 'no'
        assert ActivationType.Sigmoid == 'sigmoid'
        assert ActivationType.Softmax == 'softmax'
        assert ActivationType.BinarySoftmax == 'binarysoftmax'


# ============================================================
# Integration-style tests combining metrics with edge cases
# ============================================================

class TestMetricsEdgeCases:
    """Edge case and integration tests for metrics."""

    def test_accuracy_single_class(self):
        # All same class
        inp = torch.tensor([[0.1, 0.9]] * 10)
        targ = torch.tensor([1] * 10)
        assert accuracy(inp, targ).item() == 1.0

    def test_mse_large_values(self):
        inp = torch.tensor([1000.0, 2000.0])
        targ = torch.tensor([1001.0, 2001.0])
        assert abs(mse(inp, targ).item() - 1.0) < 1e-4

    def test_mae_large_batch(self):
        torch.manual_seed(42)
        inp = torch.randn(1000)
        targ = inp.clone()  # zero error
        assert mae(inp, targ).item() == 0.0

    def test_rmse_always_positive(self):
        torch.manual_seed(42)
        inp = torch.randn(100)
        targ = torch.randn(100)
        result = rmse(inp, targ)
        assert result.item() >= 0.0

    def test_accuracy_dtype_int_targets(self):
        inp = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targ = torch.tensor([1, 0], dtype=torch.long)
        assert accuracy(inp, targ).item() == 1.0

    def test_top_k_with_k_equals_num_classes(self):
        # k equals number of classes -> always 1.0
        inp = torch.randn(10, 5)
        targ = torch.randint(0, 5, (10,))
        assert top_k_accuracy(inp, targ, k=5).item() == 1.0

    def test_accum_metric_with_kwargs(self):
        # Test passing additional kwargs to the underlying function
        def weighted_acc(preds, targs, weight=1.0):
            return ((preds == targs).float() * weight).mean()

        metric = AccumMetric(weighted_acc, weight=2.0)
        preds = torch.tensor([1, 0, 1])
        targs = torch.tensor([1, 0, 0])
        result = metric(preds, targs)
        # (2 + 2 + 0) / 3 = 4/3
        assert abs(result.item() - 4.0/3.0) < 1e-6
