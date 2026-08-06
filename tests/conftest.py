"""Shared test fixtures and configuration for fastai tests."""
import sys
import os
import pytest

# Ensure the fastai package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def cpu_device():
    """Return the CPU torch device for tests that need explicit device placement."""
    import torch
    return torch.device('cpu')
