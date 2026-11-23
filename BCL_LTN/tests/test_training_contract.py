import numpy as np
import torch
from torch.utils.data import Dataset

from BCL_AUDIT.cifar100lt.losses import LogitCompensatedCE
from BCL_LTN.train_noisy_bcl import NoisyBCLDataset


class FakeCleanBase(Dataset):
    def __init__(self):
        self.indices = np.asarray([11, 17, 23], dtype=np.int64)
        self.targets = np.arange(30, dtype=np.int64) % 5
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, position):
        # The base implementation has a clean label, but the adapter returns
        # only the replacement observed label to the training loop.
        return (torch.zeros(3, 3, 32, 32),), int(self.targets[self.indices[position]])


def test_training_adapter_exposes_no_clean_truth_or_mask():
    base = FakeCleanBase()
    observed = np.asarray([3, 1, 4], dtype=np.int64)
    base.targets[base.indices] = observed
    wrapped = NoisyBCLDataset(base)
    item = wrapped[0]
    assert len(item) == 3
    views, observed_label, source_index = item
    assert len(views) == 1
    assert observed_label == 3
    assert source_index == 11
    assert not isinstance(item, dict)
    assert all(not isinstance(value, np.ndarray) for value in item)


def test_lc_prior_is_built_from_observed_noisy_counts():
    observed_counts = np.asarray([10, 20, 30], dtype=np.int64)
    criterion = LogitCompensatedCE(observed_counts)
    expected = torch.log(torch.tensor(observed_counts, dtype=torch.float32) /
                         float(observed_counts.sum()))
    assert torch.allclose(criterion.log_prior.cpu(), expected)
    clean_counts = np.asarray([30, 20, 10], dtype=np.int64)
    clean_criterion = LogitCompensatedCE(clean_counts)
    assert not torch.allclose(criterion.log_prior.cpu(), clean_criterion.log_prior.cpu())
