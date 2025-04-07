"""Test nn.signal."""
import numpy as np
import torch
import scipy.signal as sig
from audlib.nn.signal import hilbert


def test_hilbert():
    """Test Hilbert transform."""
    nums = np.random.rand(32, 10)  # a batch
    ndft = 16
    # Test a single example
    signp = nums[0]
    sigtc = torch.from_numpy(signp)
    hilbnp = sig.hilbert(signp, ndft)
    hilbtc = hilbert(sigtc, ndft).numpy()
    # PyTorch 2.0 returns complex tensor directly
    assert np.allclose(hilbnp, hilbtc)
    assert np.allclose(signp, hilbtc[:len(signp)].real)
    # Test a batch
    signp = nums
    sigtc = torch.from_numpy(signp)
    hilbnp = sig.hilbert(signp)
    hilbtc = hilbert(sigtc).numpy()
    assert np.allclose(hilbnp, hilbtc)
    assert np.allclose(signp, hilbtc[:, :len(signp[0])].real)


if __name__ == "__main__":
    test_hilbert()
