"""SIGNAL transform functions done in torch."""
import torch
import torch.nn.functional as F


def firfreqz(h, ndft, squared=False):
    """Compute frequency response of an FIR filter."""
    assert ndft > h.size(-1), "Incompatible DFT size!"
    h = F.pad(h, (0, ndft-h.size(-1)))
    # Update to PyTorch 2.0 FFT functions
    hspec = torch.fft.rfft(h, dim=-1)
    hspec = hspec.abs().pow(2)
    if squared:
        return hspec
    return hspec**.5


def iirfreqz(h, ndft, squared=False, powerfloor=10**-3):
    """Compute frequency response of an IIR filter."""
    assert ndft > h.size(-1), "Incompatible DFT size!"
    h = F.pad(h, (0, ndft-h.size(-1)))
    # Update to PyTorch 2.0 FFT functions
    hspec = torch.fft.rfft(h, dim=-1)
    hspec = hspec.abs().pow(2).clamp(min=powerfloor)
    if squared:
        return 1 / hspec
    return 1 / (hspec**.5)


def freqz(b, a, ndft, gain=1, iirfloor=10**-3, squared=False):
    """Compute the frequency response of a filter."""
    hhnum = firfreqz(b, ndft, squared)
    hhden = iirfreqz(a, ndft, squared, iirfloor)
    if squared:
        return hhnum * hhden * gain**2
    return hhnum*hhden * gain


def hilbert(x, ndft=None):
    r"""Analytic signal of x.

    Return the analytic signal of a real signal x, x + j\hat{x}, where \hat{x}
    is the Hilbert transform of x.

    Parameters
    ----------
    x: torch.Tensor
        Audio signal to be analyzed.
        Always assumes x is real, and x.shape[-1] is the signal length.

    Returns
    -------
    out: torch.Tensor
        Complex tensor containing the analytic signal

    """
    if ndft is None:
        sig = x
    else:
        assert ndft > x.size(-1)
        sig = F.pad(x, (0, ndft-x.size(-1)))
    
    # Update to PyTorch 2.0 FFT functions
    N = sig.size(-1)
    X = torch.fft.fft(sig, dim=-1)
    
    h = torch.ones(N, device=sig.device)
    if N % 2 == 0:
        h[1:N//2] = 2
        h[N//2] = 1
        h[N//2+1:] = 0
    else:
        h[1:(N+1)//2] = 2
        h[(N+1)//2:] = 0
        
    return torch.fft.ifft(X * h, dim=-1)
