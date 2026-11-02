from types import SimpleNamespace

import numpy as np


def make_rf_shim(shim_vector) -> SimpleNamespace:
    """Create a dimensionless per-channel RF_SHIMS vector for one RF block.

    The complex weights multiply the block's scalar RF waveform. Magnitudes
    and phases in radians use the existing MATLAB/C++ RF_SHIMS file format.
    """
    shim_vector = np.asarray(shim_vector, dtype=complex)
    if shim_vector.ndim != 1 or shim_vector.size == 0 or not np.all(np.isfinite(shim_vector)):
        raise ValueError('shim_vector must be a finite, nonempty complex vector.')
    return SimpleNamespace(type='rf_shim', shim_vector=shim_vector.copy())
