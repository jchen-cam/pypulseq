"""RF_SHIMS uses the existing MATLAB/C++ extension representation."""

import numpy as np
import pypulseq as pp
import pytest


@pytest.mark.parametrize('v141_compat', [False, True])
@pytest.mark.parametrize('use_block_cache', [False, True])
def test_rf_shim_round_trip(tmp_path, v141_compat, use_block_cache):
    weights = np.array([1, 0.5j, -0.25, 0.2 - 0.3j, 0, 1, 1j, -1j])
    rf = pp.make_block_pulse(np.pi / 6, duration=1e-3, use='excitation')
    seq = pp.Sequence(use_block_cache=use_block_cache)
    seq.add_block(rf, pp.make_rf_shim(weights), pp.make_label('REP', 'SET', 1))
    seq.add_block(rf, pp.make_rf_shim(weights))
    seq.add_block(pp.make_delay(1e-3))
    seq.set_definition('RequiredExtensions', 'RF_SHIMS')
    seq.set_definition('TxChannels', 8)

    assert len(seq.rf_shim_library.data) == 1
    assert pp.calc_duration(seq.get_block(1)) == pp.calc_duration(rf)
    assert seq.check_timing()[0]
    np.testing.assert_allclose(seq.get_block(1).rf_shim.shim_vector, weights, atol=1e-15)
    assert seq.get_block(3).rf_shim is None

    # Default export deep-copies and deduplicates event libraries.
    path = tmp_path / 'rf_shim.seq'
    seq.write(path, v141_compat=v141_compat)
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('extension RF_SHIMS '))
    wire = np.array([float(value) for value in lines[header + 1].split()])
    assert wire[1] == len(weights)
    np.testing.assert_allclose(wire[2::2], np.abs(weights), rtol=1e-8)
    np.testing.assert_allclose(wire[3::2], np.angle(weights), rtol=1e-8)

    restored = pp.Sequence(use_block_cache=use_block_cache)
    restored.read(path)
    assert restored.get_definition('RequiredExtensions') == 'RF_SHIMS'
    assert restored.get_block(1).label[0].value == 1
    assert restored.check_timing()[0]
    np.testing.assert_allclose(restored.get_block(1).rf_shim.shim_vector, weights, atol=1e-8)
    np.testing.assert_allclose(restored.get_block(2).rf_shim.shim_vector, weights, atol=1e-8)
    assert restored.get_block(3).rf_shim is None


def test_rf_shim_requires_one_rf_block():
    shim = pp.make_rf_shim([1, 1j])
    with pytest.raises(ValueError, match='requires an RF event'):
        pp.Sequence().add_block(shim)
    rf = pp.make_block_pulse(np.pi / 6, duration=1e-3, use='excitation')
    with pytest.raises(ValueError, match='Only one RF_SHIMS'):
        pp.Sequence().add_block(rf, shim, shim)


@pytest.mark.parametrize('weights', [[], [[1, 1j]], [np.nan], [np.inf]])
def test_rf_shim_invalid_vector(weights):
    with pytest.raises(ValueError, match='finite, nonempty complex vector'):
        pp.make_rf_shim(weights)
