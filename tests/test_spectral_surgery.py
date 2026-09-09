from __future__ import annotations

import pytest
import torch

from riftlora.lowrank import CompactSVD
from riftlora.scale.spectral_surgery import (
    SpectralSurgeryConfig,
    reweight_compact_spectra,
)


def _identity_update(rank: int) -> CompactSVD:
    identity = torch.eye(rank)
    return CompactSVD(identity, torch.ones(rank), identity)


def test_abs_select_preserves_directions_and_l1_spectral_mass() -> None:
    update = _identity_update(5)
    edited = reweight_compact_spectra(
        {"layer": update},
        {"layer": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])},
        config=SpectralSurgeryConfig(policy="abs_select"),
    )["layer"]

    torch.testing.assert_close(edited.u, update.u)
    torch.testing.assert_close(edited.v, update.v)
    assert edited.s.sum().item() == pytest.approx(update.s.sum().item())
    assert edited.s[0] > edited.s[2] > edited.s[-1]
    assert edited.rank == update.rank


def test_smooth_abs_is_identity_for_degenerate_sensitivity() -> None:
    update = _identity_update(4)
    edited = reweight_compact_spectra(
        {"layer": update},
        {"layer": torch.ones(4)},
        config=SpectralSurgeryConfig(policy="smooth_abs"),
    )["layer"]

    torch.testing.assert_close(edited.dense(), update.dense())


def test_spectral_surgery_rejects_negative_sensitivity() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reweight_compact_spectra(
            {"layer": _identity_update(2)},
            {"layer": torch.tensor([1.0, -1.0])},
            config=SpectralSurgeryConfig(),
        )
