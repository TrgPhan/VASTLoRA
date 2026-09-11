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


def test_signed_update_uses_descent_sign_and_preserves_mass() -> None:
    update = _identity_update(2)
    result = reweight_compact_spectra(
        {"layer": update}, {"layer": torch.ones(2)},
        signed_sensitivities={"layer": torch.tensor([1.0, -1.0])},
        config=SpectralSurgeryConfig(policy="grad_direction", asymmetric_update=False, eta=0.2),
    )["layer"]
    expected = torch.tensor([-0.2, 0.2]).exp()
    expected *= 2 / expected.sum()
    torch.testing.assert_close(result.s, expected)
    assert result.s[1] > result.s[0]


def test_asymmetric_signed_update_matches_equation() -> None:
    edited = reweight_compact_spectra(
        {"layer": _identity_update(2)}, {"layer": torch.tensor([3., 1.])},
        signed_sensitivities={"layer": torch.tensor([3., -1.])},
        config=SpectralSurgeryConfig(policy="grad_direction", eta_suppress=0.2,
                                    eta_enhance=0.4, positive_power=2, preserve_energy="none"),
    )["layer"]
    torch.testing.assert_close(edited.s, torch.tensor([-0.2 * 1.5**2, 0.4 * 0.5]).exp())


def test_signed_policy_requires_actual_signed_gradients() -> None:
    with pytest.raises(ValueError, match="signed_sensitivities"):
        reweight_compact_spectra({"layer": _identity_update(2)}, {"layer": torch.ones(2)},
                                config=SpectralSurgeryConfig(policy="grad_direction"))


def test_random_control_reproducible_and_independent_of_sensitivity_and_global_rng() -> None:
    update = _identity_update(10)
    config = SpectralSurgeryConfig(policy="random_index", random_seed=19)
    rng = torch.random.get_rng_state().clone()
    first = reweight_compact_spectra({"layer": update}, {"layer": torch.arange(10.)}, config=config)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    second = reweight_compact_spectra({"layer": update}, {"layer": torch.arange(10.).flip(0)}, config=config)
    torch.testing.assert_close(first["layer"].s, second["layer"].s)
    hard = reweight_compact_spectra({"layer": update}, {"layer": torch.arange(10.)},
                                  config=SpectralSurgeryConfig(policy="abs_select"))
    torch.testing.assert_close(first["layer"].s.sort().values, hard["layer"].s.sort().values)


def test_smooth_align_mid_has_prescribed_value_at_center() -> None:
    edited = reweight_compact_spectra(
        {"layer": _identity_update(5)}, {"layer": torch.tensor([1., 2., 3., 4., 5.])},
        config=SpectralSurgeryConfig(smooth_align_mid=True, preserve_energy="none"),
    )["layer"]
    assert edited.s[2].item() == pytest.approx(1.0)


def test_invalid_midpoint_and_overflow_are_rejected() -> None:
    with pytest.raises(ValueError, match="smooth_align_mid"):
        SpectralSurgeryConfig(smooth_align_mid=True, mid_factor=3).validate()
    with pytest.raises(ValueError, match="non-finite edited"):
        reweight_compact_spectra(
            {"layer": _identity_update(2)}, {"layer": torch.ones(2)},
            signed_sensitivities={"layer": -torch.ones(2)},
            config=SpectralSurgeryConfig(policy="grad_direction", eta_enhance=1000),
        )
