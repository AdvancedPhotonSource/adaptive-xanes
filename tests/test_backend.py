"""Consumer contracts and boundary cases for the standalone backend."""

import numpy as np
import pytest
import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from adaptive_xanes import (
    AdaptiveXANESBayesianOptimization,
    ComprehensiveAugmentedAcquisitionFunction,
    to_numpy,
    to_tensor,
)
from adaptive_xanes._gp import OutcomeUnstandardizePosteriorTransform
from regression_cases import make_engine, spectrum


@pytest.fixture
def guide():
    engine = make_engine(
        AdaptiveXANESBayesianOptimization,
        ComprehensiveAugmentedAcquisitionFunction,
        "spline_no_reference",
    )
    energies = np.linspace(8950, 9100, 11).reshape(-1, 1)
    engine.update(energies, spectrum(energies))
    engine.build()
    return engine


def test_haven_suggest_update_contract(guide):
    """Haven's ask/tell and intensity-pair tell_many conversions remain usable."""
    candidate = guide.suggest(n_suggestions=1)
    assert candidate.shape == (1, 1)
    assert not candidate.requires_grad
    assert 8950 <= candidate.item() <= 9100
    energy = list(np.atleast_1d(np.squeeze(to_numpy(candidate))))
    guide.update(to_tensor([[float(energy[0])]]), to_tensor([[0.6]]))
    assert guide.xs_untransformed.shape == (12, 1)
    assert guide.ys_untransformed[-1].item() == pytest.approx(0.6)

    energies = [[8981.0], [9063.0]]
    intensities = np.array([[100.0, 70.0], [110.0, 30.0]])
    mu = -np.log(intensities[:, 1] / intensities[:, 0])
    old_lengthscale = guide.model.covar_module.lengthscale.detach().clone()
    old_mean = guide.outcome_transform.means.clone()
    guide.update(to_tensor(energies).reshape(-1, 1), to_tensor(mu).reshape(-1, 1))
    assert guide.xs_untransformed.shape == (14, 1)
    np.testing.assert_allclose(to_numpy(guide.ys_untransformed[-2:, 0]), mu)
    assert guide.get_num_adaptive_updates_completed() == 2
    torch.testing.assert_close(guide.model.covar_module.lengthscale, old_lengthscale)
    torch.testing.assert_close(guide.outcome_transform.means, old_mean)


def test_initial_observations_can_arrive_in_batches():
    engine = make_engine(
        AdaptiveXANESBayesianOptimization,
        ComprehensiveAugmentedAcquisitionFunction,
        "spline_no_reference",
    )
    x = np.linspace(8950, 9100, 11).reshape(-1, 1)
    for batch in (x[:4], x[4:]):
        engine.update(batch, spectrum(batch))
    assert engine.get_num_adaptive_updates_completed() == 0
    engine.build()
    assert engine.get_num_adaptive_updates_completed() == 0
    assert engine.acquisition_function.weight_func is None
    np.testing.assert_array_equal(to_numpy(engine.xs_untransformed), x)


def test_build_resets_update_count_and_stop_reason(guide):
    x = np.array([[8995.0]])
    guide.update(x, spectrum(x))
    guide.stop_reason = "max_measurements_reached"
    guide.build()
    assert guide.get_num_adaptive_updates_completed() == 0
    assert guide.stop_reason is None
    assert guide.acquisition_function.weight_func is None
    assert len(guide.xs_untransformed) == 12


def test_duplicate_filter_and_exhausted_grid(guide):
    measured = guide.xs_transformed
    new_point = torch.tensor([[0.345]], dtype=torch.double)
    choices = torch.cat([measured, new_point])
    torch.testing.assert_close(guide.filter_duplicate_choices(choices), new_point)
    with pytest.warns(UserWarning, match="All discrete XANES candidates"):
        torch.testing.assert_close(guide.filter_duplicate_choices(measured), measured)


def test_duplicate_threshold_is_in_normalized_energy(guide):
    guide.duplicate_distance_threshold = 0.01
    choices = torch.tensor([[0.005], [0.015]], dtype=torch.double)
    torch.testing.assert_close(guide.filter_duplicate_choices(choices), choices[1:])


def test_suggestion_excludes_measured_grid_points(guide):
    candidate = guide.transform_data(x=guide.suggest())[0]
    assert torch.all(
        torch.abs(guide.xs_transformed - candidate) >= guide.get_duplicate_distance_threshold()
    )


def test_measurement_cap_precedes_uncertainty_checks(guide):
    guide.n_max_measurements = len(guide.xs_untransformed)
    assert guide.get_num_adaptive_updates_completed() == 0
    assert guide.should_stop()
    assert guide.stop_reason == "max_measurements_reached"


def test_weighted_uncertainty_schedule_and_clipping(guide, monkeypatch):
    guide.n_max_measurements = None
    guide.stopping_n_updates_to_begin = 2
    guide.stopping_check_interval = 2
    guide.stopping_uncertainty_threshold = 0.3
    queries = []

    def posterior(x, **kwargs):
        queries.append(x)
        assert kwargs == {"transform": False, "untransform": True}
        return torch.zeros_like(x), torch.full_like(x, 0.2)

    monkeypatch.setattr(guide, "get_posterior_mean_and_std", posterior)
    guide.acquisition_function.set_weight_func(lambda x: torch.full_like(x, 3.0))
    for n_updates in (0, 1, 3):
        guide.n_adaptive_update_calls = n_updates
        assert not guide.should_stop()
    assert not queries
    guide.n_adaptive_update_calls = 2
    assert guide.should_stop()  # 0.2 * clip(3, 0, 1) < 0.3
    assert guide.stop_reason == "weighted_max_uncertainty_below_threshold"
    assert queries[-1].shape == (55, 1)
    assert queries[-1].min() == 0 and queries[-1].max() == 1
    guide.stopping_uncertainty_threshold = 0.2
    assert not guide.should_stop()  # The inequality is strict.
    assert guide.stop_reason is None
    guide.acquisition_function.set_weight_func(lambda x: torch.full_like(x, -3.0))
    assert guide.should_stop()
    guide.stopping_uncertainty_threshold = None
    assert not guide.should_stop()


def test_posterior_transform_reverses_mean_and_variance():
    posterior = GPyTorchPosterior(MultivariateNormal(
        torch.tensor([1.5], dtype=torch.double),
        torch.tensor([[0.25]], dtype=torch.double),
    ))
    transform = OutcomeUnstandardizePosteriorTransform(
        torch.tensor([2.0], dtype=torch.double),
        torch.tensor([4.0], dtype=torch.double),
    )
    result = transform(posterior, X=torch.tensor([[0.1]], dtype=torch.double))
    torch.testing.assert_close(result.mean, torch.tensor([[8.0]], dtype=torch.double))
    torch.testing.assert_close(result.variance, torch.tensor([[4.0]], dtype=torch.double))


@pytest.mark.parametrize("which", ("x", "y"))
def test_update_rejects_wrong_shape(which):
    guide = AdaptiveXANESBayesianOptimization(bounds=([8950.0], [9100.0]))
    x, y = np.ones((3, 1)), np.ones((3, 1))
    if which == "x":
        x = x.ravel()
    else:
        y = y.ravel()
    with pytest.raises(ValueError, match="Expected .* data of shape"):
        guide.update(x, y)
    assert guide.xs_untransformed.numel() == 0


def test_suggest_requires_build():
    guide = AdaptiveXANESBayesianOptimization(bounds=([8950.0], [9100.0]))
    with pytest.raises(RuntimeError, match="Acquisition function is not built"):
        guide.suggest()


@pytest.mark.parametrize("case", ("haven_spline", "spline_no_reference"))
def test_automatic_weights_with_default_float32(case):
    """Automatic-weight query grids must not cast the double GP to float32."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        engine = make_engine(
            AdaptiveXANESBayesianOptimization,
            ComprehensiveAugmentedAcquisitionFunction,
            case,
        )
        # Haven constructs the optional lengthscale override as float32.
        engine.kernel_lengthscales = torch.tensor([12.0], dtype=torch.float32)
        x = np.linspace(8950, 9100, 11).reshape(-1, 1)
        engine.update(x, spectrum(x))
        engine.build()
        for _ in range(6):
            candidate = engine.suggest()
            assert torch.isfinite(candidate).all()
            assert candidate.dtype == torch.double
            engine.update(candidate, spectrum(candidate.numpy()))
        assert engine.acquisition_function.weight_func is not None
        assert engine.model.train_inputs[0].dtype == torch.double
    finally:
        torch.set_default_dtype(previous)
