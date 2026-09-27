"""Causality, EDF provenance and recursion checks for fitted marginals."""

import numpy as np
import pytest

from src.marginals import fit_marginal


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(451)
    returns = np.empty(350)
    variance = 0.00012
    error = 0.0
    previous = 0.0
    for i in range(len(returns)):
        variance = 0.00001 + 0.08 * error**2 + 0.85 * variance
        error = np.sqrt(variance) * rng.standard_normal()
        returns[i] = 0.0002 + 0.15 * previous + error
        previous = returns[i]
    return fit_marginal(returns[:250]), returns[250:]


def test_future_change_does_not_change_prefix(fitted):
    model, future = fitted
    perturbed = future.copy()
    perturbed[40:] = -10 * perturbed[40:] + 0.08
    original_pit = model.transform_future(future)
    perturbed_pit = model.transform_future(perturbed)
    np.testing.assert_array_equal(original_pit[:40], perturbed_pit[:40])
    assert np.any(original_pit[40:] != perturbed_pit[40:])


def test_future_filter_uses_terminal_formation_state(fitted):
    model, future = fitted
    standardized, errors, variances = model.filter_future(future[:2])
    first_variance = model.omega + model.alpha * model.last_error**2 + model.beta * model.last_variance
    first_mean = model.mu + model.phi * (model.last_return - model.mu) + model.theta * model.last_error
    assert variances[0] == pytest.approx(first_variance)
    assert errors[0] == pytest.approx(100 * future[0] - first_mean)
    assert standardized[0] == pytest.approx(errors[0] / np.sqrt(first_variance))
    assert variances[1] == pytest.approx(model.omega + model.alpha * errors[0]**2 + model.beta * first_variance)


def test_cdf_stays_frozen_and_excludes_boundaries(fitted):
    model, _ = fitted
    n = len(model.train_residuals)
    expected = np.searchsorted(np.sort(model.train_residuals), model.train_residuals, side="right") / (n + 1)
    np.testing.assert_array_equal(model.train_pit, expected)
    extremes = model.transform_future(np.array([-100.0, 100.0, -100.0, 100.0]))
    assert (extremes >= 1 / (n + 1)).all()
    assert (extremes <= n / (n + 1)).all()
    assert extremes[0] == pytest.approx(1 / (n + 1))
    np.testing.assert_array_equal(model.train_pit, expected)


def test_transform_does_not_mutate_model(fitted):
    model, future = fitted
    before = model.to_dict()
    first = model.transform_future(future)
    model.transform_future(future * 3)
    np.testing.assert_array_equal(first, model.transform_future(future))
    assert before == model.to_dict()


def test_fitted_constraints_and_diagnostics(fitted):
    model, _ = fitted
    assert model.omega > 0
    assert model.alpha >= 0 and model.beta >= 0
    assert model.alpha + model.beta < 0.999
    assert abs(model.phi) <= 0.98 and abs(model.theta) <= 0.98
    assert model.converged
    diagnostics = model.to_dict()
    assert len(diagnostics["candidates"]) == 4
    eligible = [candidate["aic"] for candidate in diagnostics["candidates"] if candidate["converged"]]
    assert model.aic == min(eligible)


def test_empty_future_and_invalid_inputs(fitted):
    model, _ = fitted
    assert model.transform_future(np.array([])).shape == (0,)
    with pytest.raises(ValueError, match="finite"):
        model.transform_future(np.array([np.nan]))
    with pytest.raises(ValueError, match="30"):
        fit_marginal(np.zeros(10))
    with pytest.raises(ValueError, match="variation"):
        fit_marginal(np.zeros(50))
