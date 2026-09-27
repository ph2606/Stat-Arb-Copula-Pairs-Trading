"""Formation-only ARMA--GARCH marginals for the copula experiment.

Inputs are decimal log returns; optimization uses percentage log returns.
Four ARMA orders, p,q in {0,1}, are estimated jointly with Gaussian-QMLE
GARCH(1,1), and AIC chooses the marginal order. The paper specifies these
orders but leaves the innovation likelihood and selection criterion open.
Future transforms use frozen coefficients and the formation empirical CDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import warnings

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError:  # Correct, slower fallback for environments without numba.
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate


@njit(cache=True)
def _recursion(values, mu, phi, theta, omega, alpha, beta, initial_variance,
               last_return, last_error, last_variance, future):
    """Conditional variance at t never uses the error observed at t."""
    n = len(values)
    errors = np.empty(n)
    variances = np.empty(n)
    standardized = np.empty(n)
    objective = 0.0
    for i in range(n):
        conditional_mean = mu + phi * (last_return - mu) + theta * last_error
        if i == 0 and not future:
            variance = initial_variance
        else:
            variance = omega + alpha * last_error * last_error + beta * last_variance
        variance = max(variance, 1.0e-14)
        error = values[i] - conditional_mean
        errors[i] = error
        variances[i] = variance
        standardized[i] = error / math.sqrt(variance)
        objective += 0.5 * (math.log(2.0 * math.pi) + math.log(variance)
                            + error * error / variance)
        last_return = values[i]
        last_error = error
        last_variance = variance
    return objective, errors, variances, standardized


def _unpack(parameters: np.ndarray, p: int, q: int) -> tuple[float, ...]:
    cursor = 1
    mu = float(parameters[0])
    phi = float(parameters[cursor]) if p else 0.0
    cursor += p
    theta = float(parameters[cursor]) if q else 0.0
    cursor += q
    omega, alpha, beta = (float(x) for x in parameters[cursor:cursor + 3])
    return mu, phi, theta, omega, alpha, beta


def _validate_returns(values, *, training: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Returns must be a one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Marginal returns must be finite; no implicit gap filling.")
    if training and len(values) < 30:
        raise ValueError("At least 30 formation returns are required.")
    return values


@dataclass
class MarginalModel:
    p: int
    q: int
    mu: float
    phi: float
    theta: float
    omega: float
    alpha: float
    beta: float
    initial_variance: float
    last_return: float
    last_error: float
    last_variance: float
    train_residuals: np.ndarray
    sorted_residuals: np.ndarray
    log_likelihood: float
    aic: float
    converged: bool
    optimizer_message: str
    candidates: tuple[dict[str, Any], ...]

    @property
    def train_pit(self) -> np.ndarray:
        """Empirical-CDF ranks/(n+1), with the rightmost rank for ties."""
        return self._empirical_transform(self.train_residuals)

    def _empirical_transform(self, standardized: np.ndarray) -> np.ndarray:
        n = len(self.sorted_residuals)
        ranks = np.searchsorted(self.sorted_residuals, standardized, side="right")
        return np.clip(ranks / (n + 1.0), 1.0 / (n + 1.0), n / (n + 1.0))

    def filter_future(self, test_returns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return standardized errors, raw errors and variances (percent units).

        Every call starts at the unchanged formation terminal state. This
        makes the object safe to reuse for different scenarios and threads.
        Observations update the recursion, never the fitted coefficients.
        """
        test = _validate_returns(test_returns, training=False) * 100.0
        _, errors, variances, standardized = _recursion(
            test, self.mu, self.phi, self.theta, self.omega, self.alpha,
            self.beta, self.initial_variance, self.last_return, self.last_error,
            self.last_variance, True,
        )
        return standardized, errors, variances

    def transform_future(self, test_returns) -> np.ndarray:
        """Map causally filtered future returns through the frozen train EDF."""
        standardized, _, _ = self.filter_future(test_returns)
        return self._empirical_transform(standardized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": f"ARMA({self.p},{self.q})-GARCH(1,1)",
            "p": self.p,
            "q": self.q,
            "innovation_estimation": "Gaussian quasi-maximum likelihood",
            "selection": "AIC over p,q in {0,1}",
            "return_input": "decimal log returns; internally multiplied by 100",
            "mean_parameterization": "mu + phi*(previous_return-mu) + theta*previous_error",
            "mu_percent": self.mu,
            "phi": self.phi,
            "theta": self.theta,
            "omega_percent_squared": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.alpha + self.beta,
            "formation_count": len(self.train_residuals),
            "variance_initialization": "formation sample variance (ddof=1)",
            "initial_variance_percent_squared": self.initial_variance,
            "log_likelihood_percent_scale": self.log_likelihood,
            "aic": self.aic,
            "converged": self.converged,
            "optimizer_message": self.optimizer_message,
            "cdf": "formation EDF right ranks/(n+1), clipped to [1/(n+1),n/(n+1)]",
            "candidates": [dict(candidate) for candidate in self.candidates],
        }


def fit_marginal(train_returns) -> MarginalModel:
    """Estimate all four marginal orders using only the supplied formation data.

    The AR coefficient is bounded by 0.98 in absolute value; the MA
    coefficient obeys the same invertibility bound. GARCH has positive omega,
    nonnegative alpha/beta and alpha+beta <= 0.998999. Three deterministic
    initializations reduce sensitivity to local optima. Only converged fits
    can win AIC; failure of all candidates raises instead of silently trading.
    """
    values = _validate_returns(train_returns, training=True).copy() * 100.0
    initial_variance = float(np.var(values, ddof=1))
    if initial_variance <= 1.0e-12:
        raise ValueError("Formation returns have insufficient variation for GARCH.")
    mean = float(np.mean(values))
    sd = math.sqrt(initial_variance)
    centered = values - mean
    denominator = float(centered[:-1] @ centered[:-1])
    initial_phi = (float(centered[1:] @ centered[:-1]) / denominator
                   if denominator > 0 else 0.0)
    initial_phi = float(np.clip(initial_phi, -0.8, 0.8))
    variance_lower = max(initial_variance * 1.0e-8, 1.0e-12)
    candidates = []
    fits = []

    for p, q in ((0, 0), (1, 0), (0, 1), (1, 1)):
        bounds = [(mean - 10 * sd, mean + 10 * sd)]
        bounds += [(-0.98, 0.98)] * (p + q)
        bounds += [(variance_lower, max(initial_variance * 10, 1.0e-4)),
                   (0.0, 0.998999), (0.0, 0.998999)]

        def objective(parameters):
            mu, phi, theta, omega, alpha, beta = _unpack(parameters, p, q)
            result = _recursion(
                values, mu, phi, theta, omega, alpha, beta, initial_variance,
                mu, 0.0, initial_variance, False,
            )[0]
            return result if np.isfinite(result) else 1.0e100

        constraint = {"type": "ineq", "fun": lambda x: 0.998999 - x[-2] - x[-1]}
        solutions = []
        for initial_alpha, initial_beta in ((0.05, 0.90), (0.10, 0.65), (0.02, 0.20)):
            start = [mean]
            if p:
                start.append(initial_phi)
            if q:
                start.append(0.0)
            start += [initial_variance * (1 - initial_alpha - initial_beta),
                      initial_alpha, initial_beta]
            with warnings.catch_warnings():
                # SLSQP briefly steps outside bounds and clips before evaluating.
                warnings.filterwarnings("ignore", message="Values in x were outside bounds")
                result = minimize(objective, np.asarray(start), method="SLSQP",
                                  bounds=bounds, constraints=[constraint],
                                  options={"maxiter": 450, "ftol": 1.0e-8})
            feasible = (np.isfinite(result.fun) and np.isfinite(result.x).all()
                        and result.x[-2] >= -1.0e-10 and result.x[-1] >= -1.0e-10
                        and result.x[-2] + result.x[-1] < 0.999)
            if feasible:
                solutions.append(result)
        converged = [result for result in solutions if result.success]
        eligible = converged or solutions
        if not eligible:
            candidates.append({"p": p, "q": q, "converged": False,
                               "aic": None, "message": "No finite feasible fit"})
            continue
        best = min(eligible, key=lambda result: result.fun)
        aic = float(2 * len(best.x) + 2 * best.fun)
        candidates.append({"p": p, "q": q, "converged": bool(best.success),
                           "aic": aic, "log_likelihood": float(-best.fun),
                           "message": str(best.message)})
        if best.success:
            fits.append((aic, p, q, best))

    if not fits:
        raise RuntimeError(f"No ARMA-GARCH candidate converged: {candidates}")
    aic, p, q, best = min(fits, key=lambda item: item[0])
    mu, phi, theta, omega, alpha, beta = _unpack(best.x, p, q)
    nll, errors, variances, standardized = _recursion(
        values, mu, phi, theta, omega, alpha, beta, initial_variance,
        mu, 0.0, initial_variance, False,
    )
    standardized.setflags(write=False)
    sorted_residuals = np.sort(standardized)
    sorted_residuals.setflags(write=False)
    return MarginalModel(
        p=p, q=q, mu=mu, phi=phi, theta=theta, omega=omega, alpha=alpha,
        beta=beta, initial_variance=initial_variance, last_return=float(values[-1]),
        last_error=float(errors[-1]), last_variance=float(variances[-1]),
        train_residuals=standardized, sorted_residuals=sorted_residuals,
        log_likelihood=float(-nll), aic=aic, converged=bool(best.success),
        optimizer_message=str(best.message), candidates=tuple(candidates),
    )
