"""Bivariate copulas for the mixed-copula pairs-trading replication.

All models are fitted by (unpenalized) maximum pseudo-likelihood to formation
pseudo-observations. Mixtures jointly estimate every component parameter and
two free simplex weights, retaining the exact single-component fits as feasible
candidates. Numerical multistart optimization is not a global-optimum proof.

Bounds are deliberate numerical conventions: |rho| <= .9999, Student-t df in
[2.01, 200] (finite variance), Clayton theta in [.001, 200], positive Frank theta
in [.001, 400], and Gumbel theta in [1, 100]. Positive Archimedean dependence is
appropriate to the three specified pairs; it is not a general negative-
dependence fitting implementation. Probabilities are clipped to [1e-10,1-1e-10]
when evaluated, avoiding undefined densities at exact empirical-CDF endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable
import warnings

import numpy as np
from scipy import optimize, special, stats

EPS = 1e-10
FAMILIES = ("Gaussian", "Student-t", "Clayton", "Frank", "Gumbel")
PARAM_NAMES = {
    "Gaussian": ("rho",), "Student-t": ("rho", "nu"),
    "Clayton": ("theta",), "Frank": ("theta",), "Gumbel": ("theta",),
}
PARAM_BOUNDS = {
    "Gaussian": ((-.9999, .9999),),
    "Student-t": ((-.9999, .9999), (2.01, 200.0)),
    "Clayton": ((.001, 200.0),),
    "Frank": ((.001, 400.0),),
    "Gumbel": ((1.0, 100.0),),
}


def _uv(u, v):
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)


class _Prepared:
    """Cache transforms within a fit, especially expensive t quantiles."""

    def __init__(self, u, v):
        self.u, self.v = _uv(u, v)
        self.lu, self.lv = np.log(self.u), np.log(self.v)
        self.lx, self.ly = np.log(-self.lu), np.log(-self.lv)
        self.z1, self.z2 = special.ndtri(self.u), special.ndtri(self.v)
        # An instance-owned cache avoids retaining formation datasets globally.
        self.t_quantiles = lru_cache(maxsize=16)(self._t_quantiles)

    def _t_quantiles(self, nu):
        return stats.t.ppf(self.u, nu), stats.t.ppf(self.v, nu)


def _frank_log_d(p, theta):
    # D = exp(-theta*u)+exp(-theta*v)-exp(-theta*(u+v))-exp(-theta).
    # This representation retains tiny D at very high positive dependence.
    pos = np.logaddexp(-theta * p.u, -theta * p.v)
    neg = np.logaddexp(-theta * (p.u + p.v), -theta)
    return pos + np.log(-np.expm1(np.minimum(neg - pos, -np.finfo(float).eps)))


def _logpdf(name, parameters, p):
    if name == "Gaussian":
        rho = parameters[0]
        det = 1 - rho * rho
        q = (rho * rho * (p.z1**2 + p.z2**2) - 2 * rho * p.z1 * p.z2) / det
        return -.5 * (np.log(det) + q)
    if name == "Student-t":
        rho, nu = parameters
        x, y = p.t_quantiles(float(nu))
        det = 1 - rho * rho
        q = (x * x - 2 * rho * x * y + y * y) / det
        biv = (special.gammaln((nu + 2) / 2) - special.gammaln(nu / 2)
               - np.log(nu * np.pi) - .5 * np.log(det)
               - (nu + 2) / 2 * np.log1p(q / nu))
        marginal_const = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - .5 * np.log(nu * np.pi)
        return biv - 2 * marginal_const + (nu + 1) / 2 * (np.log1p(x*x/nu) + np.log1p(y*y/nu))
    theta = parameters[0]
    if name == "Clayton":
        ls = np.logaddexp(-theta * p.lu, -theta * p.lv)
        ls += np.log1p(-np.exp(-ls))
        return np.log1p(theta) - (1 + theta) * (p.lu + p.lv) - (2 + 1/theta) * ls
    if name == "Frank":
        return (np.log(theta) + np.log(-np.expm1(-theta))
                - theta * (p.u + p.v) - 2 * _frank_log_d(p, theta))
    if name == "Gumbel":
        ls = np.logaddexp(theta * p.lx, theta * p.ly)
        q = np.exp(ls / theta)
        return (-q - p.lu - p.lv + (theta - 1) * (p.lx + p.ly)
                + (2/theta - 2) * ls + np.log1p((theta - 1) / q))
    raise ValueError(f"Unknown copula: {name}")


def _conditional(name, parameters, p):
    if name == "Gaussian":
        rho = parameters[0]
        sd = np.sqrt(1 - rho*rho)
        return special.ndtr((p.z1-rho*p.z2)/sd), special.ndtr((p.z2-rho*p.z1)/sd)
    if name == "Student-t":
        rho, nu = parameters
        x, y = p.t_quantiles(float(nu))
        first = (x-rho*y) * np.sqrt((nu+1)/((nu+y*y)*(1-rho*rho)))
        second = (y-rho*x) * np.sqrt((nu+1)/((nu+x*x)*(1-rho*rho)))
        return stats.t.cdf(first, nu+1), stats.t.cdf(second, nu+1)
    theta = parameters[0]
    if name == "Clayton":
        ls = np.logaddexp(-theta*p.lu, -theta*p.lv)
        ls += np.log1p(-np.exp(-ls))
        first = -(1+1/theta)*ls - (theta+1)*p.lv
        second = -(1+1/theta)*ls - (theta+1)*p.lu
    elif name == "Frank":
        ld = _frank_log_d(p, theta)
        first = -theta*p.v + np.log(-np.expm1(-theta*p.u)) - ld
        second = -theta*p.u + np.log(-np.expm1(-theta*p.v)) - ld
    elif name == "Gumbel":
        ls = np.logaddexp(theta*p.lx, theta*p.ly)
        common = -np.exp(ls/theta) + (1/theta-1)*ls
        first = common + (theta-1)*p.ly - p.lv
        second = common + (theta-1)*p.lx - p.lu
    else:
        raise ValueError(f"Unknown copula: {name}")
    return np.clip(np.exp(first), 0, 1), np.clip(np.exp(second), 0, 1)


@dataclass
class Copula:
    name: str
    parameters: tuple[float, ...] = ()
    components: tuple["Copula", ...] = ()
    weights: tuple[float, ...] = ()
    loglik: float = float("nan")
    n_obs: int = 0
    success: bool = True
    message: str = "Parameters supplied directly; not fitted."
    fit_details: dict = field(default_factory=dict)

    @property
    def n_params(self):
        return sum(c.n_params for c in self.components) + len(self.components)-1 if self.components else len(self.parameters)

    @property
    def params(self):
        return dict(zip(PARAM_NAMES.get(self.name, ()), map(float, self.parameters)))

    def logpdf(self, u, v):
        p = _Prepared(u, v)
        if not self.components:
            return _logpdf(self.name, self.parameters, p)
        values = np.stack([_logpdf(c.name, c.parameters, p) for c in self.components])
        with np.errstate(divide="ignore"):
            lw = np.log(np.asarray(self.weights))
        return special.logsumexp(values + lw.reshape((-1,) + (1,)*(values.ndim-1)), axis=0)

    def conditional(self, u, v):
        """Return (P[U<=u | V=v], P[V<=v | U=u]); never swap the legs."""
        p = _Prepared(u, v)
        if not self.components:
            return _conditional(self.name, self.parameters, p)
        values = [_conditional(c.name, c.parameters, p) for c in self.components]
        return tuple(np.clip(sum(w * hs[i] for w, hs in zip(self.weights, values)), 0, 1) for i in range(2))

    def cdf(self, u, v):
        """CDF, principally for mathematical validation of h-functions."""
        p = _Prepared(u, v)
        if self.components:
            return sum(w*c.cdf(u, v) for w, c in zip(self.weights, self.components))
        if self.name == "Gaussian":
            points = np.stack([p.z1, p.z2], axis=-1)
            rho = self.parameters[0]
            return stats.multivariate_normal.cdf(points, mean=[0, 0], cov=[[1, rho], [rho, 1]])
        if self.name == "Student-t":
            rho, nu = self.parameters
            points = np.stack(p.t_quantiles(float(nu)), axis=-1)
            return stats.multivariate_t.cdf(points, shape=[[1, rho], [rho, 1]], df=nu, random_state=0)
        theta = self.parameters[0]
        if self.name == "Clayton":
            ls = np.logaddexp(-theta*p.lu, -theta*p.lv)
            ls += np.log1p(-np.exp(-ls))
            return np.exp(-ls/theta)
        if self.name == "Frank":
            return -( _frank_log_d(p, theta) - np.log(-np.expm1(-theta))) / theta
        if self.name == "Gumbel":
            return np.exp(-np.exp(np.logaddexp(theta*p.lx, theta*p.ly)/theta))
        raise ValueError(self.name)

    def to_dict(self):
        result = {
            "name": self.name, "params": self.params, "loglik": float(self.loglik),
            "n_params": self.n_params, "n_obs": self.n_obs,
            "success": bool(self.success), "message": self.message,
            "aic": float(2*self.n_params - 2*self.loglik),
            "bic": float(np.log(max(1, self.n_obs))*self.n_params - 2*self.loglik),
            "fit_details": self.fit_details,
        }
        if self.components:
            result["weights"] = {c.name: float(w) for c, w in zip(self.components, self.weights)}
            result["components"] = {c.name: c.params for c in self.components}
        return result


def _encode(name, parameters):
    values = np.asarray(parameters, dtype=float).copy()
    if name in ("Gaussian", "Student-t"):
        values[0] = np.arctanh(values[0])
        if name == "Student-t":
            values[1] = np.log(values[1])
    else:
        values = np.log(values)
    return values


def _decode(name, values):
    parameters = np.asarray(values, dtype=float).copy()
    if name in ("Gaussian", "Student-t"):
        parameters[0] = np.tanh(parameters[0])
        if name == "Student-t":
            parameters[1] = np.exp(parameters[1])
    else:
        parameters = np.exp(parameters)
    return tuple(map(float, parameters))


def _bounds(name):
    raw = np.asarray(PARAM_BOUNDS[name])
    lo, hi = _encode(name, raw[:, 0]), _encode(name, raw[:, 1])
    return list(zip(lo, hi))


def _fit_single(name, p, tau):
    rho = float(np.clip(np.sin(np.pi*tau/2), -.999, .999))
    if name == "Gaussian":
        initial = [(rho,), (float(np.clip(np.corrcoef(p.z1, p.z2)[0, 1], -.999, .999)),)]
    elif name == "Student-t":
        initial = [(rho, nu) for nu in (4., 12., 60.)]
    elif name == "Clayton":
        theta = np.clip(2*max(tau, 0)/max(1-tau, .01), .001, 180.)
        initial = [(theta,), (max(.001, theta/3),)]
    elif name == "Frank":
        theta = np.clip(6*tau/max(1-tau, .02), .001, 350.)
        initial = [(theta,), (max(.001, theta/3),)]
    else:
        theta = np.clip(1/max(1-tau, .01), 1., 95.)
        initial = [(theta,), (1.2,)]

    def objective(values):
        ll = np.sum(_logpdf(name, _decode(name, values), p))
        return -float(ll) if np.isfinite(ll) else 1e100

    results = []
    for parameters in initial:
        result = optimize.minimize(objective, _encode(name, parameters), method="L-BFGS-B", bounds=_bounds(name),
                                   options={"maxiter": 180, "ftol": 1e-11, "gtol": 1e-5})
        if np.isfinite(result.fun) and result.fun < 1e99:
            results.append(result)
    if not results:
        raise RuntimeError(f"No finite fit for {name}")
    converged = [r for r in results if r.success]
    if not converged:
        raise RuntimeError(f'No converged fit for {name}: {[str(r.message) for r in results]}')
    best = min(converged, key=lambda r: r.fun)
    params = _decode(name, best.x)
    active = [PARAM_NAMES[name][i] for i, (value, (lo, hi)) in enumerate(zip(best.x, _bounds(name)))
              if min(abs(value-lo), abs(value-hi)) < 1e-5]
    return Copula(name, params, loglik=-float(best.fun), n_obs=p.u.size,
                  success=bool(best.success), message=str(best.message),
                  fit_details={"starts": len(initial), "successful_starts": sum(bool(r.success) for r in results),
                               "boundary_parameters": active, "bounds": PARAM_BOUNDS[name]})


def _fit_mixture(name, components, p):
    names = tuple(c.name for c in components)
    widths = [len(c.parameters) for c in components]
    offsets = np.cumsum([0]+widths)
    x_parameters = np.concatenate([_encode(c.name, c.parameters) for c in components])
    bounds = [bound for cname in names for bound in _bounds(cname)] + [(0., 1.), (0., 1.)]

    def unpack(values):
        params = [_decode(cname, values[offsets[i]:offsets[i+1]]) for i, cname in enumerate(names)]
        weights = np.array([values[-2], values[-1], 1-values[-2]-values[-1]])
        return params, weights

    def objective(values):
        params, weights = unpack(values)
        if np.min(weights) < -1e-7:
            return 1e6 + 1e6*abs(np.min(weights))
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        logs = np.stack([_logpdf(cname, par, p) for cname, par in zip(names, params)])
        with np.errstate(divide="ignore"):
            ll = special.logsumexp(logs + np.log(weights)[:, None], axis=0).sum()
        return -float(ll) if np.isfinite(ll) else 1e100

    # Preserve every exact vertex. In particular, a poor numerical mixture fit
    # can never reverse its theoretical nesting advantage in raw likelihood.
    candidates = []
    for index, component in enumerate(components):
        weights = np.eye(3)[index]
        x0 = np.r_[x_parameters, weights[:2]]
        candidates.append((component.loglik, x0, component.success, f"Retained nested {component.name} fit"))
    constraints = {"type": "ineq", "fun": lambda x: 1-x[-2]-x[-1],
                   "jac": lambda x: np.r_[np.zeros(len(x)-2), -1., -1.]}
    results = []
    for weights in ((1/3, 1/3), (1., 0.), (0., 1.), (0., 0.)):
        x0 = np.r_[x_parameters, weights]
        with warnings.catch_warnings():
            # SLSQP legitimately clips exploratory line-search steps; retain
            # other warnings and expose actual convergence in fit metadata.
            warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds", category=RuntimeWarning)
            result = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                                       options={"maxiter": 160, "ftol": 1e-7})
        results.append(result)
        if result.success and np.isfinite(result.fun) and min(unpack(result.x)[1]) >= -1e-7:
            candidates.append((-float(result.fun), result.x, bool(result.success), str(result.message)))
    ll, values, success, message = max(candidates, key=lambda candidate: candidate[0])
    parameters, weights = unpack(values)
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    fitted_components = tuple(Copula(cname, par) for cname, par in zip(names, parameters))
    model = Copula(name, components=fitted_components, weights=tuple(map(float, weights)),
                   loglik=ll, n_obs=p.u.size, success=success, message=message,
                   fit_details={"starts": 4, "successful_starts": sum(bool(r.success) for r in results),
                                "nested_best_loglik": max(c.loglik for c in components),
                                "selection": "Unpenalized joint maximum pseudo-likelihood; exact vertices retained"})
    # Report likelihood for precisely the returned normalized weights.
    model.loglik = float(model.logpdf(p.u, p.v).sum())
    return model


def fit_candidates(u: Iterable[float], v: Iterable[float]) -> dict[str, Copula]:
    """Fit the five single copulas and the CFG/CtG mixtures.

    Input must be aligned one-dimensional formation pseudo-observations. The
    marginal module uses right ranks divided by n+1 (a scaled EDF). This function never estimates
    marginals from test data and does not perform model selection with AIC/BIC.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if u.ndim != 1 or v.ndim != 1 or len(u) != len(v) or len(u) < 10:
        raise ValueError("Supply equal-length one-dimensional arrays with at least 10 observations")
    if not np.isfinite(u).all() or not np.isfinite(v).all() or min(u.min(), v.min()) < 0 or max(u.max(), v.max()) > 1:
        raise ValueError("Copula observations must be finite probabilities in [0, 1]")
    tau = stats.kendalltau(u, v).statistic
    if not np.isfinite(tau):
        raise ValueError("Copula formation series must both vary")
    p = _Prepared(u, v)
    fitted = {name: _fit_single(name, p, float(tau)) for name in FAMILIES}
    fitted["CFG"] = _fit_mixture("CFG", tuple(fitted[name] for name in ("Clayton", "Frank", "Gumbel")), p)
    fitted["CtG"] = _fit_mixture("CtG", tuple(fitted[name] for name in ("Clayton", "Student-t", "Gumbel")), p)
    return fitted
