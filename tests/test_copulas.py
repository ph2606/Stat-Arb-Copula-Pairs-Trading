"""Mathematical checks independent of the trading rules and data download."""

from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from copulas import Copula, fit_candidates


MODELS = (
    Copula("Gaussian", (.65,)), Copula("Student-t", (.65, 5.)),
    Copula("Clayton", (2.,)), Copula("Frank", (5.,)), Copula("Gumbel", (2.,)),
)


class CopulaMathematicsTests(unittest.TestCase):
    def test_archimedean_conditionals_are_cdf_derivatives(self):
        u = np.array([.12, .31, .52, .84])
        v = np.array([.64, .22, .73, .45])
        delta = 1e-6
        for model in MODELS[2:]:
            h1, h2 = model.conditional(u, v)
            assert_allclose(h1, (model.cdf(u, v+delta)-model.cdf(u, v-delta))/(2*delta), atol=1e-7)
            assert_allclose(h2, (model.cdf(u+delta, v)-model.cdf(u-delta, v))/(2*delta), atol=1e-7)

    def test_density_is_derivative_of_conditional(self):
        # All families, including elliptical copulas, get an independent local
        # derivative check without relying on stochastic multivariate-t CDFs.
        u = np.array([.08, .27, .58, .85])
        v = np.array([.18, .68, .52, .92])
        delta = 2e-6
        for model in MODELS:
            derivative = (model.conditional(u+delta, v)[0]-model.conditional(u-delta, v)[0])/(2*delta)
            assert_allclose(np.exp(model.logpdf(u, v)), derivative, rtol=2e-5, atol=2e-6)

    def test_densities_integrate_to_one_and_have_uniform_margins(self):
        # Gauss-Legendre quadrature avoids evaluating singular square corners.
        nodes, weights = np.polynomial.legendre.leggauss(180)
        nodes, weights = (nodes+1)/2, weights/2
        u, v = np.meshgrid(nodes, nodes)
        for model in MODELS:
            density = np.exp(model.logpdf(u, v))
            total = np.sum(density * weights[:, None] * weights[None, :])
            self.assertAlmostEqual(total, 1., delta=.003, msg=model.name)
            for fixed_v in (.2, .5, .8):
                margin = np.sum(np.exp(model.logpdf(nodes, fixed_v))*weights)
                self.assertAlmostEqual(margin, 1., delta=.001, msg=model.name)

    def test_leg_exchange_symmetry_and_probability_bounds(self):
        u = np.array([0., .05, .3, .7, .95, 1.])
        v = np.array([.4, .6, .2, .8, .9, .99])
        for model in MODELS:
            assert_allclose(model.logpdf(u, v), model.logpdf(v, u), rtol=1e-10, atol=1e-10)
            h1, h2 = model.conditional(u, v)
            swapped1, swapped2 = model.conditional(v, u)
            assert_allclose(h1, swapped2)
            assert_allclose(h2, swapped1)
            self.assertTrue(np.all((h1 >= 0) & (h1 <= 1)))
            self.assertTrue(np.all((h2 >= 0) & (h2 <= 1)))

    def test_independence_and_high_dependence_limits_are_finite(self):
        u, v = np.array([.1, .4, .8]), np.array([.7, .4, .2])
        for model in (Copula("Gaussian", (0.,)), Copula("Gumbel", (1.,))):
            assert_allclose(model.logpdf(u, v), 0., atol=1e-13)
            assert_allclose(model.conditional(u, v), (u, v), atol=1e-13)
        extreme = (Copula("Gaussian", (.9999,)), Copula("Student-t", (.9999, 2.01)),
                   Copula("Clayton", (200.,)), Copula("Frank", (400.,)), Copula("Gumbel", (100.,)))
        for model in extreme:
            u = np.array([0., .02, .5, .98, 1.])
            v = np.array([.01, .025, .49, .981, 1.])
            self.assertTrue(np.isfinite(model.logpdf(u, v)).all(), model.name)
            self.assertTrue(np.isfinite(np.array(model.conditional(u, v))).all(), model.name)

    def test_mixture_conditional_uses_prior_weights(self):
        # Copula margins are uniform, so h is the direct weighted sum, without
        # posterior component responsibilities or density-normalized weights.
        components = MODELS[2:]
        mixture = Copula("CFG", components=components, weights=(.2, .3, .5))
        u, v = np.array([.1, .3, .8]), np.array([.15, .55, .7])
        expected_density = sum(w*np.exp(c.logpdf(u, v)) for w, c in zip(mixture.weights, components))
        assert_allclose(np.exp(mixture.logpdf(u, v)), expected_density, rtol=1e-13)
        for j in range(2):
            expected = sum(w*c.conditional(u, v)[j] for w, c in zip(mixture.weights, components))
            assert_allclose(mixture.conditional(u, v)[j], expected, rtol=1e-13)


class CopulaFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2754)
        sample = rng.multivariate_normal([0, 0], [[1, .65], [.65, 1]], size=350)
        cls.u = stats.rankdata(sample[:, 0]) / 351
        cls.v = stats.rankdata(sample[:, 1]) / 351
        cls.fits = fit_candidates(cls.u, cls.v)

    def test_gaussian_parameter_recovery(self):
        self.assertAlmostEqual(self.fits["Gaussian"].params["rho"], .65, delta=.1)

    def test_mixture_nesting_and_likelihood_accounting(self):
        self.assertEqual(set(self.fits), {"Gaussian", "Student-t", "Clayton", "Frank", "Gumbel", "CFG", "CtG"})
        for model in self.fits.values():
            self.assertTrue(np.isfinite(model.loglik))
            self.assertAlmostEqual(model.loglik, np.sum(model.logpdf(self.u, self.v)), places=7)
            self.assertIn("successful_starts", model.to_dict()["fit_details"])
        for name, component_names, n_params in (("CFG", ("Clayton", "Frank", "Gumbel"), 5),
                                              ("CtG", ("Clayton", "Student-t", "Gumbel"), 6)):
            model = self.fits[name]
            self.assertGreaterEqual(model.loglik + 1e-7, max(self.fits[c].loglik for c in component_names))
            self.assertTrue(np.all(np.array(model.weights) >= 0))
            self.assertAlmostEqual(sum(model.weights), 1., places=12)
            self.assertEqual(model.n_params, n_params)

    def test_rejects_invalid_inputs(self):
        for u, v in (([.2]*20, [.3]*20), ([.2]*20, [np.nan]*20), ([-.1]*20, [.3]*20), ([.1]*5, [.2]*5)):
            with self.assertRaises(ValueError):
                fit_candidates(u, v)


if __name__ == "__main__":
    unittest.main()
