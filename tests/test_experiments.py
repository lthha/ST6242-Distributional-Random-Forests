"""Checks of estimands and bootstrap weights against independent calculations."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from drf_core import candidate_splits, mmd_split_scores, gaussian_kernel, forest_weights
from ST6242_DRF_full_forest_experiment import normal_embedding, normal_self, law_embeddings, functional_errors, conditional_risk, regime_gram, risk_decomposition
import ST6242_DRF_smooth_experiment as smooth


class ExperimentChecks(unittest.TestCase):
    def test_fast_mmd_against_direct_child_matrices(self):
        y = np.random.default_rng(8).normal(size=31)
        counts = np.arange(5, 27)
        actual = mmd_split_scores(y, counts, 0.7)
        kernel = gaussian_kernel(y, y, 0.7)
        expected = []
        for m in counts:
            mmd = kernel[:m, :m].mean() + kernel[m:, m:].mean() - 2*kernel[:m, m:].mean()
            expected.append(m*(len(y)-m)/len(y)**2*mmd)
        np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_candidates_respect_fraction_and_distinct_values(self):
        x = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7])
        order, counts, thresholds = candidate_splits(x, .2)
        self.assertTrue(np.all(counts >= 2) and np.all(len(x)-counts >= 2))
        self.assertTrue(np.all(x[order][counts-1] < thresholds))
        self.assertTrue(np.all(thresholds < x[order][counts]))

    def test_bootstrap_weights_reproduce_sklearn_multioutput_prediction(self):
        rng = np.random.default_rng(10)
        x, query = rng.normal(size=(70, 3)), rng.normal(size=(12, 3))
        y = np.column_stack((x[:, 0]**2, np.sin(x[:, 1])))
        model = RandomForestRegressor(n_estimators=17, min_samples_leaf=4, random_state=44).fit(x, y)
        weights = forest_weights(model, x, query)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(weights @ y, model.predict(query), atol=1e-13)
        self.assertGreaterEqual(weights.min(), 0)

    def test_normal_kernel_integral_against_numerical_integration(self):
        from scipy.integrate import quad
        h, variance, mean, point = .8, 2.0, -.3, 1.2
        value = quad(lambda z: np.exp(-(point-z)**2/(2*h*h))*stats.norm.pdf(z, loc=mean, scale=np.sqrt(variance)), -np.inf, np.inf)[0]
        actual = normal_embedding(np.array([[point]]), np.array([mean]), np.array([[variance]]), h)[0]
        self.assertAlmostEqual(actual, value, places=11)
        self.assertAlmostEqual(normal_self(np.array([[variance]]), h), 1/np.sqrt(1+2*variance/h**2), places=14)

    def test_mixture_embedding_matches_component_sum(self):
        y = np.array([[-2.], [0.], [.5], [1.5]])
        cross, pp = law_embeddings(y, "bimodality", 1)
        variance, a = 1-.95**2, .95
        expected = .5/np.sqrt(1+variance)*(np.exp(-(y[:, 0]-a)**2/(2*(1+variance))) + np.exp(-(y[:, 0]+a)**2/(2*(1+variance))))
        np.testing.assert_allclose(cross[:, 1], expected)
        self.assertTrue(np.all((pp > 0) & (pp <= 1)))

    def test_weighted_functionals_and_pointmass_mmd(self):
        weights = np.array([[.5, .5], [1., 0.]])
        y = np.array([[-1.], [1.]])
        error = functional_errors(weights, y, np.array([0, 0]), "variance")
        np.testing.assert_allclose(error, [0., 1.])
        kernel = gaussian_kernel(y, y)
        cross = np.column_stack((kernel[:, 0], kernel[:, 1]))
        risk = conditional_risk(np.eye(2), kernel, np.array([0, 1]), cross, np.ones(2))
        np.testing.assert_allclose(risk, 0, atol=1e-14)

    def test_regime_gram_against_independent_formulas(self):
        from scipy.integrate import quad
        self.assertAlmostEqual(regime_gram("variance", 1.0)[0, 1], 1/np.sqrt(6), places=14)
        self.assertAlmostEqual(regime_gram("dependence", 1.4)[0, 1], 1/(1 + 2/1.96), places=14)
        a, var = .95, 1 - .95**2
        self.assertAlmostEqual(regime_gram("bimodality", 1.0)[0, 1],
                               np.exp(-a*a/(2*(2+var)))/np.sqrt(2+var), places=14)
        # E k(Z, T) with Z ~ N(0,1) and T ~ t_3/sqrt(3), integrating over T.
        density = lambda t: np.sqrt(3)*stats.t.pdf(np.sqrt(3)*t, df=3)
        exact = quad(lambda t: np.exp(-t*t/4)/np.sqrt(2)*density(t), -np.inf, np.inf)[0]
        self.assertAlmostEqual(regime_gram("tail", 1.0)[0, 1], exact, places=6)
        gram = regime_gram("variance", 1.0)
        distance = gram[0, 0] + gram[1, 1] - 2*gram[0, 1]
        self.assertAlmostEqual(distance, 1/np.sqrt(3) + 1/3 - 2/np.sqrt(6), places=14)

    def test_step_decomposition_matches_general_bias_formula(self):
        rng = np.random.default_rng(3)
        n, q = 40, 9
        weights = rng.dirichlet(np.ones(n), size=q)
        train_t, test_t = rng.uniform(size=n), rng.uniform(size=q)
        y = rng.normal(size=n) * np.sqrt(smooth.variance_function(train_t, 0))
        v_train, v_test = smooth.variance_function(train_t, 0), smooth.variance_function(test_t, 0)
        kernel = gaussian_kernel(y, y)
        risk, bias = smooth.conditional_risk(weights, kernel, y, v_train, v_test, 1.0)
        groups, train_groups = (test_t > .5).astype(int), (train_t > .5).astype(int)
        cross, pp = law_embeddings(y[:, None], "variance", 1.0)
        np.testing.assert_allclose(risk, conditional_risk(weights, kernel, groups, cross, pp), atol=1e-13)
        parts = risk_decomposition(weights, train_groups, groups, regime_gram("variance", 1.0))
        np.testing.assert_allclose(bias, parts["bias"], atol=1e-13)
        np.testing.assert_allclose(parts["n_eff"], 1/np.sum(weights**2, axis=1))


if __name__ == "__main__":
    unittest.main()
