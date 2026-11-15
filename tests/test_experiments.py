"""Checks of estimands and bootstrap weights against independent calculations."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from drf_core import candidate_splits, mmd_split_scores, gaussian_kernel, forest_weights
from ST6242_DRF_full_forest_experiment import normal_embedding, normal_self, law_embeddings, functional_errors, conditional_risk


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


if __name__ == "__main__":
    unittest.main()
