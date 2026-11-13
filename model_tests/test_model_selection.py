import unittest

from model_tests.model_selection import select_model


class ModelSelectionTests(unittest.TestCase):
    def test_training_time_is_ignored_and_slow_inference_is_rejected(self):
        results = {
            "accurate": dict(cv_mse=dict(mean=1.0), response_time=dict(p95=.5), fit_seconds=100000),
            "fast_fit": dict(cv_mse=dict(mean=2.0), response_time=dict(p95=.1), fit_seconds=.001),
            "too_slow": dict(cv_mse=dict(mean=.1), response_time=dict(p95=61), fit_seconds=1),
            "slow_outlier": dict(cv_mse=dict(mean=.2), response_time=dict(p95=5,max=11), fit_seconds=1),
        }
        self.assertEqual(select_model(results, results, "cv_mse", 10), "accurate")
        results["accurate"]["fit_seconds"] = 999999999
        self.assertEqual(select_model(results, results, "cv_mse", 10), "accurate")

    def test_response_time_breaks_equal_cv_error_and_no_eligible_model_is_explicit(self):
        results = {"a": dict(cv_selection=dict(mean=1), response_time=dict(p95=2)),
                   "b": dict(cv_selection=dict(mean=1), response_time=dict(p95=1))}
        self.assertEqual(select_model(results, results, "cv_selection", 10), "b")
        with self.assertRaisesRegex(ValueError,"No model"):
            select_model(results, results, "cv_selection", .01)


if __name__ == "__main__":
    unittest.main()
