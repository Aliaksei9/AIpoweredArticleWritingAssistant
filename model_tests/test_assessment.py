import math
import unittest

from src.assesment_conform_of_PP_metric_based import AssesmentConformPPmetricsBased


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.assessor = AssesmentConformPPmetricsBased()

    def test_range_and_reference_values(self):
        self.assertIsNone(self.assessor.assess(0, 0, 0, 1000))
        self.assertEqual(self.assessor.assess(10000, 80, 5, 100000), 0.0)
        for counts in ((1, 1, 1), (1000, 100, 20), (10**18, 10**18, 10**18)):
            score = self.assessor.assess(*counts, 1000)
            self.assertGreaterEqual(score, -10)
            self.assertLessEqual(score, 10)

    def test_engagement_has_more_weight_than_views(self):
        view_gain = self.assessor.assess(20000, 80, 5, 100000)
        reaction_gain = self.assessor.assess(10000, 160, 5, 100000)
        comment_gain = self.assessor.assess(10000, 80, 10, 100000)
        self.assertGreater(reaction_gain, view_gain)
        self.assertGreater(comment_gain, view_gain)
        self.assertAlmostEqual(reaction_gain, comment_gain)
        activity = .2*math.log1p(1)+.4*math.log1p(2)+.4*math.log1p(1)
        expected = 10*math.tanh(math.log(activity/math.log(2))/1.5)
        self.assertAlmostEqual(reaction_gain, expected, places=6)

    def test_realistic_engagement_spans_both_sides_of_zero(self):
        self.assertLess(self.assessor.assess(7000, 30, 1, 100000), -4)
        self.assertGreater(self.assessor.assess(20000, 160, 10, 100000), 2)
        for index in range(3):
            low = [10000, 80, 5, 100000]
            high = low.copy()
            high[index] *= 2
            self.assertGreater(self.assessor.assess(*high), self.assessor.assess(*low))

    def test_audience_normalization(self):
        score = self.assessor.assess(750, 100, 15, 1000)
        self.assertEqual(score, self.assessor.assess(1500, 200, 30, 2000))
        self.assertLess(self.assessor.assess(750, 100, 15, 2000), score)

    def test_missing_inputs_and_zero_audience(self):
        for missing_index in range(4):
            inputs = [500, 50, 10, 1000]
            inputs[missing_index] = None
            self.assertIsNone(self.assessor.assess(*inputs))
        self.assertIsNone(self.assessor.assess(500, 50, 10, 0))
        self.assertIsNone(self.assessor.assess(0, 50, 10, 1000))

    def test_zero_engagement_is_valid_and_does_not_cap_other_metrics_at_two_points(self):
        self.assertIsNotNone(self.assessor.assess(10000, 0, 0, 100000))
        self.assertGreater(self.assessor.assess(100000, 800, 0, 100000), 2)
        self.assertGreater(self.assessor.assess(10000, 80, 1, 100000),
                           self.assessor.assess(10000, 80, 0, 100000))

    def test_extreme_valid_ratios_do_not_overflow_or_become_an_invalid_zero(self):
        for values in ((1e-300, 0, 0, 1e300), (1e300, 1e300, 1e300, 1e-300),
                       (1, 0, 0, 1), (1e-300, 1e-300, 1e-300, 1e300)):
            score = self.assessor.assess(*values)
            self.assertTrue(math.isfinite(score))
            self.assertTrue(-10 <= score <= 10)

    def test_invalid_inputs_and_config(self):
        for value in (-1, math.inf, math.nan):
            for metric_index in range(4):
                inputs = [500, 50, 10, 1000]
                inputs[metric_index] = value
                with self.assertRaises(ValueError):
                    self.assessor.assess(*inputs)
            with self.assertRaises(ValueError):
                AssesmentConformPPmetricsBased(view_reference_rate=value)
        with self.assertRaises(ValueError):
            AssesmentConformPPmetricsBased(comment_reference_rate=0)
        with self.assertRaises(ValueError):
            AssesmentConformPPmetricsBased(temperature=0)
        custom = AssesmentConformPPmetricsBased(1.0, 0.1, 0.02)
        self.assertEqual(custom.assess(1000, 100, 20, 1000), 0.0)
