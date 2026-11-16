import inspect
import unittest

import numpy as np

from SCoRE import CS, SCoRE_MDR, SCoRE_MDR_bf, SCoRE_MDR_w, SCoRE_SDR, SCoRE_SDR_w
from SCoRE import (
    SCoRE_MSR, SCoRE_MSR_bf, SCoRE_MSR_w, SCoRE_WSR, SCoRE_WSR_w,
    eval_MDR, eval_MSR, eval_SDR, eval_WSR, score_evalues, cost_ebh,
)


class ScoreApiTests(unittest.TestCase):
    def setUp(self):
        self.lcalib = np.array([0, 1, 0, 1])
        self.scalib = np.array([0.1, 0.4, 0.2, 0.8])
        self.stest = np.array([0.15, 0.5, 0.9])

    def test_methods_accept_stest_directly(self):
        dcalib = (self.lcalib, self.scalib)

        np.testing.assert_array_equal(
            CS(dcalib, self.stest, 0.5, mult_test=False),
            np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            SCoRE_MDR(dcalib, self.stest, 0.5, 0.5),
            np.array([0, 1]),
        )
        selected_bf, evalues = SCoRE_MDR_bf(dcalib, self.stest, 0.5, 0.5, return_evals=True)
        self.assertEqual(selected_bf.dtype.kind, "i")
        self.assertEqual(evalues.shape, self.stest.shape)
        np.testing.assert_array_equal(
            SCoRE_SDR(dcalib, self.stest, 0.5, 0.5),
            np.array([], dtype=int),
        )

    def test_weighted_methods_accept_stest_directly(self):
        dcalib = (self.lcalib, self.scalib)
        wcalib = np.ones_like(self.scalib)
        wtest = np.ones_like(self.stest)

        np.testing.assert_array_equal(
            SCoRE_MDR_w(dcalib, self.stest, wcalib, wtest, 0.5, 0.5),
            np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            SCoRE_SDR_w(dcalib, self.stest, wcalib, wtest, 0.5, 0.5),
            np.array([], dtype=int),
        )

    def test_plain_python_list_stest_is_not_treated_as_legacy_tuple(self):
        dcalib = (self.lcalib[:2], self.scalib[:2])
        selected = CS(dcalib, [0.15, 0.5], 0.5, mult_test=False)

        self.assertEqual(selected.dtype.kind, "i")

    def test_legacy_dtest_tuple_still_works(self):
        dcalib = (self.lcalib, self.scalib)
        direct = SCoRE_MDR(dcalib, self.stest, 0.5, 0.5)
        legacy = SCoRE_MDR(dcalib, (None, self.stest), 0.5, 0.5)
        legacy_with_labels = SCoRE_MDR(dcalib, (np.ones_like(self.stest), self.stest), 0.5, 0.5)

        np.testing.assert_array_equal(direct, legacy)
        np.testing.assert_array_equal(direct, legacy_with_labels)

    def test_cs_requires_binary_loss(self):
        dcalib = (np.array([0, 0.5, 1]), np.array([0.1, 0.2, 0.3]))

        with self.assertRaisesRegex(ValueError, "binary"):
            CS(dcalib, self.stest, 0.5)

    def test_sdr_api_does_not_expose_oracle_mode(self):
        self.assertNotIn("oracle", inspect.signature(SCoRE_SDR).parameters)
        self.assertNotIn("oracle", inspect.signature(SCoRE_SDR_w).parameters)

    def test_msr_names_and_compatibility_aliases(self):
        self.assertIs(SCoRE_MDR, SCoRE_MSR)
        self.assertIs(SCoRE_MDR_bf, SCoRE_MSR_bf)
        self.assertIs(SCoRE_MDR_w, SCoRE_MSR_w)
        self.assertIs(eval_MDR, eval_MSR)
        np.testing.assert_array_equal(
            SCoRE_MSR(D_calib=(self.lcalib, self.scalib),
                      D_test=(None, self.stest), alpha=0.5, gamma=0.5), [0, 1])

    def test_wsr_tuples_and_independent_unit_cost_defaults(self):
        cc, ct = np.array([0.5, 2, 1, 3]), np.array([2, 1, 4])
        for use_cc in (False, True):
            for use_ct in (False, True):
                dc = (self.lcalib, self.scalib, cc) if use_cc else (self.lcalib, self.scalib)
                dt = (None, self.stest, ct) if use_ct else (None, self.stest)
                expected = score_evalues(self.lcalib, self.scalib, self.stest, 0.5,
                                         cc=cc if use_cc else None, ct=ct if use_ct else None)
                for container in (tuple, list):
                    selected, values = SCoRE_WSR(
                        D_calib=container(dc), D_test=container(dt),
                        alpha=0.5, gamma=0.5, return_evals=True)
                    np.testing.assert_allclose(values, expected)
                    np.testing.assert_array_equal(selected, cost_ebh(expected, ct if use_ct else None, 0.5))

    def test_test_losses_do_not_affect_selection(self):
        dc = (self.lcalib, self.scalib, np.ones(4))
        for procedure in (SCoRE_WSR, SCoRE_SDR, SCoRE_MSR, SCoRE_MSR_bf):
            expected = procedure(dc, (None, self.stest, np.ones(3)), 0.5, 0.5)
            for losses in (np.zeros(3), np.full(3, np.nan), object()):
                actual = procedure(dc, (losses, self.stest, np.ones(3)), 0.5, 0.5)
                np.testing.assert_array_equal(actual, expected)

    def test_sdr_and_msr_use_unit_normalization(self):
        dc = (self.lcalib, self.scalib)
        dt = (None, self.stest)
        cost_dc = dc + (np.array([5, 10, 15, 20]),)
        cost_dt = dt + (np.array([10, 20, 30]),)
        np.testing.assert_array_equal(CS(dc, dt, 0.5), CS(cost_dc, cost_dt, 0.5))
        for procedure in (SCoRE_SDR, SCoRE_MSR, SCoRE_MSR_bf):
            np.testing.assert_array_equal(
                procedure(dc, dt, 0.5, 0.5), procedure(cost_dc, cost_dt, 0.5, 0.5))
        for procedure in (SCoRE_SDR_w, SCoRE_MSR_w):
            np.testing.assert_array_equal(
                procedure(dc, dt, np.ones(4), np.ones(3), 0.5, 0.5),
                procedure(cost_dc, cost_dt, np.ones(4), np.ones(3), 0.5, 0.5))
        # WSR must actually use the supplied costs, yielding a different set.
        self.assertFalse(np.array_equal(SCoRE_WSR(cost_dc, cost_dt, 0.5, 0.5),
                                        SCoRE_SDR(cost_dc, cost_dt, 0.5, 0.5)))

    def test_wsr_unit_costs_equal_sdr_with_boosting_and_shift(self):
        dc, dt = (self.lcalib, self.scalib), (None, self.stest)
        for prune in (None, "hete", "homo"):
            for wsr, sdr, weights in ((SCoRE_WSR, SCoRE_SDR, ()),
                                     (SCoRE_WSR_w, SCoRE_SDR_w,
                                      (np.array([1, 2, 3, 1]), np.array([2, 1, 3])))):
                options = dict(prune=prune, return_evals=True, random_state=42)
                a = wsr(dc, dt, *weights, 0.5, 0.5, **options)
                b = sdr(dc, dt, *weights, 0.5, 0.5, **options)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_plain_three_score_list_is_not_a_cost_tuple(self):
        dc = (self.lcalib, self.scalib)
        np.testing.assert_array_equal(SCoRE_WSR(dc, list(self.stest), 0.5, 0.5),
                                      SCoRE_WSR(dc, (None, self.stest), 0.5, 0.5))

    def test_tuple_cost_validation(self):
        for bad in ([1], [0, 1, 1], [-1, 1, 1], [np.nan, 1, 1], [[1, 1, 1]]):
            with self.assertRaises(ValueError):
                SCoRE_WSR((self.lcalib, self.scalib), (None, self.stest, bad), 0.5, 0.5)
        with self.assertRaises(ValueError):
            SCoRE_WSR((self.lcalib, self.scalib, [1]), (None, self.stest), 0.5, 0.5)

    def test_risk_evaluation_names(self):
        losses, rewards, selected = np.array([0, 2, 0]), np.ones(3), [0, 1]
        self.assertEqual(eval_MSR(losses, rewards, selected)[0], 2 / 3)
        self.assertEqual(eval_SDR(losses, rewards, selected)[0], 1)
        self.assertEqual(eval_WSR(losses, rewards, selected, [1, 3, 2])[0], 0.5)


if __name__ == "__main__":
    unittest.main()
