import inspect
import unittest

import numpy as np

from SCoRE import CS, SCoRE_MDR, SCoRE_MDR_bf, SCoRE_MDR_w, SCoRE_SDR, SCoRE_SDR_w


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


if __name__ == "__main__":
    unittest.main()
