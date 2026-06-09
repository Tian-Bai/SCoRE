import unittest

import numpy as np

import SCoRE
from SCoRE import (
    BH,
    CS,
    SCoRE_MDR,
    SCoRE_SDR,
    gen_data_1,
)


class TestPublicApi(unittest.TestCase):
    def test_public_exports_are_explicit(self):
        self.assertEqual(SCoRE.__version__, "0.1.0")
        self.assertIn("SCoRE_SDR", SCoRE.__all__)
        self.assertNotIn("SCoRE_SDR_fast", SCoRE.__all__)
        self.assertNotIn("SCoRE_SDR_bin", SCoRE.__all__)
        self.assertNotIn("np", SCoRE.__all__)

    def test_bh_returns_rejected_indices_in_pvalue_order(self):
        selected = BH([0.01, 0.04, 0.03, 0.20], 0.1)
        np.testing.assert_array_equal(selected, np.array([0, 2, 1]))

    def test_selection_procedures_return_index_arrays(self):
        lcalib = np.array([0, 1, 0, 1])
        scalib = np.array([0.1, 0.4, 0.2, 0.8])
        stest = np.array([0.15, 0.5, 0.9])

        cs_selected = CS((lcalib, scalib), (None, stest), 0.5, mult_test=False)
        mdr_selected = SCoRE_MDR((lcalib, scalib), (None, stest), 0.5, 0.5)

        self.assertEqual(cs_selected.dtype.kind, "i")
        self.assertEqual(mdr_selected.dtype.kind, "i")
        np.testing.assert_array_equal(cs_selected, np.array([0, 1]))
        np.testing.assert_array_equal(mdr_selected, np.array([0, 1]))

    def test_sdr_returns_expected_fixture(self):
        lcalib = np.array([0, 1, 0, 1])
        scalib = np.array([0.1, 0.4, 0.2, 0.8])
        stest = np.array([0.15, 0.5, 0.9])

        selected, evalues = SCoRE_SDR(
            (lcalib, scalib), (None, stest), 0.5, 0.5, return_evals=True
        )

        np.testing.assert_array_equal(selected, np.array([], dtype=int))
        np.testing.assert_allclose(evalues, np.zeros(3))

    def test_random_state_makes_pruned_evalues_reproducible(self):
        lcalib = np.array([0, 0, 1, 0])
        scalib = np.array([0.1, 0.2, 0.6, 0.7])
        stest = np.array([0.15, 0.5, 0.9])

        _, first = SCoRE_SDR(
            (lcalib, scalib),
            (None, stest),
            0.5,
            1.0,
            prune="hete",
            return_evals=True,
            random_state=123,
        )
        _, second = SCoRE_SDR(
            (lcalib, scalib),
            (None, stest),
            0.5,
            1.0,
            prune="hete",
            return_evals=True,
            random_state=123,
        )

        np.testing.assert_allclose(first, second)

    def test_data_generation_random_state_is_reproducible(self):
        first = gen_data_1(1, 5, 0.1, random_state=42)
        second = gen_data_1(1, 5, 0.1, random_state=42)

        for first_part, second_part in zip(first, second):
            np.testing.assert_allclose(first_part, second_part)

    def test_invalid_inputs_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "setting"):
            gen_data_1(3, 5, 0.1)

        with self.assertRaisesRegex(ValueError, "alpha"):
            CS((np.array([0]), np.array([0.1])), (None, np.array([0.1])), 0)

        with self.assertRaisesRegex(ValueError, "prune"):
            SCoRE_SDR(
                (np.array([0]), np.array([0.1])),
                (None, np.array([0.1])),
                0.5,
                0.5,
                prune="bad",
            )


if __name__ == "__main__":
    unittest.main()
