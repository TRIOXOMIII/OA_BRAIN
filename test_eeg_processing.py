"""Focused checks for reproducibility and spectral calculations in EEG Lab."""
import unittest

import numpy as np

from eeg_processing import BANDS, CHANNELS, SFREQ, SimulationParameters, analyse_raw, comparison_frame, generate_synthetic_raw


class EEGProcessingTests(unittest.TestCase):
    def test_signal_is_reproducible_with_matching_parameters(self):
        parameters = SimulationParameters(seed=17, focus_band="Theta", focus_channels=("C3", "C4"), focus_gain=1.4)
        first = generate_synthetic_raw(parameters)
        second = generate_synthetic_raw(parameters)
        np.testing.assert_allclose(first.get_data(), second.get_data(), rtol=0, atol=0)

    def test_raw_metadata_matches_lab_specification(self):
        raw = generate_synthetic_raw(SimulationParameters(duration_seconds=12.0))
        self.assertEqual(tuple(raw.ch_names), CHANNELS)
        self.assertEqual(raw.info["sfreq"], SFREQ)
        self.assertEqual(raw.n_times, int(12.0 * SFREQ))

    def test_relative_power_sums_to_one_hundred_percent_per_channel(self):
        analysis = analyse_raw(generate_synthetic_raw(SimulationParameters(seed=23)), CHANNELS)
        totals = analysis["relative"].groupby("Channel")["Relative power (%)"].sum()
        np.testing.assert_allclose(totals.to_numpy(), np.full(len(CHANNELS), 100.0), rtol=1e-6, atol=1e-6)
        self.assertEqual(set(analysis["relative"]["Band"]), set(BANDS))

    def test_comparison_is_zero_for_identical_simulations(self):
        parameters = SimulationParameters(seed=31)
        left = analyse_raw(generate_synthetic_raw(parameters), CHANNELS)
        right = analyse_raw(generate_synthetic_raw(parameters), CHANNELS)
        difference = comparison_frame(left, right, "relative")
        np.testing.assert_allclose(difference["Difference (B − A)"].to_numpy(), 0.0, rtol=0, atol=1e-10)

    def test_focus_gain_changes_calculated_band_power(self):
        base = SimulationParameters(seed=37, focus_band="Theta", focus_channels=("C3", "C4"), focus_gain=1.0)
        adjusted = SimulationParameters(seed=37, focus_band="Theta", focus_channels=("C3", "C4"), focus_gain=1.7)
        left = analyse_raw(generate_synthetic_raw(base), CHANNELS)
        right = analyse_raw(generate_synthetic_raw(adjusted), CHANNELS)
        difference = comparison_frame(left, right, "relative")
        self.assertGreater(float(np.abs(difference["Difference (B − A)"]).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
