import io
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airavoid.tartan_reference import (read_clock, read_reference_pickle,
    interpolate_reference, match_objects, fit_horizontal_cpa)


class TartanReferenceTests(unittest.TestCase):
    def test_clock_uses_rt_not_srt_display_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clock.srt'
            path.write_text('1\n00:00:00,000 --> 00:00:00,000\nRT: 1000 123456789\n\n'
                            '2\n00:00:00,000 --> 00:00:03,000\nRT: 1000 173456789\n')
            clock = read_clock(path)
            self.assertAlmostEqual(clock[2] - clock[1], .05)

    def test_clock_rejects_missing_or_reversed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'clock.srt'
            for text in ('', '1\nx\nRT: 10 0\n\n3\nx\nRT: 11 0\n',
                         '1\nx\nRT: 11 0\n\n2\nx\nRT: 10 0\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_clock(path)

    def test_restricted_pickle_numeric_scalar_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reference.pkl'
            for protocol in (3, 4):
                path.write_bytes(pickle.dumps({123: {4: [np.float64(3.25)]}}, protocol=protocol))
                self.assertEqual(read_reference_pickle(path), {123: {4: [3.25]}})

    def test_pickle_global_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unsafe.pkl'
            path.write_bytes(b"cos\nsystem\n(S'echo must_not_execute'\ntR.")
            with self.assertRaises(ValueError):
                read_reference_pickle(path)

    def test_offline_matching_is_one_to_one(self):
        objects = [{'id': 9, 'box': [0, 0, 10, 10]}, {'id': 10, 'box': [0, 0, 10, 10]}]
        labels = [{'icao': 900, 'box': [0, 0, 10, 10]}]
        matches = match_objects(objects, labels)
        self.assertEqual(len(matches), 1)
        self.assertEqual([o['id'] for o in objects], [9, 10])
        self.assertNotIn('icao', objects[0])

    def test_interpolation_does_not_extrapolate_or_bridge_missing_data(self):
        points = [{'icao': 1, 'utc_s': t, **{k: float(t) for k in
                  ('north_m', 'east_m', 'vertical_proxy_m', 'horizontal_range_m', 'slant_proxy_m')}} for t in (10, 12)]
        self.assertEqual(interpolate_reference(points, 1, 11)['north_m'], 11)
        self.assertIsNone(interpolate_reference(points, 1, 9))
        self.assertIsNone(interpolate_reference(points, 1, 13))
        self.assertIsNone(interpolate_reference(points, 2, 11))
        self.assertIsNone(interpolate_reference(points, 1, 11, max_gap=1))

    def test_reference_cpa_does_not_use_future_points(self):
        points = [{'utc_s': float(t), 'north_m': 1000. - 100*t, 'east_m': 500.} for t in range(6)]
        first = fit_horizontal_cpa(points, 4.)
        self.assertAlmostEqual(first['dcpa_horizontal_m'], 500.)
        self.assertAlmostEqual(first['tcpa_horizontal_s'], 6.)
        points[-1]['east_m'] = 1e9
        self.assertEqual(first, fit_horizontal_cpa(points, 4.))

    def test_receding_cpa_is_clamped_to_present(self):
        points = [{'utc_s': float(t), 'north_m': 1000. + 100*t, 'east_m': 500.} for t in range(6)]
        self.assertEqual(fit_horizontal_cpa(points, 5.)['tcpa_horizontal_s'], 0.)


if __name__ == '__main__':
    unittest.main()
