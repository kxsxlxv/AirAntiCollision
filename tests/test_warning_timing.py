import unittest
from scripts.warning_timing import episodes


class WarningTimingTests(unittest.TestCase):
    def test_annotation_gap_does_not_count_as_persistent_warning(self):
        samples=[{'time':t,'state':'threat','range':100} for t in [0.,.2,.4,5.,5.2]]
        result=episodes(samples,.2)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(result[0]['sample_hold_duration_s'],.6)
        self.assertIsNone(result[0]['qualification_support_end_s'])

    def test_episode_reports_retrospective_onset_and_later_support(self):
        samples=[{'time':t,'state':'threat','range':100} for t in [10.,10.2,10.4,10.6]]
        result=episodes(samples,.2)[0]
        self.assertEqual(result['onset_s'],10.)
        self.assertAlmostEqual(result['qualification_support_end_s'],10.8)
