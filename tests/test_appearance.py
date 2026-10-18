import unittest
from airavoid.appearance import combine_observations


class ObservationTests(unittest.TestCase):
    def test_detector_corrects_same_target_without_duplicate(self):
        detector=[[10,10,30,20,.4,0]]
        sot=[[11,11,31,21,.9,0,'sot']]
        self.assertEqual(combine_observations(detector,sot),detector)

    def test_unmatched_target_remains_image_observation(self):
        detector=[[10,10,30,20,.4,0]]
        sot=[[100,100,130,120,.9,1,'sot']]
        self.assertEqual(len(combine_observations(detector,sot)),2)
        self.assertEqual(combine_observations(detector,sot)[1][6],'sot')

    def test_class_change_does_not_duplicate_same_physical_object(self):
        detector=[[10,10,30,20,.4,0]]
        sot=[[11,11,31,21,.9,2,'sot']]
        self.assertEqual(combine_observations(detector,sot),detector)
