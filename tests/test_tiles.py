import unittest
import numpy as np
from airavoid.detector import native_tiles,interior_boxes


class NativeTileTests(unittest.TestCase):
    def test_existing_qut_grid_unchanged(self):
        self.assertEqual(native_tiles(1280,720),[(x,y) for y in [0,304] for x in [0,288,576]])

    def test_full_pixel_coverage_small_tall_and_4k(self):
        for width,height in [(320,240),(1920,1080),(3840,2160),(720,2160)]:
            covered=np.zeros((height,width),bool)
            for x,y in native_tiles(width,height):covered[y:y+416,x:x+704]=True
            self.assertTrue(covered.all(),(width,height))

    def test_sixty_pixel_targets_fit_a_crop_at_boundaries(self):
        width,height=3840,2160;tiles=native_tiles(width,height)
        for y in range(0,height-60,17):
            for x in range(0,width-60,29):
                self.assertTrue(any(tx<=x and ty<=y and tx+704>=x+60 and ty+416>=y+60 for tx,ty in tiles))

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):native_tiles(1280,720,overlap=1)

    def test_crop_guard_distinguishes_internal_and_true_image_edges(self):
        boxes=np.array([[0,20,30,40,.5,0],[30,20,60,40,.5,0]],float)
        self.assertEqual(len(interior_boxes(boxes,100,0,704,416,1280,720,2)),1)
        self.assertEqual(len(interior_boxes(boxes,0,0,704,416,1280,720,2)),2)
