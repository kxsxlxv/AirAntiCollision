"""Check identity-evaluation conventions on exact, missed, and switched tracks."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
AVAILABLE=(ROOT/'external/mot/TrackTrack/3. Tracker/trackeval/metrics/hota.py').exists() and (ROOT/'external/mot/python_deps/scipy').exists()
if AVAILABLE:from mot_metrics import sequence_data,HOTA,Identity,CLEAR

@unittest.skipUnless(AVAILABLE,'Optional isolated MOT research dependencies are absent')
class MOTMetricsTest(unittest.TestCase):
    def evaluate(self,tracks):
        rows=[dict(frame=i,tracks=t) for i,t in enumerate(tracks)]
        gt=np.array([[i+1,1,0,0,10,10,1,1,1] for i in range(len(rows))],float)
        data=sequence_data(rows,gt)
        return Identity().eval_sequence(data),CLEAR({'PRINT_CONFIG':False}).eval_sequence(data),HOTA().eval_sequence(data)
    def test_exact_identity(self):
        identity,clear,hota=self.evaluate([[[7,0,0,10,10]]]*4)
        self.assertEqual(identity['IDF1'],1);self.assertEqual(clear['IDSW'],0)
        np.testing.assert_array_equal(hota['HOTA'],1)
    def test_switch_is_penalized(self):
        identity,clear,hota=self.evaluate([[[7,0,0,10,10]]]*2+[[[8,0,0,10,10]]]*2)
        self.assertEqual(identity['IDF1'],.5);self.assertEqual(clear['IDSW'],1)
        np.testing.assert_allclose(hota['HOTA'],np.sqrt(.5))
    def test_missing_track(self):
        identity,clear,hota=self.evaluate([[]]*4)
        self.assertEqual(identity['IDF1'],0);self.assertEqual(clear['CLR_FN'],4)
    def test_reacquisition_switch(self):
        identity,clear,hota=self.evaluate([[[7,0,0,10,10]],[[99,100,100,110,110]],[[8,0,0,10,10]]])
        self.assertEqual(clear['IDSW'],1);self.assertEqual(clear['Frag'],1)

if __name__=='__main__':unittest.main()
