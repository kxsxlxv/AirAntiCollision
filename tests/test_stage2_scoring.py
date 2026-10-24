"""Offline oracle/scoring contracts: exact boxes, timestamps, ID scope, no GT camera."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
AVAILABLE=(ROOT/'external/mot/TrackTrack/3. Tracker/trackeval/metrics/hota.py').exists() and (ROOT/'external/mot/python_deps/scipy').exists()
if AVAILABLE:
    from stage2_oracle import oracle_rows
    from stage2_score import match_boxes,continuity,gt_array
    from mot_metrics import sequence_data,HOTA,Identity,CLEAR
    from stage2_assess_trace import replay,text_open,stabilize_observation


@unittest.skipUnless(AVAILABLE,'Optional isolated MOT research dependencies are absent')
class Stage2OfflineContracts(unittest.TestCase):
    def setUp(self):
        self.gt=[{'frame_id':0,'object_id':5,'x':10.,'y':20.,'w':10.,'h':10.}]
        self.row={'frame':0,'time':0.,'source_pts_s':1.23,'camera':{'valid':False,'status':'unobserved','reference_epoch':0},
                  'stabilization_transform':np.eye(3).tolist(),
                  'tracks':[[91.,10.,20.,20.,30.],[92.,60.,60.,70.,70.]],
                  'measurements':[{'track_id':91,'measured_bbox':[10.,20.,20.,30.],'confidence':.2,'ambiguous':True,'source':'detector'},
                                  {'track_id':92,'measured_bbox':[60.,60.,70.,70.],'confidence':.1,'ambiguous':False,'source':'detector'}]}

    def test_identity_oracle_changes_ids_only_and_resolves_known_identity(self):
        before=json.dumps(self.row,sort_keys=True)
        result=next(oracle_rows([self.row],self.gt,'oracle_identity_only'))
        self.assertEqual(result['measurements'][0]['track_id'],5)
        self.assertFalse(result['measurements'][0]['ambiguous'])
        self.assertGreater(result['measurements'][1]['track_id'],1000000)
        for a,b in zip(result['measurements'],self.row['measurements']):
            self.assertEqual(a['measured_bbox'],b['measured_bbox']);self.assertEqual(a['confidence'],b['confidence'])
        self.assertEqual(result['camera'],self.row['camera'])
        self.assertEqual(result['stabilization_transform'],self.row['stabilization_transform'])
        self.assertEqual(result['source_pts_s'],1.23)
        self.assertEqual(json.dumps(self.row,sort_keys=True),before)

    def test_box_oracle_removes_false_positive_and_marks_source(self):
        result=next(oracle_rows([self.row],self.gt,'oracle_box_and_identity'))
        self.assertEqual(len(result['measurements']),1)
        self.assertEqual(result['measurements'][0]['source'],'oracle_box')
        self.assertIsNone(result['measurements'][0]['confidence'])
        self.assertTrue(result['offline_oracle_gt_access'])

    def test_gt_adapter_and_trackeval_exact_frames(self):
        data=sequence_data([self.row],gt_array(self.gt))
        self.assertEqual(data['num_gt_dets'],1)
        result=Identity().eval_sequence(data)
        self.assertEqual(result['IDTP'],1)
        self.assertEqual(result['IDFP'],1)
        self.assertEqual(result['IDFN'],0)

    def test_matcher_really_prioritizes_cardinality_over_overlap_sum(self):
        overlaps=np.eye(5)*.5
        for i in range(4):overlaps[i,i+1]=.99
        with patch('stage2_score.iou',return_value=overlaps):
            pairs,_=match_boxes(np.zeros((5,4)),np.zeros((5,4)),.5)
        self.assertEqual(len(pairs),5)

    def test_continuity_keeps_missing_opportunity_separate(self):
        gt=self.gt+[dict(self.gt[0],frame_id=1)]
        empty=dict(self.row,frame=1,measurements=[],tracks=[])
        result=continuity([self.row,empty],gt)
        self.assertEqual(result['matched_fraction'],.5)
        self.assertEqual(result['target_frame_opportunities'],2)

    def test_compressed_assessment_has_same_schema_and_unobserved_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'input.jsonl';dest=Path(td)/'out.jsonl.gz'
            later=dict(self.row,frame=1,time=.5,measurements=[],tracks=[])
            src.write_text(json.dumps(self.row)+'\n'+json.dumps(later)+'\n')
            report=replay(src,dest,100,100,methods=['lai_lin'])
            with text_open(dest) as f:rows=[json.loads(x) for x in f]
            self.assertEqual(report['frames'],2)
            self.assertEqual(rows[0]['schema'],'stage2-assessment-v1')
            self.assertTrue(all(o['results']['lai_lin']['decision']=='unobserved' for o in rows[1]['objects']))

    def test_homography_divides_coordinates_and_uses_local_jacobian(self):
        h=np.array([[1.2,.1,3.],[.2,.8,4.],[.001,-.002,1.]])
        box=[10.,20.,20.,30.];p=np.array([15.,25.]);position,scale=stabilize_observation(box,h,'homography')
        def project(v):
            a=h@np.r_[v,1.];return a[:2]/a[2]
        np.testing.assert_allclose(position,project(p))
        eps=1e-4
        jac=np.column_stack([(project(p+eps*np.eye(2)[i])-project(p-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
        self.assertAlmostEqual(scale,np.sqrt(abs(np.linalg.det(jac))),places=8)

    def test_replay_rejects_noncausal_timestamps_and_ignores_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'input.jsonl';dest=Path(td)/'output.jsonl'
            bad=dict(self.row,frame=1,time=0.)
            src.write_text(json.dumps(self.row)+'\n'+json.dumps(bad)+'\n')
            with self.assertRaisesRegex(ValueError,'strictly increasing'):replay(src,dest,100,100,methods=['lai_lin'])
            predicted=dict(self.row['measurements'][0],measured=False,source='tracker_predict')
            row=dict(self.row,measurements=[predicted])
            src.write_text(json.dumps(row)+'\n')
            report=replay(src,dest,100,100,methods=['lai_lin'])
            self.assertEqual(report['ignored_predicted_observations'],1)
            self.assertEqual(json.loads(dest.read_text())['objects'],[])
            unmarked=dict(self.row,measurements=[dict(self.row['measurements'][0],source='oracle_box')])
            src.write_text(json.dumps(unmarked)+'\n')
            with self.assertRaisesRegex(ValueError,'explicitly labelled offline oracle'):replay(src,dest,100,100,methods=['lai_lin'])

    def test_camera_validity_loss_resets_same_epoch_compensated_history(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'input.jsonl';dest=Path(td)/'output.jsonl'
            rows=[]
            for i in range(3):
                row=dict(self.row,frame=i,time=i*.1,measurements=[dict(self.row['measurements'][0],ambiguous=False)])
                row['camera']=dict(self.row['camera'],valid=i<2,evidence_reset=i==2)
                rows.append(row)
            src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            replay(src,dest,100,100,methods=['lai_lin'],camera_model='similarity')
            out=[json.loads(x) for x in dest.read_text().splitlines()]
            e=out[-1]['objects'][0]['results']['lai_lin']
            self.assertEqual(e['validity']['measured_samples'],1)
            self.assertEqual(e['reason_codes'],['history'])


if __name__=='__main__':unittest.main()
