import copy
import json
import math
import random
import unittest
from airavoid.aviation_regions import compute_attention_regions


def target(tid,box,velocity=(0,0),state='MONITORING',**extra):
    return dict(id=tid,box=list(box),velocity_px_s=list(velocity),state=state,
                evidence_age_s=0.,reason='stable_measured_motion_no_convergence_cue',**extra)


class AttentionRegionTests(unittest.TestCase):
    def test_no_eligible_observation_does_not_create_a_safe_region(self):
        self.assertEqual(compute_attention_regions([],200,100),[])
        obj=target(1,[20,20,30,30]);obj['state']='INSUFFICIENT EVIDENCE'
        self.assertEqual(compute_attention_regions([obj],200,100),[])

    def test_individual_attention_region_retains_object_link(self):
        groups=compute_attention_regions([target(9,[20,30,40,50])],200,100)
        self.assertEqual(len(groups),1);region=groups[0]
        self.assertEqual(region['member_ids'],[9]);self.assertEqual(region['severity'],'attention')
        for actual,expected in zip(region['bounds_px'],[12,22,48,58]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(region['meaning'],'image_envelope_attention_not_physical_collision_or_safe_region')
        self.assertEqual(region['complement'],'unknown_or_unmeasured_not_safe')

    def test_connected_components_are_transitive_not_a_prespecified_cluster_count(self):
        objects=[target(7,[80,50,90,60]),target(2,[30,50,40,60]),
                 target(5,[55,50,65,60]),target(20,[150,50,160,60])]
        groups=compute_attention_regions(objects,200,120)
        self.assertEqual([group['member_ids'] for group in groups],[[2,5,7],[20]])
        self.assertIn('bridge empty pixels',groups[0]['grouping'])

    def test_convergence_severity_propagates_through_whole_component(self):
        objects=[target(1,[20,20,30,30]),target(2,[40,20,50,30],state='POSSIBLE CONVERGENCE')]
        groups=compute_attention_regions(objects,200,100)
        self.assertEqual(groups[0]['member_ids'],[1,2]);self.assertEqual(groups[0]['severity'],'convergence')

    def test_measured_half_second_sweep_changes_overlap(self):
        objects=[target(1,[20,20,30,30]),target(2,[85,20,95,30])]
        self.assertEqual(len(compute_attention_regions(objects,200,100)),2)
        objects[0]['velocity_px_s']=[100,0]
        self.assertEqual(compute_attention_regions(objects,200,100)[0]['member_ids'],[1,2])

    def test_overlapping_axis_bounds_do_not_merge_disjoint_diagonal_envelopes(self):
        objects=[target(1,[20,20,30,30],velocity=(160,160)),
                 target(2,[20,85,30,95],velocity=(160,160))]
        groups=compute_attention_regions(objects,200,220)
        self.assertEqual(len(groups),2)
        a,b=groups[0]['bounds_px'],groups[1]['bounds_px']
        self.assertGreater(min(a[2],b[2])-max(a[0],b[0]),0)
        self.assertGreater(min(a[3],b[3])-max(a[1],b[1]),0)

    def test_touching_closed_envelopes_form_one_group(self):
        groups=compute_attention_regions([target(1,[20,20,30,30]),target(2,[46,20,56,30])],200,100)
        self.assertEqual(groups[0]['member_ids'],[1,2])

    def test_polygons_clip_to_source_under_very_large_drift(self):
        objects=[target(1,[5,5,25,25],velocity=(1e308,-1e308)),
                 target(2,[170,70,195,95],velocity=(-1e308,1e307))]
        groups=compute_attention_regions(objects,200,100)
        self.assertTrue(groups)
        for region in groups:
            self.assertGreaterEqual(len(region['hull_px']),3)
            for x,y in region['hull_px']:
                self.assertTrue(math.isfinite(x) and math.isfinite(y))
                self.assertTrue(0<=x<=200 and 0<=y<=100)
        json.dumps(groups,allow_nan=False)

    def test_current_outside_box_is_unobserved_even_if_motion_enters_image(self):
        self.assertEqual(compute_attention_regions([target(1,[-40,20,-20,40],velocity=(100,0))],200,100),[])

    def test_partial_unflagged_box_is_clipped_but_explicit_clipped_flag_is_rejected(self):
        groups=compute_attention_regions([target(1,[-3,5,10,15])],200,100)
        self.assertEqual(groups[0]['bounds_px'][0],0.)
        self.assertEqual(compute_attention_regions([target(1,[-3,5,10,15],clipped=True)],200,100),[])

    def test_stale_and_nonfinite_measurements_are_excluded(self):
        for age in [.04,.25,-.1,float('nan'),float('inf'),None]:
            obj=target(1,[20,20,40,40]);obj['evidence_age_s']=age
            self.assertEqual(compute_attention_regions([obj],200,100),[])
        for field,value in [('box',[20,20,float('nan'),40]),('velocity_px_s',[float('inf'),0]),
                            ('velocity_px_s',None),('box',[20,20,10,10])]:
            obj=target(1,[20,20,40,40]);obj[field]=value
            self.assertEqual(compute_attention_regions([obj],200,100),[])

    def test_ambiguous_unreliable_tiny_and_predicted_observations_are_excluded(self):
        for extra in [dict(ambiguous=True),dict(reliable=False),dict(valid=False),dict(tiny=True),
                      dict(unobserved=True),dict(measured=False),dict(source='sot')]:
            self.assertEqual(compute_attention_regions([target(1,[20,20,40,40],**extra)],200,100),[])
        self.assertEqual(compute_attention_regions([target(1,[20,20,25,24])],200,100),[])
        obj=target(1,[20,20,40,40]);obj['reason']='unstable_scale_or_aspect'
        self.assertEqual(compute_attention_regions([obj],200,100),[])

    def test_duplicate_object_identity_is_not_silently_linked_to_two_regions(self):
        objects=[target(1,[20,20,40,40]),target(1,[100,20,120,40]),target(2,[160,60,180,80])]
        groups=compute_attention_regions(objects,200,100)
        self.assertEqual([group['member_ids'] for group in groups],[[2]])

    def test_input_order_does_not_change_groups_hull_order_or_source_data(self):
        objects=[target(7,[80,50,90,60]),target(2,[30,50,40,60]),target(5,[55,50,65,60])]
        original=copy.deepcopy(objects);expected=compute_attention_regions(objects,200,120)
        for seed in range(8):
            shuffled=copy.deepcopy(objects);random.Random(seed).shuffle(shuffled)
            self.assertEqual(compute_attention_regions(shuffled,200,120),expected)
        self.assertEqual(objects,original)

    def test_twenty_simultaneous_targets_have_no_arbitrary_group_limit(self):
        objects=[target(i,[20+40*i,40,30+40*i,50]) for i in range(20)]
        groups=compute_attention_regions(objects,900,100)
        self.assertEqual(len(groups),20)
        self.assertEqual([group['member_ids'] for group in groups],[[i] for i in range(20)])

    def test_invalid_image_geometry_is_rejected(self):
        for width,height in [(0,100),(100,-1),(float('nan'),100),(100,float('inf')),(None,100)]:
            with self.assertRaises(ValueError):compute_attention_regions([],width,height)


if __name__=='__main__':unittest.main()
