import json
import unittest
import numpy as np
from airavoid.demo_cue import EarlyEncounterCue,DemoEncounterCue,MONITORING,POSSIBLE,INSUFFICIENT


CAMERA={'valid':True,'mode':'auto','status':'background_similarity'}


def observation(t,tid=1,growth=.2,x=70.,y=50.,span=64.,source='detector',**guards):
    size=span*np.exp(growth*t)
    return {'id':tid,'box':[x-size/2,y-size/4,x+size/2,y+size/4],
            'last_seen':float(t),'source':source,'confidence':.7,
            'assessment':{'research':{'decision':'uncertain','reason_codes':['fixture']},'state':'uncertain'},
            'demo_observation':dict(measured=source in ('detector','zoom'),**guards)}


def drive(cue,make=lambda t:[observation(t)],end=1.2,camera=CAMERA):
    result=None
    for t in np.arange(0,end+.0001,.04):result=cue.update(make(float(t)),float(t),camera,(200,100))
    return result


class DemoCueTests(unittest.TestCase):
    def test_alias_and_short_history_confirmation_are_causal(self):
        self.assertIs(DemoEncounterCue,EarlyEncounterCue)
        cue=EarlyEncounterCue();states=[]
        for t in np.arange(0,.8001,.04):
            result=cue.update([observation(t)],float(t),CAMERA,(200,100));states.append((t,result['state']))
        first=min(t for t,s in states if s==POSSIBLE)
        self.assertGreaterEqual(first,.3+.16-1e-9)
        self.assertLess(first,.7)
        target=result['targets'][0]
        self.assertAlmostEqual(target['growth_per_s'],.2,places=8)
        self.assertTrue(all(p[0]<=.8+1e-9 for p in target['measured_trail']))
        self.assertLessEqual(target['window_s'],.6+1e-8)

    def test_stable_monitoring_priority_and_arrow_do_not_create_a_warning(self):
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,growth=0,span=40),observation(t,tid=2,growth=0,span=64)])
        self.assertEqual(result['state'],MONITORING);self.assertEqual(result['priority_id'],2)
        self.assertEqual(result['direction']['status'],'resolved')
        self.assertEqual(result['direction']['reason'],'stable_measured_target_image_separation_illustration')
        self.assertFalse(result['direction']['warning_evidence'])
        self.assertTrue(all(x['state']==MONITORING for x in result['targets']))

    def test_central_and_conflicting_monitoring_targets_still_have_no_arrow(self):
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,growth=0,x=100)])
        self.assertEqual(result['state'],MONITORING)
        self.assertEqual(result['direction']['status'],'unresolved')
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,growth=0,x=55,span=32),
                                                 observation(t,tid=2,growth=0,x=145,span=32)])
        self.assertEqual(result['state'],MONITORING)
        self.assertEqual(result['direction']['reason'],'conflicting_image_separation_directions')

    def test_measured_growing_side_target_resolves_illustrative_direction(self):
        result=drive(EarlyEncounterCue())
        self.assertEqual(result['state'],POSSIBLE);self.assertEqual(result['priority_id'],1)
        self.assertEqual(result['direction']['status'],'resolved')
        np.testing.assert_allclose(result['direction']['vector'],[1,0],atol=1e-8)
        self.assertIn('not_safe_maneuver',result['direction']['meaning'])

    def test_symmetric_central_target_has_no_invented_direction(self):
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,x=100)])
        self.assertEqual(result['state'],POSSIBLE)
        self.assertEqual(result['direction']['status'],'unresolved')
        self.assertIn('symmetric',result['direction']['reason'])

    def test_opposite_converging_targets_suppress_conflicting_arrow(self):
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,x=55),observation(t,tid=2,x=145)])
        self.assertEqual(result['state'],POSSIBLE)
        self.assertEqual(result['direction']['reason'],'conflicting_convergence_targets')

    def test_sot_and_explicit_unmeasured_boxes_never_seed_growth(self):
        result=drive(EarlyEncounterCue(),lambda t:[observation(t,source='sot')])
        self.assertEqual(result['state'],INSUFFICIENT);self.assertEqual(result['targets'][0]['samples'],0)
        def make(t):
            obj=observation(t);obj['demo_observation']['measured']=False
            return [obj]
        result=drive(EarlyEncounterCue(),make)
        self.assertEqual(result['targets'][0]['samples'],0)
        self.assertEqual(result['direction']['status'],'unresolved')

    def test_invalid_camera_and_explicit_ambiguity_remove_cue(self):
        for camera in [dict(CAMERA,valid=False),dict(CAMERA,mode='unknown')]:
            result=drive(EarlyEncounterCue(),camera=camera)
            self.assertEqual(result['state'],INSUFFICIENT);self.assertIsNone(result['priority_id'])
        for guard in ['ambiguous','clipped']:
            result=drive(EarlyEncounterCue(),lambda t:[observation(t,**{guard:True})])
            self.assertEqual(result['state'],INSUFFICIENT);self.assertEqual(result['targets'][0]['samples'],0)

    def test_camera_reset_requires_new_history(self):
        cue=EarlyEncounterCue();drive(cue,end=.8)
        result=cue.update([observation(.84)],.84,dict(CAMERA,evidence_reset=True),(200,100))
        self.assertEqual(result['state'],INSUFFICIENT);self.assertEqual(result['targets'][0]['samples'],1)
        self.assertEqual(result['direction']['status'],'unresolved')

    def test_recession_and_high_relative_drift_do_not_warn(self):
        for make in [lambda t:[observation(t,growth=-.2)],lambda t:[observation(t,x=70+40*t)]]:
            result=drive(EarlyEncounterCue(),make)
            self.assertNotEqual(result['state'],POSSIBLE)
            if result['direction']['status']=='resolved':
                self.assertFalse(result['direction']['warning_evidence'])
                self.assertEqual(result['direction']['reason'],'stable_measured_target_image_separation_illustration')

    def test_aspect_instability_is_not_growth_evidence(self):
        def make(t):
            obj=observation(t);obj['box'][1]=40 if int(t/.04)%2 else 25
            obj['box'][3]=60 if int(t/.04)%2 else 75
            return [obj]
        result=drive(EarlyEncounterCue(),make)
        self.assertEqual(result['state'],INSUFFICIENT)
        self.assertEqual(result['targets'][0]['reason'],'unstable_scale_or_aspect')

    def test_missing_observation_cannot_keep_a_warning_or_arrow(self):
        cue=EarlyEncounterCue();drive(cue,end=.8)
        result=cue.update([], .84,CAMERA,(200,100))
        self.assertEqual(result['state'],INSUFFICIENT);self.assertIsNone(result['priority_id'])
        self.assertEqual(result['direction']['status'],'unresolved')

    def test_json_finite_original_research_unchanged_and_monotonic_time(self):
        cue=EarlyEncounterCue();objects=[observation(0)];before=json.dumps(objects)
        result=cue.update(objects,0,CAMERA,(200,100))
        json.dumps(result,allow_nan=False);self.assertEqual(json.dumps(objects),before)
        with self.assertRaisesRegex(ValueError,'strictly increase'):cue.update(objects,0,CAMERA,(200,100))
        cue.reset();cue.update(objects,0,CAMERA,(200,100))

    def test_contract_declares_real_operating_rules_without_physical_claim(self):
        cue=EarlyEncounterCue(window=.7,confirmation=.2)
        contract=cue.contract;json.dumps(contract,allow_nan=False)
        self.assertEqual(contract['window_s'],.7);self.assertEqual(contract['confirmation_s'],.2)
        self.assertFalse(contract['physical_collision_or_safe_maneuver'])
        self.assertIn('MONITORING',contract['direction'])


if __name__=='__main__':unittest.main()
