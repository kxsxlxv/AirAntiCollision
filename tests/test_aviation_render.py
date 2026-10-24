import copy
import unittest
import numpy as np
from airavoid.aviation_render import (AviationRenderer,roi_grid_layout,fit_rect,
    source_class,range_text,fresh_observation,crop_bounds)


class AviationRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.renderer=AviationRenderer()

    def fixture(self,count=2,fresh=True):
        objects=[];targets=[]
        for i in range(count):
            x=60+(i%8)*75;y=60+(i//8)*90
            box=[x-9,y-4,x+9,y+4]
            objects.append({'id':i+1,'box':box,'last_seen':1. if fresh else .97,
                'source':'detector','class_id':0,'class_display':'БПЛА самолётного типа',
                'demo_observation':{'measured':fresh}})
            targets.append({'id':i+1,'box':box.copy(),'state':'MONITORING',
                'evidence_age_s':0.,'measured_trail':[[.9,x-3,y],[1.,x,y]],
                'velocity_px_s':[30.,0.],'growth_per_s':.05,
                'range_approx_m':45.4,'range_display_m':45})
        advice={'demo':{'targets':targets,'priority_id':1,'state':'MONITORING',
            'attention_regions':[],'direction':{'status':'resolved','vector':np.array([0.,-2.])}}}
        return objects,advice

    def test_grid_packs_every_object_without_overlap_through_32(self):
        for n in [0,1,2,3,4,8,16,32]:
            with self.subTest(n=n):
                cells=roi_grid_layout(n);self.assertEqual(len(cells),n)
                for i,(x,y,w,h) in enumerate(cells):
                    self.assertGreater(w,0);self.assertGreater(h,0)
                    self.assertGreaterEqual(x,2726);self.assertGreaterEqual(y,124)
                    self.assertLessEqual(x+w,3820);self.assertLessEqual(y+h,1012)
                    for xx,yy,ww,hh in cells[:i]:
                        self.assertTrue(x+w<=xx or xx+ww<=x or y+h<=yy or yy+hh<=y)
        a,b=roi_grid_layout(2)
        self.assertEqual(a[0],b[0]);self.assertLess(a[1],b[1])

    def test_source_fit_and_crop_keep_actual_aspect_and_bounds(self):
        for width,height in [(1280,960),(1920,1080),(480,800)]:
            (_,_,w,h),scale=fit_rect(width,height,AviationRenderer.video_area)
            self.assertLessEqual(abs(w-width*scale),.5)
            self.assertLessEqual(abs(h-height*scale),.5)
        for aspect in [.5,1.6,3.5]:
            l,t,r,b=crop_bounds([3,4,20,12],640,480,aspect)
            self.assertGreaterEqual(l,0);self.assertGreaterEqual(t,0)
            self.assertLessEqual(r,640);self.assertLessEqual(b,480)
            self.assertLessEqual(l,3);self.assertGreaterEqual(r,20)

    def test_only_same_frame_measured_detections_are_fresh(self):
        obj={'last_seen':1.,'source':'detector','demo_observation':{'measured':True}}
        self.assertTrue(fresh_observation(obj,1.))
        self.assertFalse(fresh_observation(obj,1.03))
        self.assertFalse(fresh_observation(obj,.98))
        self.assertFalse(fresh_observation(dict(obj,source='sot'),1.))
        self.assertFalse(fresh_observation(dict(obj,demo_observation={'measured':False}),1.))

    def test_range_only_formats_supplied_finite_positive_values(self):
        for value in [None,0,-1,np.nan,np.inf]:
            self.assertEqual(range_text({'range_approx_m':value}),'D —')
        self.assertEqual(range_text({'range_display_m':45,'range_approx_m':45.4}),'≈ 45 м')
        self.assertEqual(source_class({'class':'fixed-wing UAV'}),'самолёт')
        self.assertEqual(source_class({'class_id':2}),'птица')
        self.assertEqual(source_class({}),'класс —')

    def test_full_size_output_preserves_source_and_input_payload(self):
        frame=np.full((480,640,3),(25,65,105),np.uint8)
        objects,advice=self.fixture();original=frame.copy()
        vector=advice['demo']['direction']['vector'].copy();boxes=copy.deepcopy(objects)
        result=self.renderer.draw(frame,objects,1.,{},advice)
        self.assertEqual(result.shape,(1080,3840,3));self.assertEqual(result.dtype,np.uint8)
        np.testing.assert_array_equal(frame,original)
        np.testing.assert_array_equal(advice['demo']['direction']['vector'],vector)
        self.assertEqual(objects,boxes)
        np.testing.assert_array_equal(result[600,500],frame[0,0])
        self.assertEqual(self.renderer.last_layout['object_ids'],[1,2])
        self.assertTrue(self.renderer.last_layout['direction_drawn'])

    def test_stale_roi_never_crops_the_current_frame_or_keeps_arrow(self):
        frame=np.full((480,640,3),(0,0,255),np.uint8)
        objects,advice=self.fixture(fresh=False)
        result=self.renderer.render(frame,objects,1.,{},advice)
        roi=result[124:1012,2726:3820]
        self.assertEqual(np.sum(np.all(roi==[0,0,255],axis=2)),0)
        self.assertEqual(self.renderer.last_layout['fresh_ids'],[])
        self.assertFalse(self.renderer.last_layout['direction_drawn'])
        self.assertEqual(self.renderer.last_layout['state'],'INSUFFICIENT EVIDENCE')

    def test_32_distinct_real_rgb_patches_all_reach_roi_grid(self):
        objects,advice=self.fixture(32)
        frame=np.zeros((480,640,3),np.uint8);colors=[]
        for obj in objects:
            i=obj['id'];color=(40+i*3,60+i*2,80+i*4);colors.append(color)
            x1,y1,x2,y2=map(int,obj['box']);frame[y1:y2,x1:x2]=color
        output=self.renderer.render(frame,objects,1.,{},advice)
        self.assertEqual(self.renderer.last_layout['object_ids'],list(range(1,33)))
        self.assertEqual(len(self.renderer.last_layout['roi_rects']),32)
        roi=output[124:1012,2726:3820]
        for color in colors:self.assertGreater(np.count_nonzero(np.all(roi==color,axis=2)),0)

    def test_cached_attention_region_is_removed_when_measurement_is_stale(self):
        frame=np.zeros((480,640,3),np.uint8);objects,advice=self.fixture(1,False)
        advice['demo']['attention_regions']=[{'member_ids':[1],'severity':'convergence',
            'hull_px':[[20,20],[100,20],[100,100],[20,100]]}]
        self.renderer.render(frame,objects,1.,{},advice)
        self.assertEqual(self.renderer.last_layout['region_count'],0)

    def test_default_title_is_short_but_custom_title_survives(self):
        self.assertEqual(AviationRenderer('RGB • анализ сближения').title,'ОБЗОР')
        self.assertEqual(AviationRenderer('Мой обзор').title,'Мой обзор')

    def test_dense_group_replaces_only_field_labels_and_keeps_all_rois(self):
        objects,advice=self.fixture(32);frame=np.zeros((480,640,3),np.uint8)
        advice['demo']['attention_regions']=[{'member_ids':list(range(1,33)),
            'severity':'attention','hull_px':[[30,30],[615,30],[615,370],[30,370]]}]
        self.renderer.render(frame,objects,1.,{},advice)
        layout=self.renderer.last_layout
        self.assertEqual(layout['field_label_ids'],[1])
        self.assertEqual(len(layout['roi_rects']),32)
        self.assertEqual(layout['object_ids'],list(range(1,33)))
        self.assertEqual(layout['group_badges'][0]['count'],32)
        self.assertEqual(layout['group_badges'][0]['member_ids'],list(range(1,33)))

    def test_single_member_groups_keep_their_individual_field_ids(self):
        objects,advice=self.fixture(2);frame=np.zeros((480,640,3),np.uint8)
        advice['demo']['attention_regions']=[{'member_ids':[i],'severity':'attention',
            'hull_px':[[10,10],[100,10],[100,100],[10,100]]} for i in [1,2]]
        self.renderer.render(frame,objects,1.,{},advice)
        self.assertEqual(self.renderer.last_layout['field_label_ids'],[1,2])
        self.assertEqual(self.renderer.last_layout['group_badges'],[])

    def test_group_badges_do_not_overlap_one_another(self):
        objects,advice=self.fixture(4);frame=np.zeros((480,640,3),np.uint8)
        advice['demo']['attention_regions']=[{'member_ids':ids,'severity':'attention',
            'hull_px':[[30,30],[400,30],[400,120],[30,120]]} for ids in [[1,2],[3,4]]]
        self.renderer.render(frame,objects,1.,{},advice)
        a,b=[g['rect'] for g in self.renderer.last_layout['group_badges']]
        self.assertTrue(a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1])


if __name__=='__main__':unittest.main()
