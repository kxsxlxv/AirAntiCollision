import copy
import unittest
import numpy as np
from airavoid.demo_render import DemoRenderer,CYAN,video_placement,direction_label


class DemoRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.renderer=DemoRenderer()

    def payload(self,resolved=False):
        return {'demo':{'state':'MONITORING','priority_id':7,'targets':[{
            'id':7,'state':'MONITORING','reason':'machine_reason',
            'reason_text':'Наблюдаем цель по независимым измерениям',
            'box':[600,450,630,470],'growth_per_s':.02,
            'drift_normalized_per_s':.14,'window_s':.5,'samples':14,
            'measured_trail':[[.9,610,460],[1.,615,460]],
            'velocity_px_s':[4.,0.]}],
            'direction':{'status':'resolved' if resolved else 'unresolved',
                'vector':np.array([2.,0.]),'reason_text':'Иллюстрация по осям изображения'}}}

    def objects(self,last_seen=1.):
        return [{'id':7,'last_seen':last_seen,'class_id':0,'class_display':'БПЛА самолётного типа'}]

    def test_four_three_and_widescreen_sources_keep_uniform_scale(self):
        for width,height in [(1280,960),(1920,1080),(480,800)]:
            (x,y,w,h),scale=video_placement(width,height)
            self.assertLessEqual(abs(w-width*scale),.5)
            self.assertLessEqual(abs(h-height*scale),.5)
            self.assertGreaterEqual(x,24);self.assertGreaterEqual(y,100)
            self.assertLessEqual(x+w,1496);self.assertLessEqual(y+h,984)
        self.assertEqual(video_placement(1280,960)[0][3],884)

    def test_renderer_shape_and_inputs_are_unchanged(self):
        frame=np.full((960,1280,3),(20,80,140),np.uint8);original=frame.copy()
        advice=self.payload(True);vector=advice['demo']['direction']['vector'].copy()
        output=self.renderer.draw(frame,self.objects(),1.,{},advice)
        self.assertEqual(output.shape,(1080,1920,3));self.assertEqual(output.dtype,np.uint8)
        np.testing.assert_array_equal(frame,original)
        np.testing.assert_array_equal(advice['demo']['direction']['vector'],vector)
        # Uniform source pixels far from all target annotations retain BGR colour.
        np.testing.assert_array_equal(output[250,400],frame[0,0])

    def test_unresolved_direction_cannot_draw_arrow_even_with_vector(self):
        frame=np.zeros((960,1280,3),np.uint8)
        hidden=self.renderer.render(frame,self.objects(),1.,{},self.payload(False))
        shown=self.renderer.render(frame,self.objects(),1.,{},self.payload(True))
        color=np.array(CYAN[::-1],np.uint8)
        self.assertEqual(np.sum(np.all(hidden[782:906,35:160]==color,axis=2)),0)
        self.assertGreater(np.sum(np.all(shown[782:906,35:160]==color,axis=2)),100)

    def test_stale_target_cannot_support_current_highlight_or_direction(self):
        frame=np.zeros((960,1280,3),np.uint8)
        result=self.renderer.render(frame,self.objects(last_seen=0.),1.,{},self.payload(True))
        color=np.array(CYAN[::-1],np.uint8)
        self.assertEqual(np.sum(np.all(result[782:906,35:160]==color,axis=2)),0)

    def test_projected_path_stays_inside_actual_source_image(self):
        frame=np.zeros((960,1280,3),np.uint8);advice=self.payload(False)
        advice['demo']['targets'][0]['velocity_px_s']=[-1e6,0.]
        result=self.renderer.render(frame,self.objects(),1.,{},advice)
        np.testing.assert_array_equal(result[300:650,40:150],np.broadcast_to(
            np.array([13,8,3],np.uint8),(350,110,3)))

    def test_direction_words_follow_screen_axes_without_flight_axes(self):
        self.assertEqual(direction_label([-1,0]),'ВЛЕВО')
        self.assertEqual(direction_label([0,1]),'ВНИЗ')
        self.assertIsNone(direction_label([0,0]))

    def test_display_focus_stays_stable_without_inventing_geometric_priority(self):
        renderer=DemoRenderer();frame=np.zeros((960,1280,3),np.uint8)
        advice=self.payload(True);advice['demo']['priority_id']=None
        first=advice['demo']['targets'][0]
        second=copy.deepcopy(first);second['id']=8;second['box']=[100,100,110,110]
        advice['demo']['targets'].append(second)
        objects=self.objects()+[{'id':8,'last_seen':1.,'class_id':0}]
        renderer.render(frame,objects,1.,{},advice)
        self.assertEqual(renderer.focus_id,7)
        second['box']=[100,100,180,180]
        out=renderer.render(frame,objects,1.,{},advice)
        self.assertEqual(renderer.focus_id,7)
        color=np.array(CYAN[::-1],np.uint8)
        self.assertEqual(np.sum(np.all(out[782:906,35:160]==color,axis=2)),0)


if __name__=='__main__':unittest.main()
