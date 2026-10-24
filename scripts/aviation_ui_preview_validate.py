"""Check aviation preview coverage/packing; never treat fixtures as flight evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(manifest):
    manifest=Path(manifest);data=json.loads(manifest.read_text());records=[]
    for preview in data['previews']:
        image=Path(preview['output']);layout=preview.get('renderer_last_layout')
        if not isinstance(layout,dict):raise ValueError('Renderer must expose actual layout metadata')
        expected=preview.get('expected_roi_ids',preview.get('actual_source_object_ids'))
        ids=layout['object_ids'];rectangles=layout['roi_rects']
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected) or len(rectangles)!=len(expected):
            raise ValueError('Some object IDs were duplicated or silently omitted')
        if preview['renderer_source_sha256']!=sha(ROOT/'airavoid/aviation_render.py'):
            raise ValueError('Previews are stale after a renderer change; rerun the suite')
        if preview['grouping_source_sha256']!=sha(ROOT/'airavoid/aviation_regions.py'):
            raise ValueError('Previews are stale after a grouping change; rerun the suite')
        if preview['output_sha256']!=sha(image):raise ValueError('Preview PNG changed')
        with Image.open(image) as picture:
            if picture.size!=(3840,1080):raise ValueError('Wrong actual PNG dimensions')
            if preview['kind']=='synthetic_layout_fixture_only' and picture.getpixel((0,0))!=(100,57,8):
                raise ValueError('Synthetic fixture lost its burned-in banner')
        for i,a in enumerate(rectangles):
            x,y,w,h=a
            if w<=0 or h<=0 or x<0 or y<0 or x+w>3840 or y+h>1080:raise ValueError('ROI outside the canvas')
            for b in rectangles[i+1:]:
                q,r,s,t=b
                if min(x+w,q+s)>max(x,q) and min(y+h,r+t)>max(y,r):raise ValueError('ROI cells overlap')
        badges=layout.get('group_badges',[]);focus=layout.get('focus_id');labels=set(layout.get('field_label_ids',[]))
        expected_groups=[g['member_ids'] for g in preview.get('attention_regions',[]) if len(g['member_ids'])>1]
        if {tuple(b['member_ids']) for b in badges}!={tuple(m) for m in expected_groups}:
            raise ValueError('Multi-object attention group lost its compact count badge')
        fx,fy,fw,fh=layout['field_rect']
        for i,badge in enumerate(badges):
            if badge['count']!=len(badge['member_ids']):raise ValueError('Group badge count differs from actual members')
            if (set(badge['member_ids'])-{focus})&labels:raise ValueError('Nonfocus member labels were not suppressed inside the group')
            x,y,right,bottom=badge['rect']
            if x<fx or y<fy or right>fx+fw or bottom>fy+fh:raise ValueError('Group badge outside the image field')
            for other in badges[i+1:]:
                q,r,s,t=other['rect']
                if min(right,s)>max(x,q) and min(bottom,t)>max(y,r):raise ValueError('Group count badges overlap')
        records.append({'kind':preview['kind'],'path':str(image),'object_count':len(ids),'ROI_count':len(rectangles),
                        'object_ids':ids,'region_count':layout['region_count'],'all_IDs_retained':True,
                        'all_ROIs_inside_canvas':True,'ROI_overlap_count':0,'group_badge_count':len(badges),
                        'group_badge_overlap_count':0,'group_member_labels_are_compact':True,'PNG_sha256':preview['output_sha256']})
    return {'state':'complete','validated_utc':datetime.now(timezone.utc).isoformat(),'preview_manifest_sha256':sha(manifest),
            'real_recorded_previews':sum(r['kind']=='recorded_rgb_and_recorded_online_trace' for r in records),
            'synthetic_layout_fixtures':sum(r['kind']=='synthetic_layout_fixture_only' for r in records),
            'runtime_scope':'cached CPU presentation checks only; no live neural FPS or physical safety result',
            'records':records}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(ROOT/'artifacts/aviation_ui/previews/preview_manifest.json'))
    p.add_argument('--output',default=str(ROOT/'reports/aviation_ui_preview_validation.json'));a=p.parse_args()
    result=validate(a.manifest);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False))
