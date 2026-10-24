"""Deterministic visual groups of current measured image envelopes.

These regions express attention in the image, not physical collision volumes or
semantic segmentation. Their complement is unknown/unmeasured, never 'safe'.
Connected components of overlapping swept envelopes determine the group count;
the final convex hull can bridge empty pixels within a connected component.
No calibration, GT, inferred range, camera state, or clustering K is used.
"""
from collections import Counter
from math import isfinite


_STATES={'MONITORING','POSSIBLE CONVERGENCE'}
_HORIZON_S=.5
_MIN_SPAN_PX=6.
_FRESH_TOLERANCE_S=1e-6
_EPS=1e-12
_MEANING='image_envelope_attention_not_physical_collision_or_safe_region'


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _hull(points):
    points=sorted(set(points))
    if len(points)<=2:return points
    lower=[];upper=[]
    for point in points:
        while len(lower)>=2 and _cross(lower[-2],lower[-1],point)<=0:lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper)>=2 and _cross(upper[-2],upper[-1],point)<=0:upper.pop()
        upper.append(point)
    return lower[:-1]+upper[:-1]


def _clip(polygon,width,height):
    """Sutherland--Hodgman clipping of a convex polygon, in normalized units."""
    for axis,bound,greater in [(0,0.,True),(0,width,False),(1,0.,True),(1,height,False)]:
        if not polygon:return []
        result=[];previous=polygon[-1]
        inside_previous=previous[axis]>=bound if greater else previous[axis]<=bound
        for current in polygon:
            inside=current[axis]>=bound if greater else current[axis]<=bound
            if inside!=inside_previous:
                ratio=(bound-previous[axis])/(current[axis]-previous[axis])
                intersection=[previous[i]+ratio*(current[i]-previous[i]) for i in (0,1)]
                intersection[axis]=bound
                result.append(tuple(intersection))
            if inside:result.append(current)
            previous=current;inside_previous=inside
        polygon=result
    return _hull([(min(width,max(0.,x)),min(height,max(0.,y))) for x,y in polygon])


def _overlap(first,second):
    """Separating-axis test: closed envelopes sharing an edge are connected."""
    for polygon in (first,second):
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            normal=(-(b[1]-a[1]),b[0]-a[0])
            p=[x*normal[0]+y*normal[1] for x,y in first]
            q=[x*normal[0]+y*normal[1] for x,y in second]
            if max(p)<min(q)-_EPS or max(q)<min(p)-_EPS:return False
    return True


def _eligible(target,width,height,scale):
    if target.get('state') not in _STATES:return None
    if target.get('measured') is False or target.get('reliable') is False or target.get('valid') is False:return None
    if any(target.get(key) for key in ('ambiguous','clipped','tiny','unobserved')):return None
    if 'source' in target and target['source'] not in ('detector','zoom'):return None
    reason=str(target.get('reason','')).lower()
    if any(part in reason for part in ('unstable','too_small','tiny','unreliable','ambiguous','clipped',
                                      'no_fresh','unobserved','insufficient','discontinuity')):return None
    try:
        age=float(target['evidence_age_s']);raw_id=target['id']
        if isinstance(raw_id,bool):return None
        tid=int(raw_id)
        if raw_id!=tid:return None
        box=tuple(float(v) for v in target['box'])
        velocity=tuple(float(v) for v in target['velocity_px_s'])
    except (KeyError,TypeError,ValueError,OverflowError):return None
    if not isfinite(age) or abs(age)>_FRESH_TOLERANCE_S:return None
    if len(box)!=4 or len(velocity)!=2 or not all(isfinite(v) for v in (*box,*velocity)):return None
    x1,y1,x2,y2=box
    if x2<=x1 or y2<=y1 or max(x2-x1,y2-y1)<_MIN_SPAN_PX:return None
    # An entirely invisible current box cannot create an observed attention area.
    x1=max(0.,min(width,x1));x2=max(0.,min(width,x2))
    y1=max(0.,min(height,y1));y2=max(0.,min(height,y2))
    if x2<=x1 or y2<=y1 or max(x2-x1,y2-y1)<_MIN_SPAN_PX:return None
    padding=max(8./scale,.2*max((x2-x1)/scale,(y2-y1)/scale))
    left=x1/scale-padding;right=x2/scale+padding
    top=y1/scale-padding;bottom=y2/scale+padding
    dx=_HORIZON_S*(velocity[0]/scale);dy=_HORIZON_S*(velocity[1]/scale)
    largest=max(abs(dx),abs(dy))
    if largest>16.:
        # Preserve direction, keeping the endpoint well beyond the visible image.
        # The clipped swept envelope is unchanged; huge offscreen coordinates
        # cannot overflow hull cross products or create NaN intersections.
        dx=16.*(dx/largest);dy=16.*(dy/largest)
    corners=[(left,top),(right,top),(right,bottom),(left,bottom)]
    polygon=_clip(_hull(corners+[(x+dx,y+dy) for x,y in corners]),width/scale,height/scale)
    if len(polygon)<3:return None
    return {'id':tid,'state':target['state'],'polygon':polygon}


def compute_attention_regions(targets,width,height):
    """Return linked convex visual groups in raw source-image pixel coordinates.

    Only current reliable MONITORING/POSSIBLE CONVERGENCE measurements qualify;
    ``evidence_age_s`` must be zero within 1 microsecond and measured velocity
    must be finite. The envelope is the bbox padded by max(8 px,20% span), swept
    for 0.5 s along the local measured image velocity. Padding and projection are
    visual conventions, not uncertainty bounds or predicted physical occupancy.

    Member IDs and group order are deterministic. Invalid targets are excluded;
    duplicate eligible IDs are ambiguous and all their entries are excluded.
    The input objects are never changed. Complexity is O(n^2) for overlap tests.
    """
    try:width=float(width);height=float(height)
    except (TypeError,ValueError,OverflowError):raise ValueError('Finite source image dimensions required') from None
    if not isfinite(width) or not isfinite(height) or min(width,height)<1:
        raise ValueError('Finite source image dimensions of at least one pixel required')
    scale=max(width,height)
    eligible=[]
    for target in targets:
        if not isinstance(target,dict):continue
        item=_eligible(target,width,height,scale)
        if item is not None:eligible.append(item)
    multiplicity=Counter(item['id'] for item in eligible)
    eligible=sorted((item for item in eligible if multiplicity[item['id']]==1),key=lambda item:item['id'])
    parent=list(range(len(eligible)))
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    for i,first in enumerate(eligible):
        for j in range(i+1,len(eligible)):
            if _overlap(first['polygon'],eligible[j]['polygon']):
                a,b=root(i),root(j)
                if a!=b:parent[max(a,b)]=min(a,b)
    components={}
    for index,item in enumerate(eligible):components.setdefault(root(index),[]).append(item)
    result=[]
    for members in components.values():
        hull=_hull([point for member in members for point in member['polygon']])
        points=[[float(min(width,max(0.,x*scale))),float(min(height,max(0.,y*scale)))] for x,y in hull]
        result.append({'member_ids':[member['id'] for member in members],
                       'severity':'convergence' if any(member['state']=='POSSIBLE CONVERGENCE' for member in members) else 'attention',
                       'hull_px':points,
                       'bounds_px':[min(p[0] for p in points),min(p[1] for p in points),
                                    max(p[0] for p in points),max(p[1] for p in points)],
                       'meaning':_MEANING,'projection_horizon_s':_HORIZON_S,
                       'grouping':'connected overlap of swept image envelopes; convex visual hull may bridge empty pixels',
                       'complement':'unknown_or_unmeasured_not_safe'})
    return sorted(result,key=lambda group:tuple(group['member_ids']))
