"""Recover short detector gaps from tracked image features within one shot.

Neighbor boxes alone are never evidence of visibility. Every recovered frame
needs a spatially supported person detection or verified image features. For
feature-based interior recovery, independent traces from both endpoints agree.
"""
import cv2
import numpy as np
from .timeline import intervals


def overlap(a,b):
    intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(intersection/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection,1e-6))


def accepted_candidate(candidates, expected):
    """A weak person proposal is usable only near the observed same-shot actor."""
    found=[]
    for candidate in candidates:
        box=np.asarray(candidate[:4],np.float32);confidence=float(candidate[4])
        if not np.isfinite(candidate).all() or confidence<.1 or overlap(box,expected)<.55:continue
        score=overlap(box,expected)*confidence
        found.append((score,box,confidence))
    found.sort(key=lambda v:v[0],reverse=True)
    if not found:return None
    # Duplicate full-frame/crop proposals agree; competing identities do not.
    if any(overlap(found[0][1],r[1])<.5 and r[0]>found[0][0]*.7 for r in found[1:]):return None
    return found[0][1],found[0][2]


def _trace(images, anchor, stop, box):
    step=1 if stop>anchor else -1
    h,w=images[anchor].shape
    size=box[2:]-box[:2]
    lo=np.maximum(np.rint(box[:2]+size*[.2,.08]),0).astype(int)
    hi=np.minimum(np.rint(box[2:]-size*[.2,.12]),[w,h]).astype(int)
    mask=np.zeros((h,w),np.uint8);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    points=cv2.goodFeaturesToTrack(images[anchor],maxCorners=160,qualityLevel=.015,minDistance=3,mask=mask)
    if points is None or len(points)<10:return {}
    initial=len(points);out={};current=box.copy();transform=np.eye(3)
    corners=np.array([[box[0],box[1]],[box[2],box[1]],
                      [box[2],box[3]],[box[0],box[3]]])
    for frame in range(anchor+step,stop+step,step):
        prev=images[frame-step];now=images[frame]
        nxt,status,error=cv2.calcOpticalFlowPyrLK(prev,now,points,None,winSize=(21,21),maxLevel=3)
        if nxt is None:break
        back,reverse,_=cv2.calcOpticalFlowPyrLK(now,prev,nxt,None,winSize=(21,21),maxLevel=3)
        if back is None:break
        keep=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back-points,axis=(1,2))<1.5)&(error[:,0]<25)
        p,q=points[keep,0],nxt[keep,0]
        if len(p)<max(10,initial*.2):break
        affine,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3)
        # A dancing body is not rigid: the torso, arms and hair can form
        # different motions. Require a supported local motion, not a majority
        # rigid fit, after the per-feature round-trip/appearance checks above.
        if affine is None or inliers.sum()<20 or inliers.mean()<.3:break
        tracked=q.copy()
        p,q=p[inliers[:,0]>0],q[inliers[:,0]>0]
        if len(p)<10 or np.any(np.ptp(p,axis=0)<(current[2:]-current[:2])*.12):break
        scale=np.linalg.norm(affine[:,0])
        if not .85<scale<1.18:break
        transform=np.vstack([affine,[0,0,1]])@transform
        moved=corners@transform[:2,:2].T+transform[:2,2]
        current=np.r_[moved.min(0),moved.max(0)].astype(np.float32)
        current[[0,2]]=np.clip(current[[0,2]],0,w);current[[1,3]]=np.clip(current[[1,3]],0,h)
        out[frame]=(current.copy(),int(len(q)))
        # Keep all individually verified features for the next frame. Discarding
        # non-rigid limbs at every step would exhaust an otherwise valid trace.
        points=tracked[:,None].astype(np.float32)
    return out


def recover_shot_gaps(images, boxes, valid, *, max_gap=15, max_edge=5):
    """Pure single-shot boundary; images and boxes share the same pixel scale."""
    output=boxes.copy();mask=valid.copy();report=[]
    for a,b in intervals(~valid):
        left=a>0;right=b<len(valid)
        if not (left or right) or b-a>max_gap or (not(left and right) and b-a>max_edge):continue
        forward=_trace(images,a-1,b-1,boxes[a-1]) if left else {}
        backward=_trace(images,b,a,boxes[b]) if right else {}
        for frame in range(a,b):
            f,r=forward.get(frame),backward.get(frame)
            if left and right:
                if f is None or r is None or overlap(f[0],r[0])<.55:continue
                weight=(frame-a+1)/(b-a+1);box=f[0]*(1-weight)+r[0]*weight
                evidence=dict(forward_points=f[1],backward_points=r[1],agreement=overlap(f[0],r[0]))
            else:
                found=f or r
                if found is None:continue
                box=found[0];evidence=dict(edge=True,tracked_points=found[1])
            output[frame]=box;mask[frame]=True
            report.append(dict(frame=frame,method='image_features',**evidence))
    return output,mask,report


def recover_video_gaps(video, boxes, valid, shots, check=lambda:None, progress=lambda *x:None):
    from flowhmr.utils.runtime_video_processing import open_video_at_frame
    first=shots[0]['start'];output=boxes.copy();mask=valid.copy();report=[]
    for index,shot in enumerate(shots):
        check();a,b=shot['start']-first,shot['end']-first
        if valid[a:b].all() or not valid[a:b].any():continue
        # Decode only bounded gaps and their endpoints; no full-shot image cache.
        for left,right in intervals(~valid[a:b]):
            if right-left>15 or ((left==0 or right==b-a) and right-left>5):continue
            start=max(left-1,0);stop=min(right+1,b-a)
            cap=open_video_at_frame(str(video),first+a+start,check);images=[]
            try:
                for frame in range(start,stop):
                    check();ok,im=cap.read()
                    if not ok:raise ValueError('人物复核未能完整读取视频。')
                    scale=min(1.,960/im.shape[1])
                    images.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None,fx=scale,fy=scale))
            finally:cap.release()
            recovered,visible,evidence=recover_shot_gaps(images,boxes[a+start:a+stop]*scale,valid[a+start:a+stop])
            newly=visible&~valid[a+start:a+stop]
            output[a+start:a+stop][newly]=recovered[newly]/scale
            mask[a+start:a+stop]|=visible
            report.extend(dict(**{k:v for k,v in e.items() if k!='frame'},source_frame=first+a+start+e['frame'],shot=shot['id']) for e in evidence)
        progress(index+1,len(shots))
    return output,mask,report


def redetect_gaps(detector, video, boxes, valid, shots, check=lambda:None, progress=lambda *x:None):
    """Recheck bounded gaps in context; retain uncertain candidates as evidence."""
    from flowhmr.utils.runtime_video_processing import open_video_at_frame
    first=shots[0]['start'];output=boxes.copy();mask=valid.copy();report=[]
    for index,shot in enumerate(shots):
        a,b=shot['start']-first,shot['end']-first
        timeline=np.arange(a,b);known=timeline[valid[a:b]]
        if not len(known):continue
        expected=np.array([np.interp(timeline,known,boxes[known,k]) for k in range(4)]).T
        for left,right in intervals(~valid[a:b]):
            # Direct per-frame human evidence also supports a short entry gap.
            # The stricter five-frame edge limit applies to optical tracking,
            # which has only one observed endpoint, not to person detection.
            if right-left>15:continue
            cap=open_video_at_frame(str(video),first+a+left,check)
            try:
                for local in range(left,right):
                    check();ok,im=cap.read()
                    if not ok:raise ValueError('人物复核未能完整读取视频。')
                    target=expected[local];center=(target[:2]+target[2:])/2;size=target[2:]-target[:2]
                    lo=np.maximum(np.floor(center-size*.85),0).astype(int)
                    hi=np.minimum(np.ceil(center+size*.85),[im.shape[1],im.shape[0]]).astype(int)
                    full=detector.run_human_detection(im,bbox_thr=.05,default_to_full_image=False)
                    candidates=[] if full is None else full.tolist()
                    if np.all(hi>lo):
                        cropped=detector.run_human_detection(im[lo[1]:hi[1],lo[0]:hi[0]],bbox_thr=.05,default_to_full_image=False)
                        if cropped is not None:
                            cropped[:,:4]+=np.tile(lo,2);candidates.extend(cropped.tolist())
                    found=accepted_candidate(candidates,target)
                    report.append(dict(source_frame=first+a+local,shot=shot['id'],candidates=candidates,
                                       accepted=found is not None,method='context_person_detection'))
                    if found is not None:
                        # Weak detections can latch onto a limb or hair. Use them
                        # to verify presence, then stabilize the crop from the
                        # neighboring measured boxes in this same shot.
                        output[a+local]=target;mask[a+local]=True
            finally:cap.release()
        progress(index+1,len(shots))
    return output,mask,report
