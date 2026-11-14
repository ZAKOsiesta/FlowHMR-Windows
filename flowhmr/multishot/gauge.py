"""Camera gauge from SAM global orientation; no generated-motion input.

Matrices are column-vector world-to-OpenCV-camera rotations. Y is world up.
Only yaw is transferred across shots. A shot's tilt is independently estimated
from its long-time mean body up, never from the preceding shot's tilt.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def mean_rotation(r):
    u,_,vt=np.linalg.svd(np.sum(r,axis=0))
    return u @ np.diag([1,1,np.linalg.det(u@vt)]) @ vt


def degrees(r):
    return np.degrees(Rotation.from_matrix(np.asarray(r).reshape(-1,3,3)).magnitude()).reshape(np.asarray(r).shape[:-2])


def yaw(r):
    # Closest rotation about Y, also well defined during modest body lean.
    return np.arctan2(r[...,0,2]-r[...,2,0],r[...,0,0]+r[...,2,2])


def ry(angle):
    return Rotation.from_rotvec(np.asarray(angle)[...,None]*np.array([0.,1.,0.])).as_matrix()


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


def linear_heading(matrices, times, target):
    angles=np.unwrap(yaw(matrices))
    x=np.asarray(times,dtype=float)-target
    slope,intercept=np.linalg.lstsq(np.stack([x,np.ones(len(x))],1),angles,rcond=None)[0]
    residual=angles-(slope*x+intercept)
    return intercept,slope,float(np.sqrt(np.mean(residual**2)))


def build_condition(camera_R, body_camera, shots, *, fps=30., source_start=0,
                    edge_frames=6, max_camera_speed_deg_s=180., jitter_rms_deg_limit=1.5,
                    source_frames=None):
    n=len(body_camera)
    times=np.arange(n)+source_start if source_frames is None else np.asarray(source_frames)
    if camera_R.shape!=(n,3,3) or body_camera.shape!=(n,3,3): raise ValueError('Invalid rotation arrays')
    result=np.empty_like(camera_R,dtype=float)
    shot_report=[];cut_report=[];previous=None
    for shot in shots:
        a,b=int(shot['start']),int(shot['end'])
        raw=Rotation.from_matrix(camera_R[a:b]).as_matrix()
        relative=raw@raw[0].T
        elapsed=np.diff(times[a:b])
        if np.any(elapsed<=0): raise ValueError('Source frames must increase')
        delta=degrees(relative[1:]@relative[:-1].transpose(0,2,1))*fps/elapsed
        # High-frequency angular excursion from a 7-frame-scale low pass.
        vectors=Rotation.from_matrix(relative).as_rotvec()
        smooth=gaussian_filter1d(vectors,2.,axis=0,mode='nearest')
        jitter=float(np.sqrt(np.mean(degrees(relative@Rotation.from_rotvec(smooth).as_matrix().transpose(0,2,1))**2)))
        max_speed=float(delta.max()) if len(delta) else 0.
        reasons=[]
        if max_speed>max_camera_speed_deg_s: reasons.append('angular_speed')
        if jitter>jitter_rms_deg_limit: reasons.append('high_frequency_rotation')
        if reasons: relative=np.tile(np.eye(3),(b-a,1,1))
        body_local=relative.transpose(0,2,1)@body_camera[a:b]
        # Whole-shot time average, not a per-frame upright constraint. For short
        # shots this remains uncertain (reported); individual dance lean stays.
        up=body_local[:,:,1].mean(0);up/=np.linalg.norm(up)
        forward=np.array([0.,0.,1.]);forward-=up*np.dot(up,forward)
        if np.linalg.norm(forward)<1e-6:
            forward=np.array([1.,0.,0.]);forward-=up*np.dot(up,forward)
        forward/=np.linalg.norm(forward)
        right=np.cross(up,forward)
        reference=np.stack([right,up,forward],axis=1)
        base=relative@reference
        implied=base.transpose(0,2,1)@body_camera[a:b]
        alpha=0.
        if previous is not None:
            c=(times[a]+times[a-1])/2
            lp=max(previous['start'],a-edge_frames);rp=min(b,a+edge_frames)
            left=result[lp:a].transpose(0,2,1)@body_camera[lp:a]
            lvalue,lslope,lnoise=linear_heading(left,times[lp:a],c)
            rvalue,rslope,rnoise=linear_heading(implied[:rp-a],times[a:rp],c)
            gap=times[a]>times[a-1]+1
            if gap:
                lvalue=float(yaw(left[-1]));rvalue=float(yaw(implied[0]))
            alpha=wrap(rvalue-lvalue)
            cut_report.append(dict(source_frame=int(times[a]),edge_frames_left=a-lp,edge_frames_right=rp-a,
                across_gap=bool(gap),continuity_basis='held_heading' if gap else 'linear_cut_extrapolation',
                camera_yaw_gauge_deg=float(np.degrees(alpha)),
                extrapolation_residual_deg=float(np.degrees(wrap(rvalue-alpha-lvalue))),
                left_velocity_deg_s=float(np.degrees(lslope)*fps),right_velocity_deg_s=float(np.degrees(rslope)*fps),
                velocity_disagreement_deg_s=float(np.degrees(rslope-lslope)*fps),
                left_fit_rms_deg=float(np.degrees(lnoise)),right_fit_rms_deg=float(np.degrees(rnoise))))
        result[a:b]=base@ry(alpha)
        world=result[a:b].transpose(0,2,1)@body_camera[a:b]
        upworld=world[:,:,1].mean(0);upworld/=np.linalg.norm(upworld)
        speed=degrees(result[a+1:b]@result[a:b-1].transpose(0,2,1))*fps/elapsed
        shot_report.append(dict(shot_id=shot.get('id'),source_frames=[int(times[a]),int(times[b-1])+1],
            raw_camera_peak_deg_s=max_speed,raw_camera_high_frequency_rms_deg=jitter,
            static_fallback=bool(reasons),static_reasons=reasons,leveling_window_frames=b-a,
            short_leveling_window=(b-a)<round(fps),
            world_mean_up_deviation_deg=float(np.degrees(np.arccos(np.clip(upworld[1],-1,1)))),
            actual_camera_peak_deg_s=float(speed.max()) if len(speed) else 0.,
            body_lean_p50_deg=float(np.median(np.degrees(np.arccos(np.clip(world[:,1,1],-1,1)))))))
        previous=shot
    if not np.isfinite(result).all() or not np.allclose(np.linalg.det(result),1,atol=1e-5):
        raise ValueError('Camera gauge is not a finite proper rotation')
    return result,dict(shots=shot_report,cuts=cut_report,parameters=dict(edge_frames=edge_frames,
        camera_speed_limit_deg_s=max_camera_speed_deg_s,jitter_rms_limit_deg=jitter_rms_deg_limit,
        threshold_basis='Conservative dance-camera operating limits, not universal physical impossibility; raw and triggered shots retained',
        leveling='one whole-shot time-mean SAM up; no transfer of pitch/roll; no per-frame body correction',
        continuity='first-order (constant angular velocity) extrapolation of horizontal heading at cut midpoint; slope disagreement reported, not forced'))


def linked_condition(camera_R, body_camera, valid, shots, breaks=(), *, source_start=0):
    """Keep a shot's gauge through absence; break only explicitly rejected cuts.

    Missing frames are excluded from the shot mean. Absolute frame times retain
    the timing of observations; a cut across absence uses held heading instead
    of claiming that angular velocity was observed inside the gap.
    """
    indices=np.flatnonzero(valid)
    result=np.tile(np.eye(3),(len(valid),1,1))
    boundaries=[0,*sorted(set(int(x) for x in breaks if 0<x<len(valid))),len(valid)]
    reports=[]
    for first,stop in zip(boundaries,boundaries[1:]):
        selected=indices[(indices>=first)&(indices<stop)]
        if not len(selected):continue
        local=[]
        for shot in shots:
            a=int(np.searchsorted(selected,max(first,shot['start'])))
            b=int(np.searchsorted(selected,min(stop,shot['end'])))
            if a<b:local.append(dict(id=shot['id'],start=a,end=b))
        rotation,report=build_condition(camera_R[selected],body_camera[selected],local,
                                       source_frames=selected+source_start)
        result[selected]=rotation
        reports.append(dict(source_frames=[int(selected[0])+source_start,int(selected[-1])+source_start+1],**report))
    return result,reports
