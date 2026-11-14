"""Private arm-fit projection; never exported as the recovered camera."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .gauge import ry, degrees
from .arm_geometry import SM, MH, project, polar


def unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def horizontal_swing(source, target):
    """Shortest rotation mapping two up vectors, constrained to a horizontal axis."""
    delta = target - source
    axis = np.stack([delta[..., 2], np.zeros(delta.shape[:-1]), -delta[..., 0]], -1)
    length = np.linalg.norm(axis, axis=-1)
    axis = unit(axis)
    sine = np.sum(axis * np.cross(source, target), axis=-1)
    cosine = np.sum(source * target, axis=-1) - np.sum(axis * source, axis=-1) * np.sum(axis * target, axis=-1)
    angle = np.arctan2(sine, cosine)
    angle = np.where(length > 1e-10, angle, 0.)
    matrix = Rotation.from_rotvec((axis * angle[..., None]).reshape(-1, 3)).as_matrix()
    return matrix.reshape(*source.shape[:-1], 3, 3)


def continuous_tilt(condition, body_camera, shots, *, fps=30., sigma_seconds=5., support_seconds=5.):
    """Carry body lean across cuts, remove only a long, continuous local mean.

    The Gaussian has sigma=5 s and explicit support +/-5 s (truncate=1),
    rather than silently using scipy's default +/-20 s. It never crosses a
    person gap. Only recovery sees this condition; Flow inputs stay untouched.
    """
    condition = Rotation.from_matrix(condition).as_matrix()
    up = np.einsum('tji,tj->ti', condition, body_camera[:, :, 1])
    up = unit(up)
    if np.min(up[:, 1]) <= .1:
        raise ValueError('SAM up-axis cannot be represented by the upright tangent chart')
    lean = np.arctan2(up[:, [0, 2]], up[:, 1, None])
    accumulated = np.zeros_like(lean)
    previous = None
    cuts = []
    for shot in shots:
        a, b = shot['start'], shot['end']
        count = min(6, b-a)
        if previous is None:
            offset = np.zeros(2)
        else:
            pa, pb = previous
            offset = accumulated[max(pa, pb-6):pb].mean(0) - lean[a:a+count].mean(0)
            cuts.append(dict(frame=a,raw_lean_step_deg=np.degrees(lean[a:a+count].mean(0)-lean[max(pa,pb-6):pb].mean(0))))
        accumulated[a:b] = lean[a:b] + offset
        previous = (a, b)
    slow = gaussian_filter1d(accumulated, sigma_seconds*fps, axis=0, mode='nearest',
                             truncate=support_seconds/sigma_seconds)
    target_lean = accumulated - slow
    target = unit(np.stack([np.tan(target_lean[:, 0]), np.ones(len(up)), np.tan(target_lean[:, 1])], -1))
    adjustment = horizontal_swing(up, target)
    revised = condition @ adjustment.transpose(0, 2, 1)
    reconstructed = np.einsum('tji,tj->ti', revised, body_camera[:, :, 1])
    assert np.max(abs(unit(reconstructed)-target)) < 1e-7
    for row, left, right in zip(cuts, shots[:-1], shots[1:]):
        c = right['start']
        row['continuous_lean_step_deg'] = np.degrees(target_lean[c:min(c+6,right['end'])].mean(0)-target_lean[max(left['start'],c-6):c].mean(0))
    return revised, dict(sigma_seconds=sigma_seconds,support_seconds_each_side=support_seconds,
                         edge_frames=6,max_recovery_tilt_deg=float(degrees(adjustment).max()),
                         correction_axis_y_max=float(np.max(abs(Rotation.from_matrix(adjustment).as_rotvec()[:, 1]))),
                         cuts=cuts,applied_to_generation=False)


def constrain_camera_height(rotation, world_centroid, camera_centroid, *, minimum_height=.05):
    """Minimum constant swing per shot, orbiting that frame's person centroid.

    R' = R A.T, C' = mx + A(C-mx). Thus R'(mx-C') = my exactly;
    raising the viewpoint does not translate the person's anchor in the image.
    """
    vector = -np.einsum('tji,tj->ti', rotation, camera_centroid)
    before = world_centroid + vector
    if before[:, 1].min() >= minimum_height:
        return rotation.copy(), dict(applied=False,angle_deg=0.,height_min_before=float(before[:,1].min()),height_min_after=float(before[:,1].min()))
    axis = unit(np.cross(vector.mean(0), [0.,1.,0.]))
    def swing(x):
        return Rotation.from_rotvec([x[0], 0., x[1]]).as_matrix()
    def margin(x):
        return (world_centroid + vector @ swing(x).T)[:,1] - minimum_height
    initial = np.zeros(2)
    for angle in np.linspace(.001, np.deg2rad(30), 200):
        initial = axis[[0,2]] * angle
        if margin(initial).min() >= 0:break
    else:
        raise ValueError('Ground constraint would require more than 30 degrees of camera tilt')
    solution = minimize(lambda x: float(x@x), initial, method='SLSQP',
                        constraints=[dict(type='ineq',fun=margin)],
                        options=dict(ftol=1e-13,maxiter=100))
    if not solution.success or margin(solution.x).min() < -1e-7:
        raise RuntimeError('Minimum camera-height swing did not converge: '+solution.message)
    adjustment = swing(solution.x)
    revised = rotation @ adjustment.T
    after = world_centroid - np.einsum('tji,tj->ti',revised,camera_centroid)
    anchor = np.einsum('tij,tj->ti',revised,world_centroid-after)
    assert np.max(abs(anchor-camera_centroid)) < 1e-8
    return revised, dict(applied=True,angle_deg=float(np.degrees(np.linalg.norm(solution.x))),
                         world_axis_rotvec=solution.x.tolist(),height_min_before=float(before[:,1].min()),
                         height_min_after=float(after[:,1].min()),anchor_camera_max_error_m=float(np.max(abs(anchor-camera_centroid))))


def recover_stable(joints, observations, K, condition, body_camera, shots, image_size_hw):
    X = joints[:,SM].astype(float)
    Y = observations[:,MH].astype(float)
    uv = project(Y,K)
    height,width = image_size_hw
    inside = ((uv[...,0]>=0)&(uv[...,0]<width)&(uv[...,1]>=0)&(uv[...,1]<height)&(Y[...,2]>.05))
    weight = np.zeros(inside.shape,float)
    for shot in shots:
        a,b = shot['start'],shot['end']
        weight[a:b] = .05 + gaussian_filter1d(inside[a:b].astype(float),2.,axis=0,mode='nearest')
    weight /= weight.sum(1,keepdims=True)
    mx = np.sum(weight[...,None]*X,axis=1)
    my = np.sum(weight[...,None]*Y,axis=1)
    distance = np.linalg.norm(my,axis=1)
    ray = unit(my)
    smooth_distance = distance.copy()
    for shot in shots:
        a,b = shot['start'],shot['end']
        smooth_distance[a:b] = np.exp(gaussian_filter1d(np.log(distance[a:b]),4.,mode='nearest'))
    my_stable = ray*smooth_distance[:,None]
    assert np.max(abs(unit(my_stable)-ray)) < 1e-12
    recovered_condition,tilt_report = continuous_tilt(condition,body_camera,shots)
    xc = X-mx[:,None]
    yc = Y-my[:,None]
    target = np.einsum('tji,tkj->tki',recovered_condition,yc)
    # Both worlds now explicitly share gravity. Fit one yaw for the whole
    # interval, rather than allowing a free mapping to tilt the ground again.
    sine = np.sum(weight*(xc[...,2]*target[...,0]-xc[...,0]*target[...,2]))
    cosine = np.sum(weight*(xc[...,0]*target[...,0]+xc[...,2]*target[...,2]))
    mapping = ry(np.arctan2(sine,cosine))
    free_mapping = polar(np.einsum('tki,tkj,tk->ij',xc,target,weight))
    rotation = recovered_condition@mapping
    floor_rows = []
    for shot in shots:
        a,b = shot['start'],shot['end']
        rotation[a:b],row = constrain_camera_height(rotation[a:b],mx[a:b],my_stable[a:b])
        floor_rows.append(dict(source_frames=[shot['source_start'],shot['source_end']],**row))
    translation = my_stable-np.einsum('tij,tj->ti',rotation,mx)
    rt = np.tile(np.eye(4),(len(X),1,1));rt[:,:3,:3]=rotation;rt[:,:3,3]=translation
    centre = -np.einsum('tji,tj->ti',rotation,translation)
    assert centre[:,1].min() >= .05-1e-7
    report = dict(range_sigma_frames=4.,weight_sigma_frames=2.,range_ray_max_change=float(np.max(abs(unit(my_stable)-ray))),
                  tilt=tilt_report,ground_constraints=floor_rows,minimum_camera_height_m=.05,
                  physical_floor_y=0.,render_floor_y=-.015,mapping= mapping,
                  mapping_scope='one gravity-preserving yaw mapping per continuous interval; not per shot',
                  diagnostic_free_mapping=free_mapping,free_mapping_used=False,
                  camera_height_min=float(centre[:,1].min()),motion_modified=False)
    arrays=dict(weights=weight,inside=inside,world_centroid=mx,camera_centroid=my,
                stabilized_camera_centroid=my_stable,range_before=distance,range_after=smooth_distance,
                recovered_condition=recovered_condition,camera_centre=centre)
    return rt,arrays,report
