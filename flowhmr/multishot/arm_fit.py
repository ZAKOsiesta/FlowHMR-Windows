"""Bounded low-frequency upper-limb offsets with explicit dynamic constraints."""
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.spatial.transform import Rotation

EDIT=[13,14,16,17,18,19]
SM=[16,17,18,19,20,21]
MH=[5,6,7,8,62,41]


def fit_interval(motion,baseline_joints,observations,camera,shots, *, check=lambda: None, notify=lambda step: None):
    import torch
    from .arm_geometry import project
    poses=motion['poses'];n=len(poses);parents=motion['parents']
    matrix=Rotation.from_rotvec(poses.reshape(-1,3)).as_matrix().reshape(n,52,3,3)
    target=project(observations['points'][:,MH],camera['K'])
    h,w=camera['image_size_hw']
    visible=((target[...,0]>=0)&(target[...,0]<w)&(target[...,1]>=0)&(target[...,1]<h)).astype(float)
    for s in shots:
        a,b=s['start'],s['end']
        target[a:b]=gaussian_filter1d(median_filter(target[a:b],size=(5,1,1),mode='nearest'),2.,axis=0,mode='nearest')
        visible[a:b]=gaussian_filter1d(visible[a:b],2.,axis=0,mode='nearest')
    box_height=np.maximum(observations['boxes'][:,3]-observations['boxes'][:,1],40.)
    to=lambda x:torch.as_tensor(np.array(x),dtype=torch.float32)
    original=to(matrix);rest=to(motion['rest']);root=to(motion['trans']+motion['rest'][0])
    rt,translation,K,goal,bh,weight=map(to,[camera['RT'][:,:3,:3],camera['RT'][:,:3,3],camera['K'],target,box_height,visible])
    # Cosine basis frequencies k*fps/(2*n), including the constant offset.
    count=int(np.floor(2*2.*n/30))+1
    basis=to(np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(count)[None,:]/n))
    coefficients=torch.zeros((count,len(EDIT),3),requires_grad=True)
    limit=float(np.deg2rad(20.))
    def correction():
        raw=torch.einsum('tk,kjc->tjc',basis,coefficients)
        largest=torch.linalg.vector_norm(raw,dim=-1).max(dim=0).values.clamp_min(1e-8)
        return raw*torch.clamp(limit/largest,max=1)[None,:,None]
    def fk(delta):
        local=list(original.unbind(1));patch=expmat(delta)
        for k,j in enumerate(EDIT):local[j]=patch[:,k]@local[j]
        world=[];position=[]
        for j in range(52):
            p=int(parents[j])
            if p<0:world.append(local[j]);position.append(root)
            else:
                world.append(world[p]@local[j])
                position.append(position[p]+torch.einsum('tij,j->ti',world[p],rest[j]-rest[p]))
        return torch.stack(position,1)
    def metric_coordinates(joints):
        root_local=torch.einsum('tji,tkj->tki',original[:,0],joints-joints[:,:1])
        shoulder=(joints[:,16]+joints[:,17])/2;hip=(joints[:,1]+joints[:,2])/2
        up=torch.nn.functional.normalize(shoulder-hip,dim=-1)
        right=(joints[:,17]-joints[:,16])+(joints[:,2]-joints[:,1])
        right=torch.nn.functional.normalize(right-(right*up).sum(-1,keepdim=True)*up,dim=-1)
        axes=torch.stack([right,up,torch.linalg.cross(right,up)],-1)
        arm=torch.einsum('tji,tkj->tki',axes,joints[:,[18,19,20,21]]-shoulder[:,None])
        return torch.cat([root_local,arm],dim=1)
    inside=np.ones(n,bool);inside[:2]=False;inside[-2:]=False
    for s in shots:
        c=s['start'];inside[max(0,c-2):c+2]=False
    if not inside[1:-1].any():
        return poses.copy(), dict(skipped="short_interval", edited_joints=[])
    selected=torch.tensor(inside[1:-1]);quantiles=to([.5,.95,.99])
    def dynamic_quantiles(joints):
        x=metric_coordinates(joints)
        d2=torch.linalg.vector_norm(x[2:]-2*x[1:-1]+x[:-2],dim=-1)
        return torch.quantile(d2[selected],quantiles,dim=0)
    with torch.no_grad():
        original_fk=fk(torch.zeros((n,len(EDIT),3)))
        fk_error=float(torch.max(abs(original_fk-to(baseline_joints))))
        assert fk_error<2e-5,fk_error
        limits=dynamic_quantiles(original_fk)
    tolerance=1e-5  # 0.01 mm, below decoded float32 differences; not a visual tolerance.
    optimizer=torch.optim.Adam([coefficients],lr=.0015)
    records=[]
    for step in range(620):
        check()
        if step % 20 == 0: notify(step)
        if step==250:optimizer.param_groups[0]['lr']=.00015
        if step==450:optimizer.param_groups[0]['lr']=.00003
        optimizer.zero_grad();delta=correction();joints=fk(delta)
        q=torch.einsum('tij,tkj->tki',rt,joints[:,SM])+translation[:,None]
        pixels=torch.einsum('tij,tkj->tki',K,q);uv=pixels[...,:2]/pixels[...,2:].clamp_min(.05)
        residual=(uv-goal)/bh[:,None,None]/.04
        image=torch.nn.functional.huber_loss(residual,torch.zeros_like(residual),reduction='none',delta=1.).sum(-1)
        fit=(image*weight).sum()/weight.sum().clamp_min(1.)
        prior=.04*(delta/limit).square().mean()
        measured=dynamic_quantiles(joints)
        violation=torch.relu(measured-limits-tolerance)/(limits+.001)
        penalty=20. if step<100 else (300. if step<250 else 3000.)
        loss=fit+prior+penalty*violation.square().mean()
        loss.backward();optimizer.step()
        if step in [0,99,249,449,619]:
            records.append(dict(step=step,image_loss=float(fit.detach()),max_quantile_increase_cm=float(torch.max(measured-limits).detach())*100))
    with torch.no_grad():
        fitted=correction().detach()
        accepted=None
        # A deterministic feasibility backtrack, not a second generated motion.
        for scale in [1.,.75,.5,.25,.125,.0625,.03125,0.]:
            values=dynamic_quantiles(fk(fitted*scale))
            if bool(torch.all(values<=limits+tolerance)):
                accepted=scale;break
        assert accepted is not None
        changes=(fitted*accepted).numpy();after=dynamic_quantiles(fk(fitted*accepted)).numpy()
    result=poses.copy()
    for k,j in enumerate(EDIT):
        if accepted>0:result[:,j]=Rotation.from_matrix(Rotation.from_rotvec(changes[:,k]).as_matrix()@matrix[:,j]).as_rotvec()
    report=dict(frames=n,edited_joints=EDIT,cutoff_hz=2.,basis_terms=count,
                target_median_frames=5,target_gaussian_sigma_frames=2.,max_correction_deg=float(np.rad2deg(np.linalg.norm(changes,axis=-1)).max()),
                iterations=620,accepted_feasibility_scale=accepted,fk_error_m=fk_error,history=records,
                quantiles=[50,95,99],metric='52 joint root-local coordinates plus 4 own-torso-frame arm joints; exclude +/-2 cut frames',
                baseline_quantiles_cm=(limits.numpy()*100).tolist(),final_quantiles_cm=(after*100).tolist(),
                float_tolerance_cm=tolerance*100,full_body_filter_added=False,flow_calls=0)
    return result,report


def expmat(v):
    import torch
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*v.shape[:-1],3,3)
    angle=torch.linalg.vector_norm(v,dim=-1)
    a=torch.sinc(angle/torch.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*torch.pi))**2)[...,None,None]
    return torch.eye(3,device=v.device)+a*skew+b*(skew@skew)
