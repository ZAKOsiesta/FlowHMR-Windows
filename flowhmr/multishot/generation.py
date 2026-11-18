"""Multi-shot orchestration over the existing engine and native postprocessing.

This adapter owns jobs, model lifecycles and output files. Gauge calculations
remain array operations; short-gap visibility checks use source image evidence.
"""
import json
import copy
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from .runtime import array_digest, digest, save_json, save_npz
from .shots import clip_shots
from .linking import shot_candidates, link_tracks, continuity_breaks, filter_short_segments
from .timeline import intervals, held_indices


def copy_input_cache(source, destination):
    """Reuse observations, never the previous generated motion or final camera."""
    import shutil
    destination.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and (path.name in ('tracks.json','selection.json') or path.name.startswith(('static_', 'vggt_'))):
            shutil.copy2(path,destination/path.name)
        elif path.is_dir() and path.name.startswith('run_'):
            target = destination/path.name
            target.mkdir(exist_ok=True)
            for name in ('features.json','tokens.pt','observations.npz','hands.npy'):
                if (path/name).is_file():
                    shutil.copy2(path/name,target/name)


def _tracks(engine, video, folder, shots, check, progress):
    from flowhmr.pipeline.v2m_inference import StageName
    from flowhmr.utils.runtime_video_processing import TrackingResult, TrackInfo
    path = folder/'tracks.json'
    identity = dict(video=digest(video), first=shots[0]['start'], stop=shots[-1]['end'])
    try:
        saved = json.loads(path.read_text(encoding='utf8'))
        if saved['identity'] == identity:
            data = saved['result']
            if tuple(data['scanned_range']) != (identity['first'],identity['stop']-1):
                raise ValueError('Incomplete tracking cache')
            data['tracks'] = [TrackInfo(t['track_id'], np.asarray(t['frames']),
                np.asarray(t['bboxes'], np.float32), t['length'], t['avg_area']) for t in data['tracks']]
            return TrackingResult(**data)
    except (OSError, ValueError, KeyError):
        pass
    engine._activate_stage(StageName.DETECTION)
    try:
        result = engine.proc.detect_and_track(str(video), start_frame=identity['first'],
            end_frame=identity['stop']-1, cancel_check=check,
            on_progress=lambda done, total: progress(StageName.DETECTION,
                min(65, 10+55*done/max(total or 1, 1)), f'检测剪辑视频人物 · {done}/{total} 帧'))
    finally:
        engine.release_stage()
    if tuple(result.scanned_range) != (identity['first'],identity['stop']-1):
        raise ValueError('视频未能完整读取所选区间，请释放内存后重试；未保存不完整的人物跟踪。')
    save_json(path, dict(identity=identity, result={**vars(result), 'tracks':[vars(t) for t in result.tracks]}))
    return result


def _select(result, shots, selector):
    from flowhmr.utils.runtime_video_processing import VideoProcessor
    choices = shot_candidates(result, shots)
    if selector is None or not any(len(c) > 1 for c in choices):
        return None, False
    offered = sorted({c['track_index'] for row in choices for c in row})
    candidates = []
    for i in offered:
        t = result.tracks[i]
        order = np.argsort(t.frames)
        candidates.append(dict(index=len(candidates), track_index=i, track_id=int(t.track_id),
            start=int(t.frames.min()), end=int(t.frames.max()), preview_start=int(t.frames.min()),
            preview_end=int(t.frames.max()), num_frames=int(t.length), avg_area=float(t.avg_area),
            frames=t.frames[order].astype(int).tolist(), boxes=t.bboxes[order].tolist()))
    best = VideoProcessor.select_best_track(result)
    default = offered.index(best) if best in offered else 0
    chosen = selector(result, candidates, default)
    chosen = chosen if isinstance(chosen, (list, tuple)) else [default if chosen is None else chosen]
    if not chosen or any(type(i) is not int or not 0 <= i < len(offered) for i in chosen):
        chosen = [default]
    return offered[chosen[0]], len(chosen) > 1


def _recover_visibility(engine, video, folder, boxes, valid, shots, check, progress):
    from flowhmr.pipeline.v2m_inference import StageName
    from .visibility import redetect_gaps, recover_video_gaps
    identity=dict(version=2,video=digest(video),boxes=array_digest(boxes),valid=array_digest(valid),shots=shots)
    proof=folder/'visibility.json';data=folder/'visibility.npz'
    try:
        saved=json.loads(proof.read_text(encoding='utf8'))
        if saved['identity']==identity and saved['sha256']==digest(data):
            with np.load(data) as z:return z['boxes'],z['valid'],saved['report']
    except (OSError,ValueError,KeyError):pass
    before=int((~valid).sum())
    if before:
        engine._activate_stage(StageName.DETECTION)
        try:
            boxes,valid,detected=redetect_gaps(engine.proc.detector,video,boxes,valid,shots,check,
                lambda i,n:progress(StageName.DETECTION,65+25*i/n,f'复核人物可见性 · {i}/{n} 个镜头'))
        finally:engine.release_stage()
        boxes,valid,tracked=recover_video_gaps(video,boxes,valid,shots,check,
            lambda i,n:progress(StageName.DETECTION,90+10*i/n,f'核对相邻画面 · {i}/{n} 个镜头'))
    else:detected,tracked=[],[]
    report=dict(missing_before=before,missing_after=int((~valid).sum()),detections=detected,image_features=tracked)
    check();save_npz(data,boxes=boxes,valid=valid)
    save_json(proof,dict(identity=identity,sha256=digest(data),report=report))
    return boxes,valid,report


def _cameras(engine, video, folder, shots, hw, static, interval, check, progress):
    from flowhmr.pipeline.v2m_inference import StageName
    from flowhmr.utils.runtime_vggt_camera import make_static_camera, extract_vggt_camera_isolated
    first, stop = shots[0]['start'], shots[-1]['end']
    RT = np.tile(np.eye(4, dtype=np.float32), (stop-first, 1, 1))
    K = np.empty((stop-first, 3, 3), np.float32)
    for index, shot in enumerate(shots):
        check()
        a, b = shot['start'], shot['end']
        path = folder/f"{'static' if static else 'vggt'}_{a}_{b-1}.npz"
        progress(StageName.CAMERA, 100*index/len(shots), f'镜头条件 · {index+1}/{len(shots)}')
        if static:
            camera = make_static_camera(hw, frame_range=(a,b-1), output_npz=str(path))
        else:
            camera = extract_vggt_camera_isolated(str(video), output_npz=str(path),
                frame_range=(a,b-1), expected_len=b-a, device=engine.device, frame_interval=interval,
                cancel_check=check, release_cuda=engine._release_camera_cuda,
                on_progress=lambda message: progress(StageName.CAMERA, 100*index/len(shots), message))
        RT[a-first:b-first], K[a-first:b-first] = camera.camera_RT, camera.K
    return RT, K


def _features(engine, video, folder, boxes, K, first, stop, *, chunk, stride, check, progress, details):
    import torch
    from flowhmr.pipeline.v2m_inference import StageName
    identity = dict(video=digest(video), boxes=array_digest(boxes), K=array_digest(K),
                    first=first, stop=stop, stride=stride, backend=engine.sam_cache_backend,
                    observations_version=1, chunk=chunk)
    proof, token_path, obs_path, hands_path = (folder/name for name in
        ('features.json', 'tokens.pt', 'observations.npz', 'hands.npy'))
    try:
        meta = json.loads(proof.read_text(encoding='utf8'))
        if meta['identity'] == identity and all(digest(folder/name) == value for name,value in meta['files'].items()):
            tokens = torch.load(token_path, map_location='cpu', weights_only=True)
            with np.load(obs_path, allow_pickle=False) as z:
                observations = dict(z)
            if tokens.shape == (stop-first,3072) and observations['points'].shape == (stop-first,70,3):
                details(sam_batch=dict(requested=chunk, reused=True, backend=engine.sam_cache_backend))
                return tokens, observations, str(hands_path)
    except (OSError, ValueError, KeyError, RuntimeError):
        pass
    engine._activate_stage(StageName.FEATURE_EXTRACT)
    chunks, hands = [], []
    try:
        tokens = engine.sam_extractor.extract_video_tokens(str(video),
            bbox_xyxy=torch.from_numpy(boxes), K_all=torch.from_numpy(K), start_frame=first,
            max_frames=stop, frame_batch_size=chunk or 16, adaptive_batch=chunk is None,
            sam_hand_stride=stride, hand_boxes_out=hands, observations_out=chunks,
            batch_cb=lambda info: details(sam_batch={**info, 'requested':chunk, 'reused':False,
                                                   'backend':engine.sam_cache_backend}),
            progress_cb=lambda stage, done, total, message: (check(), progress(StageName.FEATURE_EXTRACT,
                100*done/max(total or 1,1), f'人物特征 · {done}/{total} 帧')))
    finally:
        engine.release_stage()
    if not chunks or not hands or tokens.shape != (stop-first,3072):
        raise ValueError('SAM 没有返回完整的同次人物观测，未保存不完整结果。')
    observations = {key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    if observations['points'].shape != (stop-first,70,3) or not all(np.isfinite(v).all() for v in observations.values()):
        raise ValueError('SAM 人物观测包含无效数据。')
    observations.update(boxes=boxes[first:stop], K=K[first:stop], source_frames=np.arange(first,stop))
    check()
    torch.save(tokens, token_path)
    save_npz(obs_path, **observations)
    np.save(hands_path, np.concatenate(hands))
    save_json(proof, dict(identity=identity, files={p.name:digest(p) for p in (token_path,obs_path,hands_path)}))
    return tokens, observations, str(hands_path)


def _join_motion(parts, valid, first):
    from scipy.spatial.transform import Rotation
    from .gauge import yaw, ry, wrap
    count = len(valid)
    base = parts[0][2]
    out = {key:np.array(value,copy=True) for key,value in base.items()}
    temporal = [key for key in ('poses','trans','Rh','foot_contacts') if key in base]
    for key in temporal:
        out[key] = np.empty((count,*base[key].shape[1:]),base[key].dtype)
    previous = None
    placements = []
    for a,b,motion,broken in parts:
        poses, trans = motion['poses'].copy(), motion['trans'].copy()
        if previous is not None:
            # Independent generation does not establish a new world origin.
            # Across absence OR a rejected cut, hold heading/position by convention.
            angle = wrap(yaw(Rotation.from_rotvec(previous[0]).as_matrix())-
                         yaw(Rotation.from_rotvec(poses[0,0]).as_matrix()))
            rotation = ry(angle)
            poses[:,0] = Rotation.from_matrix(rotation @ Rotation.from_rotvec(poses[:,0]).as_matrix()).as_rotvec()
            trans = trans @ rotation.T
            shift = previous[1]-trans[0]
            shift[1] = 0  # segment heights have already been placed on the common ground
            trans += shift
            placements.append(dict(source_start=first+a, yaw_deg=float(np.degrees(angle)), shift_m=shift.tolist(),
                                   basis='held heading and horizontal position across '+('pose discontinuity' if broken else 'absence')))
        for key in temporal:
            out[key][a:b] = poses if key=='poses' else trans if key=='trans' else poses[:,0] if key=='Rh' else motion[key]
        previous = (poses[-1,0], trans[-1])
    hold = held_indices(valid)
    for key in temporal:
        out[key] = out[key][hold]
    # Upstream auxiliary generated coordinates precede the final postprocessing
    # and do not describe the stitched world. The authoritative poses/trans do.
    for key in ('local_joints_positions','smooth_root_pos','pelvis_world'):
        out.pop(key,None)
    out.update(valid_frames=valid, num_frames=np.int64(count), source_frames=np.arange(first,first+count))
    return out, placements


def process_linked(engine, video, sub_dir, name, meta, shots, *, detection_range, person_selector,
                   progress, details, check, seed, cfg_scale, static_camera, vggt_interval,
                   sam_chunk_size, sam_hand_stride, smoothing_frames, hamer, hamer_stride, linked_base=None):
    import torch
    from flowhmr.pipeline.v2m_inference import StageName
    from flowhmr.utils.runtime_capabilities import detection_request_key
    from flowhmr.utils.runtime_video_processing import VideoProcessor
    from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
    from .gauge import linked_condition
    folder = Path(sub_dir)/'linked'
    folder.mkdir(exist_ok=True)
    result = _tracks(engine, video, folder, shots, check, progress)
    selection_file=folder/'selection.json'
    try:
        selection=json.loads(selection_file.read_text(encoding='utf8'))
        selected,multiple=selection['selected'],False
        if selected is not None and not 0<=selected<len(result.tracks):raise ValueError('Invalid saved selection')
    except (OSError,ValueError,KeyError):
        selected, multiple = _select(result, shots, person_selector)
    if multiple:
        VideoProcessor.save_detected_track(str(Path(sub_dir)/f'{name}_bbox.npz'), result, selected, detection_range)
        details(shot_linking_report=dict(mode='original', reason='multiple_people', shots=len(shots)),
                message='多人加切镜暂不连接，已按原流程分别处理人物。')
        return None
    save_json(selection_file,dict(selected=selected))
    boxes, valid, linked = link_tracks(result, shots, selected)
    boxes, valid, visibility = _recover_visibility(engine,video,folder,boxes,valid,shots,check,progress)
    first, stop = shots[0]['start'], shots[-1]['end']
    valid, discarded = filter_short_segments(valid, first=first)
    report = dict(mode='linked', shots=len(shots), tracks=linked, intervals=[], placements=[],
                  minimum_segment_frames=15, discarded_segments=discarded,
                  visibility=dict(missing_before=visibility['missing_before'],missing_after=visibility['missing_after'],
                                  evidence='input/linked/visibility.json'))
    details(shot_linking_report=json.loads(json.dumps(report,allow_nan=False)),
            track_range=[first,stop-1], message=f'已自动连接 {len(shots)} 个镜头')
    save_json(folder/'report.json',report)
    if not valid.any():
        raise ValueError('处理区间内没有至少连续 15 帧的人物观测，请调整区间；短暂碎段已记录并排除。')
    save_npz(Path(sub_dir)/f'{name}_bbox.npz', bbox=boxes, start_end=[first,stop-1],
             detection_request=detection_request_key(detection_range), valid_frames=valid)
    hw = (meta.height,meta.width)
    raw, K = _cameras(engine, video, folder, shots, hw, static_camera, vggt_interval, check, progress)
    # Source-indexed inputs match the upstream SAM API. Invalid frames are never inferred.
    full_boxes = np.zeros((meta.total_frames,4),np.float32)
    full_K = np.tile(K[0],(meta.total_frames,1,1))
    full_boxes[first:stop], full_K[first:stop] = boxes, K
    all_points = np.full((stop-first,70,3),np.nan,np.float32)
    all_orientation = np.full((stop-first,3,3),np.nan,np.float32)
    parts, prepared, all_breaks, corrections = [], [], [], []
    total_valid=int(valid.sum())
    observed=0
    for run_index,(a,b) in enumerate(intervals(valid)):
        check()
        run = folder/f'run_{first+a}_{first+b}'
        run.mkdir(exist_ok=True)
        tokens, obs, hands = _features(engine, video, run, full_boxes, full_K, first+a,first+b,
            chunk=sam_chunk_size, stride=sam_hand_stride, check=check,
            progress=lambda stage,pct,message: progress(stage,100*(observed+(b-a)*pct/100)/total_valid,message),
            details=details)
        observed+=b-a
        all_points[a:b], all_orientation[a:b] = obs['points'], obs['global_orient_camera']
        run_shots = clip_shots(shots, first+a,first+b)
        guard = continuity_breaks(obs['points'],run_shots, K=K[a:b], image_size_hw=hw)
        breaks = [r['frame'] for r in guard if r['broken']]
        all_breaks.extend(a+x for x in breaks)
        prepared.append((a,b,run,tokens,obs,hands,guard,breaks))
    valid, discarded = filter_short_segments(valid, all_breaks, first=first)
    report['discarded_segments'].extend(discarded)
    report['valid_frames'] = int(valid.sum())
    save_json(folder/'report.json',report)
    details(shot_linking_report=copy.deepcopy(report))
    if not valid.any():
        raise ValueError('切点分段后没有至少连续 15 帧的人物观测，请调整区间；短暂碎段已记录并排除。')
    total_valid = int(valid.sum())
    all_points[~valid], all_orientation[~valid] = np.nan, np.nan
    save_npz(Path(sub_dir)/f'{name}_bbox.npz', bbox=boxes, start_end=[first,stop-1],
             detection_request=detection_request_key(detection_range), valid_frames=valid)
    rotation_all,gauge_reports=linked_condition(raw[:,:3,:3],all_orientation,valid,
        clip_shots(shots,first,stop),all_breaks,source_start=first)
    report['gauge']=gauge_reports
    generated=0
    hamer_segments = {}
    replay = {}
    for a,b,run,tokens,obs,hands,guard,breaks in prepared:
        boundaries = [0,*breaks,b-a]
        for left,right in zip(boundaries,boundaries[1:]):
            if not valid[a+left:a+right].any():
                continue
            target = run/f'segment_{left}_{right}'
            target.mkdir(exist_ok=True)
            rotation=rotation_all[a+left:a+right]
            condition = np.tile(np.eye(4),(right-left,1,1))
            condition[:,:3,:3] = rotation
            save_npz(target/'condition.npz', camera_RT=condition)
            # Use the native public camera convention conversion, exactly once.
            camera = _vggt_opencv_to_yup_rt(torch.from_numpy(condition).float())
            segment_hands = target/'hands.npy'
            if linked_base is not None:
                replay[first+a+left, first+a+right] = (tokens[left:right], camera, segment_hands)
            np.save(segment_hands,np.load(hands)[left:right])
            def generation_progress(stage,pct,message):
                progress(stage,100*(generated+(right-left)*pct/100)/total_valid,message)
            def generation_details(**values):
                if 'hamer_report' in values:
                    key = (first+a+left, first+a+right)
                    hamer_segments[key] = copy.deepcopy(values['hamer_report'])
                    report['hamer_segments'] = [dict(source_frames=list(k), report=v)
                                                for k,v in hamer_segments.items()]
                    report['hamer_report'] = {field:sum(v[field] for v in hamer_segments.values())
                        for field in ('frames','left','right','sampled_left','sampled_right','seconds')}
                    report['hamer_report']['seconds'] = round(report['hamer_report']['seconds'], 2)
                    report['hamer_report']['stride'] = hamer_stride
                    values['hamer_report'] = copy.deepcopy(report['hamer_report'])
                details(**values)
            engine._generate_video_motion(tokens[left:right].float().contiguous(),camera,str(target),name,
                working_video=str(video), start_frame=first+a+left, seed=seed,cfg_scale=cfg_scale,
                smoothing_frames=smoothing_frames,hamer=hamer,hamer_stride=hamer_stride,
                hand_boxes_path=str(segment_hands),
                progress=generation_progress,
                details=generation_details,cancel_check=check,
                reference_shapes=parts[0][2]['betas'] if parts else None)
            suffix = '' if engine.postprocess is not None or hamer else '_origin'
            with np.load(target/f'{name}_seed{seed}{suffix}.npz',allow_pickle=False) as z:
                motion = dict(z)
            if engine.postprocess is not None:
                corrections.append(copy.deepcopy(engine.postprocess.last_report))
            parts.append((a+left,a+right,motion,left in breaks))
            report['intervals'].append(dict(source_frames=[first+a+left,first+a+right],
                continuity_guard=[{**g,'source_frame':first+a+g['frame']} for g in guard],arm_fit=dict(enabled=False,removed=True)))
            generated+=right-left
            save_json(folder/'report.json',report)
    combined_correction=None
    if corrections:
        from .ground import align_segment_ground,aggregate_correction
        options=getattr(engine.postprocess,'options',{})
        parts,report['ground']=align_segment_ground(parts,corrections,first,
            fixed_floor=None if options.get('auto_floor',True) else options['floor'])
        combined_correction=aggregate_correction(report['ground'])
    motion, report['placements'] = _join_motion(parts,valid,first)
    if corrections and len(parts)>1:motion['ground_height_m']=np.float64(report['ground']['height_m'])
    if linked_base is not None:
        from .torso import compose_linked
        save_npz(folder/'standard-before-torso.npz', **motion)
        spans = [tuple(item['source_frames']) for item in report['intervals']]
        with TemporaryDirectory(dir=folder, prefix='base-pass-') as staging, linked_base():
            def generate_base(start, stop):
                check()
                segment_tokens, camera, hands_path = replay[start, stop]
                target = Path(staging)/f'{start}_{stop}'
                target.mkdir()
                engine._generate_video_motion(segment_tokens.float().contiguous(), camera, str(target), name,
                    working_video=str(video), start_frame=start, seed=seed, cfg_scale=cfg_scale,
                    smoothing_frames=smoothing_frames, hamer=False, hamer_stride=1,
                    hand_boxes_path=str(hands_path), progress=lambda *args: None,
                    details=lambda values: None, cancel_check=check, reference_shapes=motion['betas'])
                suffix = '' if engine.postprocess is not None else '_origin'
                with np.load(target/f'{name}_seed{seed}{suffix}.npz', allow_pickle=False) as data:
                    return dict(poses=data['poses'], trans=data['trans'],
                                source_frames=np.arange(start, start+len(segment_tokens)))
            motion, torso_report = compose_linked(motion, spans, generate_base, engine.smpl_model_path)
            check()
        report['torso_opt'] = torso_report
        details(torso_opt=torso_report)
    check()
    suffix = '' if engine.postprocess is not None or hamer else '_origin'
    output = Path(sub_dir)/f'{name}_seed{seed}{suffix}'
    save_npz(output.with_suffix('.npz'),**motion)
    if combined_correction is not None:
        from .ground import write_correction
        write_correction(output.with_suffix('.npz'),combined_correction)
    try:
        from flowhmr.core.export.smplh2fbx import SMPLH2FBX
    except ImportError:
        pass
    else:
        with SMPLH2FBX(smplh_model_path=engine.smpl_model_path) as exporter:
            exporter.convert_npz_to_fbx(motion,str(output.with_suffix('.fbx')),export_mesh=True)
    save_npz(folder/'camera_inputs.npz',RT=raw,K=K,points=all_points,global_orient_camera=all_orientation,
             boxes=boxes,valid=valid,image_size_hw=hw,track_start=first)
    save_json(folder/'camera_inputs.json',dict(shots=clip_shots(shots,first,stop),
        sha256=digest(folder/'camera_inputs.npz'),motion_sha256=digest(output.with_suffix('.npz'))))
    save_json(folder/'report.json',report)
    details(shot_linking_report=json.loads(json.dumps(report,allow_nan=False)))
    progress(StageName.GENERATION,100,f'已自动连接 {len(shots)} 个镜头')
    return sub_dir
