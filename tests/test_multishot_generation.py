import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from flowhmr.multishot.linking import link_tracks, shot_candidates, continuity_breaks
from flowhmr.multishot.gauge import build_condition, linked_condition, ry, yaw
from flowhmr.multishot.generation import _join_motion
from flowhmr.multishot.shots import clip_shots, detect_shots


def track(first, stop, box, ident=0):
    frames = np.arange(first,stop)
    return SimpleNamespace(frames=frames, bboxes=np.tile(box,(len(frames),1)),
                           track_id=ident,length=len(frames),avg_area=float(np.prod(np.array(box[2:])-box[:2])))


class LinkedMotionTests(unittest.TestCase):
    def test_camera_speed_uses_elapsed_source_time_across_absence(self):
        raw=ry(np.array([0.,1.]))
        _,report=build_condition(raw,raw,[dict(start=0,end=2)],source_frames=[0,30],jitter_rms_deg_limit=100.)
        self.assertAlmostEqual(report['shots'][0]['raw_camera_peak_deg_s'],180/np.pi)
        self.assertFalse(report['shots'][0]['static_fallback'])

    def test_incomplete_decode_is_not_cached_as_a_complete_uncut_shot(self):
        stream=Mock(frame_number=61)
        manager=Mock()
        manager.get_scene_list.return_value=[]
        with TemporaryDirectory() as directory:
            video=Path(directory)/'source.mp4';video.write_bytes(b'fixture')
            cache=Path(directory)/'shots.json'
            with patch('scenedetect.open_video',return_value=stream), patch('scenedetect.SceneManager',return_value=manager):
                with self.assertRaisesRegex(ValueError,'完整读取'):
                    detect_shots(video,40,90,cache)
            self.assertFalse(cache.exists())
            stream._cap.release.assert_called_once()

    def test_only_short_detection_holes_are_interpolated(self):
        t=track(1,30,[0,0,10,20])
        keep=~np.isin(t.frames,[8,9,17,18,19,20])
        t.frames,t.bboxes=t.frames[keep],t.bboxes[keep]
        _,valid,_=link_tracks(SimpleNamespace(tracks=[t]),[dict(id=0,start=0,end=31)])
        self.assertTrue(valid[0] and valid[8] and valid[9] and valid[30])
        self.assertFalse(valid[17:21].any())

    def test_gauge_continues_through_same_shot_gap_and_holds_at_missing_cut(self):
        n=40
        body=ry(np.linspace(0,.5,n));body[20:]=ry(1.)@body[20:]
        valid=np.ones(n,bool);valid[8:11]=False;valid[18:22]=False
        camera=np.tile(np.eye(3),(n,1,1))
        result,reports=linked_condition(camera,body,valid,
            [dict(id=0,start=0,end=20),dict(id=1,start=20,end=40)])
        np.testing.assert_allclose(result[7],result[11],atol=1e-12)
        headings=yaw(result.transpose(0,2,1)@body)
        self.assertAlmostEqual(headings[17],headings[22],places=10)
        self.assertEqual(reports[0]['cuts'][0]['continuity_basis'],'held_heading')

    def test_shot_selection_is_local_and_missing_frames_are_not_invented(self):
        shots = [dict(id=0,start=40,end=70),dict(id=1,start=70,end=100)]
        result = SimpleNamespace(tracks=[track(43,70,[0,0,10,20]), track(70,95,[0,0,30,40]),
                                        track(70,100,[0,0,10,10])])
        boxes,valid,report = link_tracks(result,shots,selected=0)
        self.assertEqual(report[1]['track_index'],1)
        self.assertFalse(valid[:3].any())
        self.assertFalse(valid[-5:].any())
        self.assertTrue(valid[3:-5].all())
        self.assertEqual(clip_shots(shots,65,80),[dict(id=0,start=0,end=5,source_start=65,source_end=70),
                                                dict(id=1,start=5,end=15,source_start=70,source_end=80)])

    def test_a_camera_cut_does_not_look_like_a_pose_jump(self):
        p = np.zeros((60,70,3))
        p[:,5]=[-.2,1,3]; p[:,6]=[.2,1,3]
        p[:,9]=[-.1,0,3]; p[:,10]=[.1,0,3]
        p[:,7]=[-.3,.5,3];p[:,8]=[.3,.5,3]
        p[:,62]=[-.4,.2,3];p[:,41]=[.4,.2,3]
        p[30:] = p[30:] @ ry(1.2).T + [2,1,4]
        shots=[dict(start=0,end=30),dict(start=30,end=60)]
        K=np.tile([[100,0,5000],[0,100,5000],[0,0,1]],(60,1,1))
        self.assertFalse(continuity_breaks(p,shots,K=K,image_size_hw=(10000,10000))[0]['broken'])
        p[30:,62] += [0,1,0]
        self.assertTrue(continuity_breaks(p,shots,K=K,image_size_hw=(10000,10000))[0]['broken'])

    def test_gauge_heading_is_continuous_without_flattening_body_lean(self):
        n=60
        body=Rotation.from_euler('yx',np.c_[np.linspace(0,.6,n),np.sin(np.arange(n)/10)*.15]).as_matrix()
        body[30:] = ry(1.3) @ body[30:]
        result,report=build_condition(np.tile(np.eye(3),(n,1,1)),body,
            [dict(start=0,end=30),dict(start=30,end=60)])
        self.assertLess(abs(report['cuts'][0]['extrapolation_residual_deg']),1e-8)
        self.assertTrue(np.allclose(np.linalg.det(result),1))
        self.assertGreater(np.std((result.transpose(0,2,1)@body)[:,1,1]),.001)

    def test_gap_holds_pose_and_broken_cut_preserves_world_placement(self):
        def motion(n,angle,x):
            poses=np.zeros((n,52,3),np.float32);poses[:,0,1]=angle
            return dict(poses=poses,Rh=poses[:,0].copy(),trans=np.tile([x,1.,0.],(n,1)).astype(np.float32),
                        betas=np.zeros((1,16),np.float32),num_frames=n)
        valid=np.array([False,True,True,False,True,True])
        out,report=_join_motion([(1,3,motion(2,0,2),False),(4,6,motion(2,1,9),False)],valid,100)
        self.assertTrue(np.array_equal(out['poses'][0],out['poses'][1]))
        self.assertTrue(np.array_equal(out['trans'][3],out['trans'][2]))
        self.assertAlmostEqual(out['poses'][4,0,1],0,places=6)
        self.assertEqual(report[0]['source_start'],104)
        json.dumps(report,allow_nan=False)  # This exact report is persisted in job.json.
        broken,_=_join_motion([(1,3,motion(2,0,2),False),(4,6,motion(2,1,9),True)],valid,100)
        self.assertAlmostEqual(broken['poses'][4,0,1],0,places=6)
        np.testing.assert_allclose(broken['trans'][4,[0,2]],broken['trans'][2,[0,2]])


if __name__=='__main__':
    unittest.main()
