import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask

from flowhmr.multishot.camera import Camera, project
from flowhmr.multishot.recovery import recover_camera, CameraCancelled, SM, MH
from flowhmr.multishot.tasks import CameraTasks, restore_camera_state
from flowhmr.multishot.web import register_camera_routes
from flowhmr.utils.runtime_group_scene import align_tracks
from flowhmr.multishot.similarity import transform_camera
from flowhmr.multishot.gate import gate_camera


def fixture(n=30):
    joints = np.zeros((n, 52, 3), np.float32)
    points = np.zeros((n, 70, 3), np.float64)
    body = np.array([[-.3,1.5,0],[.3,1.5,0],[-.45,1.2,0],[.45,1.2,0],[-.5,1.,0],[.5,1.,0],
                     [-.15,.9,0],[.15,.9,0],[-.15,.5,0],[.15,.5,0],[-.15,.05,0],[.15,.05,0]])
    joints[:, SM] = body
    R = np.diag([1.,-1.,-1.])
    C = np.array([0.,1.,4.])
    points[:] = [0,0,4]
    points[:, MH] = (body-C) @ R.T
    rt = np.tile(np.eye(4), (n,1,1))
    rt[:,:3,:3] = R
    K = np.tile([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]], (n,1,1))
    return rt, K, joints, points, np.ones(n,bool)


class RecoveryTest(unittest.TestCase):
    def test_static_no_scale_population_and_motion_immutable(self):
        args = fixture()
        original = [a.copy() for a in args]
        camera, report = recover_camera(*args, [dict(id=0,start=0,end=30)], (480,640))
        self.assertTrue(camera.camera_valid.all())
        self.assertIn('fixed_center', report['shots'][0]['labels'])
        self.assertIn('gravity_fallback', report['shots'][0]['labels'])
        self.assertLess(np.ptp(camera.centers, axis=0).max(), 1e-10)
        for before, after in zip(original, args):
            np.testing.assert_array_equal(before, after)

    def test_short_and_failed_shots_do_not_poison_valid_result(self):
        args = list(fixture(24))
        args[3][6:12,:,2] = -4
        args[4][12:15] = False
        camera, report = recover_camera(*args, [dict(id=i,start=a,end=b)
            for i,(a,b) in enumerate([(0,6),(6,12),(12,24)])], (480,640))
        self.assertTrue(camera.camera_valid[:6].all())
        self.assertFalse(camera.camera_valid[6:15].any())
        self.assertTrue(camera.camera_valid[15:].all())
        self.assertIn('short_shot', report['shots'][0]['labels'])
        self.assertIn('failed', report['shots'][1]['labels'])

    def test_cancellation_exits_fitting(self):
        def stop():
            raise CameraCancelled()
        with self.assertRaises(CameraCancelled):
            recover_camera(*fixture(), [dict(id=0,start=0,end=30)], (480,640), check=stop)

    def test_high_dispersion_gravity_falls_back(self):
        _, report = recover_camera(*fixture(), [dict(id=0,start=0,end=30)], (480,640),
            gravity={0:dict(mean_world_up=[0,-1,0], dispersion_max_deg=30)})
        self.assertIn('gravity_fallback', report['shots'][0]['labels'])

    def test_two_people_share_the_same_recovered_coordinate_system(self):
        n=30
        rt,K,joints,_,_=fixture(n)
        center=np.zeros((n,3));center[:,0]=np.linspace(-.5,.5,n)
        rt[:,:3,3]=-np.einsum('tij,tj->ti',rt[:,:3,:3],center)
        true_center=2.5*center+[0.,1.,5.]
        body=joints[:,[16,17,1,2,4,5,7,8]]
        tracks=[]
        for offset in ([0.,0.,0.],[1.5,0.,.3]):
            seen=np.einsum('tij,tkj->tki',rt[:,:3,:3],body+offset-true_center[:,None])
            tracks.append(dict(rt=rt,observed=seen,motion=body,floor=0.))
        scale=align_tracks(tracks)
        frame=tracks[0]['shared_frame']
        camera=transform_camera(rt,frame['rotation'],frame['scale'],frame['translation'])
        centers=-np.einsum('tji,tj->ti',camera[:,:3,:3],camera[:,:3,3])
        self.assertAlmostEqual(scale,2.5,places=7)
        np.testing.assert_allclose(centers,true_center,atol=1e-7)
        np.testing.assert_allclose(tracks[1]['offset'],[1.5,0.,.3],atol=1e-7)
        for track in tracks:
            uv,_=project(camera,K,body+track['offset'])
            expected=track['observed'] @ K[0].T
            np.testing.assert_allclose(uv,expected[:,:,:2]/expected[:,:,2:],atol=1e-6)

    def test_low_observation_is_reported_without_residual_veto(self):
        rt,K,joints,points,valid=fixture()
        rt[:,2,3]=4.
        points[:,MH[4:],0]=100.
        _,_,report=gate_camera(rt,K,joints,points,[dict(shot=0,source_frames=[0,30],scale=1.)],valid,(480,640))
        self.assertIn('low_observation',report['shots'][0]['labels'])
        self.assertFalse(report['shots'][0]['kept_baseline'])


class CameraLifecycleTest(unittest.TestCase):
    def test_archived_raw_inputs_use_body_reference_without_loading_models(self):
        from flowhmr.multishot.runtime import build_camera, digest, save_json
        from types import SimpleNamespace
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'camera').mkdir()
            rt,K,joints,points,valid=fixture(6)
            motion=root/'motion.npz'
            np.savez(motion,trans=np.zeros((6,3)),valid_frames=valid)
            joints.tofile(root/'joints.bin')
            np.savez(root/'camera/inputs.npz',RT=rt,K=K,joints=joints,points=points,
                     valid=valid,image_size_hw=np.array([480,640],np.int32))
            save_json(root/'camera/inputs.json',dict(motion_sha256=digest(motion),
                sha256=digest(root/'camera/inputs.npz'),shots=[dict(id=0,start=0,end=6)],gravity={}))
            references=[]
            def inspect(directory,body_reference):
                references.append(body_reference)
                return root/'joints.bin',dict(frames=6)
            def forbidden():
                raise AssertionError('Archived input must not load a model')
            before=digest(motion)
            report=build_camera(root,dict(id='old',body_reference='female',params={}),
                                SimpleNamespace(load=inspect),forbidden,'cpu')
            self.assertEqual(references,['female'])
            self.assertEqual(digest(motion),before)
            self.assertTrue(Camera.load(root/'camera/camera.npz').camera_valid.all())
            self.assertEqual(report['generation_calls'],0)
            self.assertEqual(json.loads((root/'camera/report.json').read_text())['body_reference'],'female')

    def test_http_queue_dedup_fail_retry_cancel_and_restart_preserve_motion(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'camera').mkdir()
            np.savez(root/'motion.npz',trans=np.zeros((6,3)),ground_height_m=.25)
            frozen=(root/'motion.npz').read_bytes()
            job = dict(id='sample', status='done', params={}, stages=[dict(name='generation',status='completed')])
            initial = copy.deepcopy(job)
            queue, calls = [], []
            def build(job_id, check, notify):
                check()
                calls.append(job_id)
                if len(calls) == 1:
                    raise RuntimeError('test model failure')
                a = fixture(6)
                Camera(a[0], a[1], a[4], np.zeros(6,np.int32), np.array([480,640])).save(root/'camera/camera.npz')
            service = CameraTasks(lambda _: copy.deepcopy(job), lambda _: root,
                lambda _, **kw: job.update(kw), queue.append, build)
            app = Flask(__name__)
            register_camera_routes(app, lambda _:copy.deepcopy(job), lambda _:root, lambda *_:None, service)
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(lambda _: service.prepare('sample'), range(8)))
            self.assertEqual(queue, [('camera','sample')])
            service.run('sample')
            self.assertEqual(job['camera_state'],'failed')
            self.assertEqual(job['status'],'done')
            client = app.test_client()
            self.assertEqual(client.post('/api/jobs/sample/camera/prepare').status_code,202)
            client.post('/api/jobs/sample/camera/cancel')
            service.run('sample')
            self.assertEqual(job['camera_state'],'cancelled')
            self.assertEqual(len(calls),1)
            client.post('/api/jobs/sample/camera/prepare')
            service.run('sample')
            self.assertEqual(client.get('/api/jobs/sample/camera/status').json['camera_state'],'ready')
            self.assertEqual(client.get('/api/jobs/sample/camera').status_code,200)
            self.assertEqual(client.get('/api/jobs/sample/camera').json['ground_y_m'],.25)
            self.assertEqual((root/'motion.npz').read_bytes(),frozen)
            for key,value in initial.items():
                self.assertEqual(job[key],value)
            interrupted = {**initial,'camera_state':'running'}
            restore_camera_state(interrupted)
            self.assertEqual(interrupted['camera_state'],'failed')
            self.assertEqual(interrupted['status'],'done')
            (root/'camera/camera.npz').unlink()
            self.assertEqual(service.status('sample')['camera_state'],'none')
            self.assertEqual(service.prepare('sample')['camera_state'],'queued')


if __name__ == '__main__':
    unittest.main()
