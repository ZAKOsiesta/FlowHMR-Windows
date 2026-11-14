"""Final coordinates and stale preview evidence are checked at their consumer."""
import json
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from flowhmr.utils.runtime_preview_inspection import PreviewInspection
from flowhmr.multishot.ground import align_segment_ground,correction_summary,write_correction,published_floor


class PreviewGroundTests(unittest.TestCase):
    def test_explicit_ground_and_same_length_replacement_invalidate_cache(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);mesh=root/'mesh';mesh.mkdir()
            (mesh/'meta.json').write_text(json.dumps(dict(num_frames=3,num_vertices=2,fps=30)))
            vertices=np.zeros((3,2,3),dtype='<f2');vertices.tofile(mesh/'vertices.bin')
            poses=np.zeros((3,52,3),np.float32);trans=np.zeros((3,3),np.float32)
            def motion():np.savez(root/'motion.npz',poses=poses,trans=trans,betas=np.zeros(16,np.float32),ground_height_m=.25)
            motion();inspection=PreviewInspection();calls=[]
            def skeleton(inputs):
                calls.append(1)
                return dict(keypoints3d=inputs['trans'][:,None,:].expand(-1,52,-1))
            geometry=(skeleton,[-1]+list(range(51)),[np.array([0]),np.array([1])])
            with patch.object(inspection,'_body_geometry',return_value=geometry):
                path,info=inspection.load(root)
                self.assertEqual(info['ground'],dict(height_m=.25,source='correction'))
                inspection.load(root);self.assertEqual(len(calls),1)
                trans[:,1]=2;motion()
                path,new=inspection.load(root)
                self.assertEqual(len(calls),2)
                self.assertEqual(np.fromfile(path,dtype='<f4').reshape(3,52,3)[0,0,1],2.)
                vertices[:,0,1]=-.5;vertices.tofile(mesh/'vertices.bin')
                inspection.load(root);self.assertEqual(len(calls),3)
                self.assertEqual(published_floor(root,dict(floor_m=-10),0),.25)

    def test_manual_floor_changes_only_constant_segment_height(self):
        trans=np.array([[1,.8,2],[2,1.3,3]],np.float32)
        parts=[(0,2,dict(trans=trans),False),(2,4,dict(trans=trans),False)]
        out,ground=align_segment_ground(parts,[dict(floor=-.2),dict(floor=.1)],fixed_floor=.4)
        self.assertEqual(ground['height_m'],.4)
        for (_,_,motion,_),shift in zip(out,[.6,.3]):
            np.testing.assert_allclose(motion['trans']-trans,np.tile([0,shift,0],(2,1)),atol=1e-7)
            np.testing.assert_allclose(np.diff(motion['trans'],axis=0),np.diff(trans,axis=0),atol=1e-7)

    def test_changed_motion_cannot_reuse_old_correction_report(self):
        with TemporaryDirectory() as tmp:
            motion=Path(tmp)/'motion.npz';np.savez(motion,trans=np.zeros((1,3)))
            write_correction(motion,dict(floor=0,preset='normal',contacts=1,elapsed_seconds=1))
            np.savez(motion,trans=np.ones((1,3)))
            with self.assertRaisesRegex(ValueError,'does not match'):
                correction_summary(motion,SimpleNamespace(last_report={}),dict(correction_floor_m=None))

    def test_camera_bundle_declares_the_same_ground_after_character_transform(self):
        from flowhmr.multishot.camera import Camera
        from flowhmr.multishot.exports import camera_bundle
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'camera').mkdir()
            camera=Camera(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(3),(2,1,1)),
                          np.ones(2,bool),np.zeros(2,np.int32),np.array([100,100]))
            camera.save(root/'camera/camera.npz')
            np.savez(root/'motion.npz',trans=np.zeros((2,3)),ground_height_m=.3)
            with ZipFile(camera_bundle(root,'npz',None,dict(scale=2,translation=[0,.1,0]))) as bundle:
                self.assertAlmostEqual(json.loads(bundle.read('camera.json'))['ground_y_m'],.7)


if __name__=='__main__':unittest.main()
