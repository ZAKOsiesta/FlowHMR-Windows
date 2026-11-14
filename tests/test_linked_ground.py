"""Exercise the actual linked generation call site and its published report."""
import json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import numpy as np
import torch
from flowhmr.multishot.generation import process_linked
from flowhmr.pipeline.v2m_inference import StageName
from flowhmr.utils.runtime_video_processing import VideoProcessor
from flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt


class LinkedGroundTests(unittest.TestCase):
    def run_case(self,root,multiple):
        frames=np.r_[0:3,17:20] if multiple else np.arange(20)
        result=SimpleNamespace(tracks=[SimpleNamespace(frames=frames,bboxes=np.tile([0,0,10,20],(len(frames),1)),track_id=1,length=len(frames),avg_area=200)])
        shots=[dict(id=0,start=0,end=10),dict(id=1,start=10,end=20)]
        engine=SimpleNamespace(postprocess=SimpleNamespace(last_report=None),smpl_model_path='unused')
        calls=[]
        def generate(tokens,camera,target,name,**kw):
            i=len(calls);floor=[-.08,.02][i];n=len(tokens)
            poses=np.zeros((n,52,3),np.float32);trans=np.zeros((n,3),np.float32)
            trans[:,1]=1+floor;trans[n//2,1]+=.2  # a legal raised frame must survive
            np.savez(Path(target)/f'{name}_seed0.npz',poses=poses,trans=trans,Rh=poses[:,0],betas=np.zeros((1,16),np.float32),foot_contacts=np.ones((n,4),np.float32))
            engine.postprocess.last_report=dict(floor=floor,preset='normal',contacts=n*2,elapsed_seconds=1,warning='')
            calls.append(dict(target=target,floor=floor))
        engine._generate_video_motion=generate
        def features(engine,video,folder,boxes,K,first,stop,**kw):
            n=stop-first;hands=folder/'hands.npy';np.save(hands,np.zeros((n,2,4)))
            return torch.zeros((n,3072)),dict(points=np.zeros((n,70,3)),global_orient_camera=np.tile(np.eye(3),(n,1,1))),str(hands)
        details=[]
        exporter=MagicMock();exporter.return_value.__enter__.return_value=MagicMock()
        with patch('flowhmr.multishot.generation._tracks',return_value=result),patch('flowhmr.multishot.generation._features',side_effect=features),\
             patch('flowhmr.multishot.generation._cameras',return_value=(np.tile(np.eye(4),(20,1,1)),np.tile(np.eye(3),(20,1,1)))),\
             patch('flowhmr.multishot.generation._recover_visibility',side_effect=lambda e,v,f,b,m,s,c,p:(b,m,dict(missing_before=int((~m).sum()),missing_after=int((~m).sum())))),\
             patch.dict(sys.modules,{'flowhmr.core.export.smplh2fbx':SimpleNamespace(SMPLH2FBX=exporter)}):
            process_linked(engine,'fixture',str(root),'input',SimpleNamespace(height=20,width=10,total_frames=20),shots,
                detection_range=(0,19),person_selector=None,progress=lambda *a:None,details=lambda **kw:details.append(kw),check=lambda:None,
                seed=0,cfg_scale=1,static_camera=True,vggt_interval=0,sam_chunk_size=24,sam_hand_stride=1,smoothing_frames=1,hamer=False,hamer_stride=1)
        return dict(np.load(root/'input_seed0.npz')),json.loads((root/'linked/report.json').read_text()),engine

    def test_multiple_estimated_floors_are_unified_and_report_is_aggregate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);motion,report,engine=self.run_case(root,True)
            np.testing.assert_allclose(motion['trans'][[0,2,17,19],1],1,atol=1e-6)
            np.testing.assert_allclose(motion['trans'][[1,18],1],1.2,atol=1e-6)
            self.assertEqual(float(motion['ground_height_m']),0.)
            self.assertEqual(report['ground']['height_m'],0.)
            from flowhmr.multishot.ground import correction_summary
            summary=correction_summary(root/'input_seed0.npz',engine.postprocess,dict(correction_floor_m=None))
            self.assertEqual(summary['floor_m'],0.);self.assertEqual(summary['contacts'],12)
            self.assertEqual(len(summary['ground']['segments']),2)

    def test_single_generated_interval_keeps_native_floor_and_trajectory(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);motion,report,engine=self.run_case(root,False)
            self.assertAlmostEqual(float(motion['trans'][0,1]),.92,places=6)
            self.assertAlmostEqual(float(motion['trans'][10,1]),1.12,places=6)
            from flowhmr.multishot.ground import correction_summary
            self.assertAlmostEqual(correction_summary(root/'input_seed0.npz',engine.postprocess,dict(correction_floor_m=None))['floor_m'],-.08)


if __name__=='__main__':unittest.main()
