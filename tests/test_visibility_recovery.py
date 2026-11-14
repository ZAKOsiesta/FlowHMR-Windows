import unittest
import cv2
import numpy as np
from unittest.mock import Mock,patch
from flowhmr.multishot.visibility import recover_shot_gaps,redetect_gaps,accepted_candidate,recover_video_gaps


class VisibilityRecoveryTests(unittest.TestCase):
    def clip(self,missing=False):
        rng=np.random.default_rng(18)
        person=rng.integers(30,240,(90,50),dtype=np.uint8)
        images=[]
        for t in range(8):
            im=np.zeros((180,240),np.uint8)+15
            if not missing or t in (0,7):im[40:130,50+t:100+t]=person
            images.append(im)
        boxes=np.zeros((8,4),np.float32);boxes[0]=[50,40,100,130];boxes[-1]=[57,40,107,130]
        valid=np.zeros(8,bool);valid[[0,7]]=True
        return images,boxes,valid

    def test_visible_person_recovers_from_bidirectional_image_evidence(self):
        images,boxes,valid=self.clip()
        actual,mask,report=recover_shot_gaps(images,boxes,valid)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(actual[3],[53,40,103,130],atol=1)

    def test_real_absence_is_not_filled_by_neighbor_boxes(self):
        images,boxes,valid=self.clip(missing=True)
        _,mask,_=recover_shot_gaps(images,boxes,valid)
        np.testing.assert_array_equal(mask,valid)

    def test_no_anchor_or_long_absence_stays_missing(self):
        images,boxes,valid=self.clip()
        valid[:]=False
        self.assertFalse(recover_shot_gaps(images,boxes,valid)[1].any())
        images,boxes,valid=self.clip()
        self.assertEqual(recover_shot_gaps(images,boxes,valid,max_gap=3)[1].sum(),2)

    def test_edge_disappearance_is_not_held(self):
        images,boxes,valid=self.clip(missing=True)
        images,boxes,valid=images[:5],boxes[:5],valid[:5]
        self.assertEqual(recover_shot_gaps(images,boxes,valid)[1].sum(),1)

    def test_direct_person_evidence_recovers_nine_frame_shot_entry(self):
        boxes=np.tile([50.,40.,100.,130.],(12,1)).astype(np.float32)
        valid=np.ones(12,bool);valid[:9]=False
        detector=Mock();detector.run_human_detection.side_effect=lambda *a,**kw:np.array([[50.,40.,100.,130.,.35]],np.float32)
        cap=Mock();cap.read.return_value=(True,np.zeros((180,240,3),np.uint8))
        with patch('flowhmr.utils.runtime_video_processing.open_video_at_frame',return_value=cap):
            _,mask,evidence=redetect_gaps(detector,'fixture',boxes,valid,[dict(id=0,start=100,end=112)])
            self.assertTrue(mask.all());self.assertEqual(len(evidence),9)
            detector.run_human_detection.side_effect=None;detector.run_human_detection.return_value=None
            self.assertEqual(redetect_gaps(detector,'fixture',boxes,valid,[dict(id=0,start=100,end=112)])[1].sum(),3)

    def test_other_person_does_not_establish_presence(self):
        self.assertIsNone(accepted_candidate([[150,10,200,100,.9]],np.array([50,40,100,130])))

    def test_person_in_previous_shot_cannot_seed_empty_next_shot(self):
        boxes=np.tile([50.,40.,100.,130.],(8,1)).astype(np.float32)
        valid=np.array([True]*4+[False]*4)
        shots=[dict(id=0,start=30,end=34),dict(id=1,start=34,end=38)]
        with patch('flowhmr.utils.runtime_video_processing.open_video_at_frame') as open_video:
            self.assertTrue(np.array_equal(recover_video_gaps('fixture',boxes,valid,shots)[1],valid))
            open_video.assert_not_called()


if __name__=='__main__':unittest.main()
