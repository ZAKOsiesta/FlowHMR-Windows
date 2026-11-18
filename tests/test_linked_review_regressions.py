"""Review regressions at the real orchestration boundary; model calls are fixtures."""
from contextlib import ExitStack, contextmanager
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from flowhmr.multishot import generation as linked
from flowhmr.multishot.linking import continuity_breaks
from flowhmr.pipeline.v2m_inference import V2MInferenceEngine
from flowhmr.utils.runtime_video_processing import TrackInfo, TrackingResult, VideoProcessor


def observed_body(n):
    points = np.zeros((n, 70, 3), np.float32)
    points[..., 2] = 3
    for joint, xy in {5: [-.2, .6], 6: [.2, .6], 9: [-.1, -.2],
                      10: [.1, -.2], 7: [-.3, .3], 8: [.3, .3],
                      62: [-.4, .1], 41: [.4, .1]}.items():
        points[:, joint, :2] = xy
    return points


def motion(n, angle=0., x=0.):
    poses = np.zeros((n, 52, 3), np.float32)
    poses[:, 0, 1] = angle
    return dict(poses=poses, Rh=poses[:, 0].copy(),
                trans=np.tile([x, 1., 0.], (n, 1)).astype(np.float32),
                betas=np.zeros((1, 16), np.float32), num_frames=n)


class LinkedReviewTests(unittest.TestCase):
    def run_linked(self, valid, points=None, *, cut=30, hamer=False, cancel_base=False):
        n = len(valid)
        points = observed_body(n) if points is None else points
        boxes = np.tile([0., 0., 100., 100.], (n, 1)).astype(np.float32)
        K = np.tile([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]], (n, 1, 1))
        shots = [dict(id=0, start=0, end=cut), dict(id=1, start=cut, end=n)]
        state, generated = {}, []
        def details(**values):
            state.update(copy.deepcopy(values))
        def features(engine, video, folder, boxes, intrinsics, first, stop, **kwargs):
            np.save(folder/'hands.npy', np.zeros((stop-first, 2, 4), np.float32))
            return torch.zeros(stop-first, 3072), dict(points=points[first:stop],
                global_orient_camera=np.tile(np.eye(3), (stop-first, 1, 1))), str(folder/'hands.npy')
        def generate(tokens, camera, target, name, **kwargs):
            size = len(tokens)
            generated.append([kwargs['start_frame'], kwargs['start_frame']+size])
            if kwargs['hamer']:
                report = dict(frames=size, left=size, right=size-1, sampled_left=size,
                              sampled_right=size-1, stride=1, seconds=1.25)
                kwargs['details'](hamer_report=report)
                # Repeated status publication must replace this segment, not double count it.
                kwargs['details'](hamer_report=report)
            suffix = '' if hamer else '_origin'
            np.savez(Path(target)/f'{name}_seed0{suffix}.npz', **motion(size, x=len(generated)*2.))
        engine = SimpleNamespace(postprocess=None, _generate_video_motion=generate)
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            stack.enter_context(patch.object(linked, '_tracks', return_value=SimpleNamespace(tracks=[])))
            stack.enter_context(patch.object(linked, '_select', return_value=(None, False)))
            stack.enter_context(patch.object(linked, 'link_tracks', return_value=(boxes, valid.copy(), [])))
            stack.enter_context(patch.object(linked, '_recover_visibility', return_value=(boxes, valid.copy(),
                dict(missing_before=int((~valid).sum()), missing_after=int((~valid).sum())))))
            stack.enter_context(patch.object(linked, '_cameras', return_value=(np.tile(np.eye(4), (n, 1, 1)), K)))
            stack.enter_context(patch.object(linked, '_features', side_effect=features))
            stack.enter_context(patch.dict(sys.modules, {'flowhmr.core.export.smplh2fbx': None}))
            @contextmanager
            def cancelled_pass():
                raise InterruptedError('Cancelled during torso optimization')
                yield

            def run():
                return linked.process_linked(engine, 'source.mp4', root, 'fixture',
                SimpleNamespace(height=100, width=100, total_frames=n), shots,
                detection_range=(0,n-1), person_selector=None, progress=lambda *x:None,
                details=details, check=lambda:None, seed=0, cfg_scale=1., static_camera=True,
                vggt_interval=0, sam_chunk_size=24, sam_hand_stride=1, smoothing_frames=0.,
                hamer=hamer, hamer_stride=1, linked_base=cancelled_pass if cancel_base else None)
            if cancel_base:
                with self.assertRaises(InterruptedError):
                    run()
                with np.load(root/'linked/standard-before-torso.npz') as standard:
                    np.testing.assert_array_equal(standard['valid_frames'], valid)
                self.assertFalse((root/'fixture_seed0_origin.npz').exists())
                self.assertFalse((root/'linked/camera_inputs.json').exists())
                self.assertFalse(list((root/'linked').glob('base-pass-*')))
                return
            run()
            suffix = '' if hamer else '_origin'
            with np.load(root/f'fixture_seed0{suffix}.npz') as z:
                output = dict(z)
            saved = json.loads((root/'linked/report.json').read_text('utf8'))
        return generated, state, output, saved

    def test_cancelled_torso_pass_keeps_standard_and_publishes_no_result(self):
        self.run_linked(np.ones(60, bool), cancel_base=True)

    def test_offscreen_upper_body_cannot_split_a_continuous_run(self):
        p = observed_body(60)
        p[30:, :, 1] -= 5  # Upper body outside the image after a close-up cut.
        p[30:, 62, 1] -= 2  # Arbitrary prediction for the unseen wrist.
        generated, state, _, _ = self.run_linked(np.ones(60, bool), p)
        self.assertEqual(generated, [[0, 60]])
        guard = state['shot_linking_report']['intervals'][0]['continuity_guard'][0]
        self.assertFalse(guard['broken'])

    def test_visible_true_pose_jump_still_splits_but_keeps_world_placement(self):
        p = observed_body(60)
        p[30:, 62, 1] += .7  # Visible genuine articulation discontinuity.
        generated, _, output, _ = self.run_linked(np.ones(60, bool), p)
        self.assertEqual(generated, [[0, 30], [30, 60]])
        np.testing.assert_allclose(output['trans'][30, [0,2]], output['trans'][29, [0,2]])

    def test_short_segment_created_by_a_true_jump_is_also_filtered(self):
        p = observed_body(60)
        p[7:, 62, 1] += .7
        generated, _, output, saved = self.run_linked(np.ones(60,bool),p,cut=7,hamer=True)
        self.assertEqual(generated, [[7,60]])
        self.assertFalse(output['valid_frames'][:7].any())
        self.assertEqual(saved['discarded_segments'][0]['source_frames'], [0,7])

    def test_unseen_wrist_does_not_hide_a_jump_in_the_visible_other_arm(self):
        p = observed_body(60)
        p[:,62,1] = -5
        p[30:,62,1] = -10
        K = np.tile([[100.,0,50],[0,100.,50],[0,0,1]],(60,1,1))
        shots = [dict(start=0,end=30),dict(start=30,end=60)]
        self.assertFalse(continuity_breaks(p,shots,K=K,image_size_hw=(100,100))[0]['broken'])
        p[30:,41,1] += .7
        self.assertTrue(continuity_breaks(p,shots,K=K,image_size_hw=(100,100))[0]['broken'])

    def test_unseen_torso_anchor_cannot_rotate_the_visible_arm_basis(self):
        p = observed_body(60)
        p[30:,9,1] = -10
        generated, _, _, _ = self.run_linked(np.ones(60,bool),p)
        self.assertEqual(generated, [[0,60]])

    def test_broken_cut_holds_heading_and_position_without_flattening_height(self):
        first, second = motion(20, 0., 2.), motion(20, 1., 9.)
        second['trans'][:, 1] = 1.2
        output, placements = linked._join_motion([(0,20,first,False),(20,40,second,True)], np.ones(40,bool), 100)
        self.assertAlmostEqual(float(output['poses'][20,0,1]), 0., places=6)
        np.testing.assert_allclose(output['trans'][20,[0,2]], output['trans'][19,[0,2]])
        self.assertAlmostEqual(float(output['trans'][20,1]), 1.2, places=6)
        self.assertEqual(placements[0]['source_start'], 120)

    def test_short_visibility_islands_never_reach_generation_or_hamer(self):
        valid = np.zeros(60, bool)
        valid[:20] = valid[25:27] = valid[35:55] = True
        generated, state, output, saved = self.run_linked(valid, hamer=True)
        self.assertEqual(generated, [[0,20], [35,55]])
        self.assertFalse(output['valid_frames'][25:27].any())
        self.assertEqual(state['hamer_report']['frames'], int(output['valid_frames'].sum()))
        self.assertEqual(saved['discarded_segments'][0]['source_frames'], [25,27])

    def test_all_fragments_rejected_before_camera_or_model_work(self):
        valid = np.zeros(60, bool)
        valid[1:3] = valid[35:42] = True
        with self.assertRaisesRegex(ValueError, '15'):
            self.run_linked(valid)

    def test_hamer_counts_all_segments_once_and_persists_the_same_report(self):
        valid = np.ones(60, bool)
        valid[20:35] = False
        generated, state, output, saved = self.run_linked(valid, hamer=True)
        self.assertEqual(state['hamer_report']['frames'], 45)
        self.assertEqual(state['hamer_report']['left'], 45)
        self.assertEqual(state['hamer_report']['right'], 43)
        self.assertEqual(state['hamer_report']['seconds'], 2.5)
        self.assertEqual(saved['hamer_report'], state['hamer_report'])
        self.assertEqual([r['source_frames'] for r in saved['hamer_segments']], generated)


class AutoCutFallbackTests(unittest.TestCase):
    def run_original(self, detector_modules=None, detector_patches=(), check=None):
        n = 20
        track = TrackInfo(0, np.arange(n), np.tile([0,0,90,90], (n,1)).astype(np.float32))
        result = TrackingResult([track], n, 100, 100, (0,n-1))
        proc = Mock()
        proc.detect_and_track.return_value = result
        proc.compute_camera_K.return_value = (torch.eye(3).repeat(n,1,1), None)
        extractor = SimpleNamespace(extract_video_tokens=Mock(return_value=torch.zeros(n,3072)))
        engine = SimpleNamespace(proc=proc, sam_extractor=extractor, device=torch.device('cpu'),
            sam_cache_backend='gpu', _activate_stage=Mock(), release_stage=Mock(),
            _generate_video_motion=Mock(return_value='generated-original'))
        state = {}
        with TemporaryDirectory() as temp, ExitStack() as stack:
            video = Path(temp)/'source.mp4'; video.write_bytes(b'fixture')
            proc.transcode.return_value = str(video)
            stack.enter_context(patch.object(VideoProcessor, 'get_video_meta',
                return_value=SimpleNamespace(total_frames=n,height=100,width=100)))
            if detector_modules is not None:
                stack.enter_context(patch.dict(sys.modules, detector_modules))
            for context in detector_patches:
                stack.enter_context(context)
            outcome = V2MInferenceEngine.process_video(engine, str(video), temp,
                shot_linking='auto', static_camera=True, end_frame=n-1,
                details_callback=lambda values:state.update(copy.deepcopy(values)), cancel_check=check)
            self.assertFalse((Path(temp)/'source/shots.json').exists())
        self.assertEqual(outcome, 'generated-original')
        engine._generate_video_motion.assert_called_once()
        return state

    def test_missing_scenedetect_uses_original_pipeline_and_reports_reason(self):
        state = self.run_original({'scenedetect':None})
        self.assertEqual(state['shot_linking_report']['mode'], 'original')
        self.assertEqual(state['shot_linking_report']['reason'], 'shot_detection_unavailable')

    def test_incomplete_cut_decode_uses_original_pipeline_and_releases_stream(self):
        stream = Mock(frame_number=12)
        manager = Mock()
        state = self.run_original(detector_patches=(
            patch('scenedetect.open_video',return_value=stream),
            patch('scenedetect.SceneManager',return_value=manager)))
        self.assertEqual(state['shot_linking_report']['reason'], 'shot_detection_incomplete')
        stream._cap.release.assert_called_once()

    def test_cancellation_is_not_swallowed_by_fallback(self):
        class Cancelled(ValueError):
            pass
        stream = Mock(frame_number=12)
        manager = Mock()
        manager.detect_scenes.side_effect = lambda source,**kwargs:source.read()
        with self.assertRaisesRegex(Cancelled, 'cancelled'):
            self.run_original(detector_patches=(
                patch('scenedetect.open_video',return_value=stream),
                patch('scenedetect.SceneManager',return_value=manager)),
                check=Mock(side_effect=[None,Cancelled('cancelled')]))
        stream._cap.release.assert_called_once()

    def test_unexpected_detector_error_is_not_silenced(self):
        manager = Mock()
        manager.detect_scenes.side_effect = RuntimeError('unexpected detector bug')
        with self.assertRaisesRegex(RuntimeError, 'unexpected detector bug'):
            self.run_original(detector_patches=(
                patch('scenedetect.open_video',return_value=Mock(frame_number=20)),
                patch('scenedetect.SceneManager',return_value=manager)))


if __name__ == '__main__':
    unittest.main()
