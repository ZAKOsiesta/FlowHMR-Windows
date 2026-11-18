import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from flowhmr.multishot.timeline import preview_shot_ids, preview_validity


class PreviewTimelineTest(unittest.TestCase):
    def test_preview_uses_source_indexed_motion_mask_not_held_mesh_samples(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'motion.npz'
            np.savez(path, source_frames=np.arange(42,46), valid_frames=[True,False,False,True])
            self.assertEqual(preview_validity(dict(track_start=42,num_frames=4),path),
                             [True,False,False,True])
            with self.assertRaises(ValueError):
                preview_validity(dict(track_start=41,num_frames=4),path)

    def test_cuts_are_local_to_preview_and_source_indexed_in_report(self):
        meta = {'num_frames': 7, 'track_start': 42}
        job = {'shot_linking_report': {'tracks': [
            {'id': 8, 'start': 40, 'end': 45},
            {'id': 9, 'start': 45, 'end': 48},
        ]}}
        self.assertEqual(preview_shot_ids(meta, job), [8, 8, 8, 9, 9, 9, -1])
        self.assertNotIn('shot_ids', meta)

    def test_single_shot_old_jobs_keep_their_original_metadata(self):
        self.assertIsNone(preview_shot_ids({'num_frames': 100}, {}))
