import unittest

from flowhmr.multishot.timeline import preview_shot_ids


class PreviewTimelineTest(unittest.TestCase):
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
