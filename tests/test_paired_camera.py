from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from flowhmr.multishot.camera import Camera, project
from flowhmr.multishot.camera_vmd import export_camera_vmd, approximate
from flowhmr.multishot.timeline import held_indices, display_keys, intervals


class PairedCameraTest(unittest.TestCase):
    def camera(self):
        rt = np.tile(np.eye(4), (6, 1, 1))
        rt[:, :3, 3] = [0, 0, 5]
        K = np.tile(np.array([[960., 0, 960], [0, 960, 540], [0, 0, 1]]), (6, 1, 1))
        return Camera(rt, K, np.array([False, True, True, False, True, True]),
                      np.array([-1, 1, 1, -1, 2, 2]), np.array([1080, 1920]), 137)

    def test_camera_and_points_share_affine_transform_including_k_one_height_shift(self):
        camera = self.camera()
        points = np.tile(np.array([[-.4, .3, 0], [.5, 1.4, .2]]), (6, 1, 1))
        uv, _ = project(camera.RT, camera.K, points)
        for scale in (1., .6, 1.7):
            shift = np.array([.2, .23, -.1])
            changed = camera.transformed(scale, shift)
            other, _ = project(changed.RT, changed.K, points * scale + shift)
            np.testing.assert_allclose(other, uv, atol=1e-10)
        np.testing.assert_array_equal(camera.RT[:, :3, 3], np.tile([0, 0, 5], (6, 1)))

    def test_hold_entry_and_later_gap_and_display_boundaries(self):
        mask = np.array([False, False, True, True, False, False, True])
        self.assertEqual(held_indices(mask).tolist(), [2, 2, 2, 3, 3, 3, 6])
        self.assertEqual(intervals(mask), [(2, 4), (6, 7)])
        keys = display_keys(mask, 10, 20)
        self.assertEqual(keys, {0: False, 9: False, 10: False, 11: False, 12: True,
                                13: True, 14: False, 15: False, 16: True, 17: False})

    def test_camera_vmd_uses_source_frames_and_no_gap_keys(self):
        camera = self.camera()
        joints = np.tile(np.array([[0., 0., 0.], [.1, .5, 0.]]), (6, 1, 1))
        camera.K[1, :2, :2] *= 1.1
        camera.K[4, :2, :2] *= 1.3
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'camera.vmd'
            report = export_camera_vmd(camera, joints, path)
            data = path.read_bytes()
        # Independent binary offsets from installed mmd_tools format.
        self.assertEqual(struct.unpack_from('<III', data, 50), (0, 0, 4))
        frames = [struct.unpack_from('<I', data, 62 + i * 61)[0] for i in range(4)]
        self.assertEqual(frames, [138, 139, 141, 142])
        fov = [struct.unpack_from('<I', data, 62 + i * 61 + 56)[0] for i in range(4)]
        self.assertEqual(fov[0], fov[1])
        self.assertEqual(fov[2], fov[3])
        self.assertEqual(len(report['shots']), 2)
        rt, K, *_ = approximate(camera, joints)
        before, _ = project(camera.RT, camera.K, joints)
        after, _ = project(rt, K, joints)
        np.testing.assert_allclose(after[camera.camera_valid], before[camera.camera_valid], atol=1e-10)

    def test_reject_bad_camera_and_mask_before_export(self):
        camera = self.camera()
        with self.assertRaises(ValueError):
            camera.transformed(float('nan'))
        with self.assertRaises(ValueError):
            Camera(camera.RT, camera.K, np.ones(5, bool), camera.shot_ids, camera.image_size_hw)
        with self.assertRaises(ValueError):
            held_indices(np.zeros(3, bool))

    def test_invalid_camera_samples_do_not_require_a_rotation_or_focal(self):
        camera = self.camera()
        camera.RT[~camera.camera_valid, :3] = 0
        camera.K[~camera.camera_valid] = 0
        joints = np.tile(np.array([[0., 0., 0.], [.1, .5, 0.]]), (6, 1, 1))
        with TemporaryDirectory() as temp:
            report = export_camera_vmd(camera, joints, Path(temp) / 'camera.vmd')
        self.assertEqual(len(report['shots']), 2)


if __name__ == '__main__':
    unittest.main()
