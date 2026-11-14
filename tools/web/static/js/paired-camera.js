// Full pixel intrinsics, independent of the viewer's free-orbit projection.
export function placedCameraTransform(transform, yaw = 0, offset = [0, 0, 0]) {
  const { scale: k, translation: b } = transform;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const rotated = [c*b[0] + s*b[2], b[1], -s*b[0] + c*b[2]];
  return { scale: k, translation: rotated.map((x, i) => x + (1-k)*offset[i]) };
}

export function applySourceCamera(camera, data, frame, transform = { scale: 1, translation: [0, 0, 0] }) {
  if (!data || frame < 0 || frame >= data.RT.length || !data.camera_valid[frame]) return false;
  const R = data.RT[frame], K = data.K[frame], [h, w] = data.image_size_hw;
  const { scale: k, translation: b } = transform;
  const C = [0, 1, 2].map((j) => -(R[0][j] * R[0][3] + R[1][j] * R[1][3] + R[2][j] * R[2][3]) * k + b[j]);
  camera.matrixAutoUpdate = false;
  camera.matrix.set(R[0][0], -R[1][0], -R[2][0], C[0],
    R[0][1], -R[1][1], -R[2][1], C[1], R[0][2], -R[1][2], -R[2][2], C[2], 0, 0, 0, 1);
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrix).invert();
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  const near = .001, far = 10000;
  camera.projectionMatrix.set(2 * K[0][0] / w, -2 * K[0][1] / w, 1 - 2 * K[0][2] / w, 0,
    0, 2 * K[1][1] / h, 2 * K[1][2] / h - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return true;
}

export function sourceViewport(width, height, imageSize) {
  const aspect = imageSize[1] / imageSize[0];
  const w = Math.min(width, height * aspect), h = w / aspect;
  return [(width - w) / 2, (height - h) / 2, w, h];
}
