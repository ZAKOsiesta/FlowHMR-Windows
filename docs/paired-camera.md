# Paired camera export

Completed jobs support **View → Source camera** for the source body and PMX characters. The first click recovers the camera when needed. **Include camera** in VMD, FlowHMR FBX and NPZ export also triggers recovery. The downloaded ZIP contains independent motion and camera files, full camera parameters, a recovery/approximation report and import instructions.

Camera recovery shares the existing job queue, with separate queued/running/ready/failed/cancelled state. Stop affects only the camera when motion is complete. Click Source camera again to retry. Missing outputs are rebuilt; validated raw-camera and SAM caches are reused. Older jobs need their normalized source video and detection boxes. Missing person observations run SAM only, never Flow generation. GeoCalib is optional; absent, unsupported or dispersed gravity uses body-lean continuity across cuts, a five-second leveling window and a ground constraint.

The numerical recovery boundary takes joint arrays, camera matrices, observations and validity. Body-reference selection belongs to the existing inspection loader. No camera operation writes poses, shape parameters or motion files. Failures are reported independently and valid shots remain usable. Without a trusted scale population, unsupported camera motion becomes a fixed person-anchored center with slow rotation. Shots shorter than 12 frames retain only their fixed similarity. Low-visibility person residuals are diagnostic and do not veto the stability policy chosen in V3.

Group scenes persist the exact shared rotation, scale and origin from `build_group_scene`. Lazy export uses that common frame. For old scenes, only the missing common-camera evidence is recovered while saved scene placements, scale and motion files remain unchanged. Multi-person cut linking is not supported. When selected characters have different scales, the camera follows the first selected character. For scene placement `R,p`, its affine shift is `R b + (1-k)p`; no individual actor camera is used as a scene camera.

The service boundary is `CameraTasks.prepare/status/cancel/run`; HTTP routes are `/camera/prepare`, `/camera/status`, `/camera/cancel`, `/camera/report` and `/camera/export`. Preparation returns 202 while queued/running. Existing motion job state stays `done`. A camera file is committed atomically after the frozen motion digest is checked.

## Data contract

`motion.npz` may carry a boolean `valid_frames[N]` and a source-frame `track_start`. Motion arrays retain the original frame count. Before the first valid frame, hold that first pose; later gaps hold the preceding valid pose. The preview hides gaps and VMD writes adjacent display keys at every transition. FBX/VRMA downloads with gaps include a report because a portable display track is not assumed.

`camera/camera.npz` version 1 contains:

| Field | Meaning |
| --- | --- |
| `RT[N,4,4]` | Paired motion world to OpenCV camera, Y-up world in metres |
| `K[N,3,3]` | Full-image pixel intrinsics |
| `camera_valid[N]` | Independent camera validity |
| `shot_ids[N]` | Integer shot identity; no interpolation or smoothing across cuts |
| `track_start` | First track frame in the normalized source timeline |
| `fps` | 30 |
| `image_size_hw[2]` | Source image height and width |
| `version` | 1 |

The preview uses full intrinsics and letterboxing. Its automatic floor reads the
declared final motion ground; it no longer applies the historical 0.015 m display
offset. Linked segments are translated by one constant vertical offset per segment
to their common correction ground, with their native estimates retained in the
report. This is separate from the viewer's manual display-ground control.

## PMX scale and height

Use the actual PMX solver transform, including its hip-height adjustment. If `k` is the target/source leg-length ratio and `a` is the standing-pelvis anchor, the root mapping is `p' = k p + b`, where `b = (1-k)a + [0, targetHipHeight-sourceHipHeight, 0]`. Camera centres and the reference joints use the same `k,b`; camera rotation and intrinsics do not change. This also represents equal leg lengths with different hip heights, for which a scale-about-one-point expression is singular.

These values come from the selected character solver. The user does not enter them. Source-camera preview for FBX/VRM characters remains disabled until their transforms are explicitly supported.

## Formats and import

VMD uses one rounded median vertical FOV per shot. Optical-axis movement compensates the focal change at the median body-joint depth. It preserves dense source-frame keys, omits invalid-camera keys and unwraps angles only within each shot. MMD units are 0.08 m; Z is mirrored. Unequal focal lengths, principal point offsets and depth-dependent perspective differences remain approximations, listed in `export-report.json`.

Camera FBX is a separate Y-up, metre scene. It averages `fx` and `fy`, centres the principal point and reports the loss. NPZ and JSON retain full intrinsics and validity. JSON/NPZ in a PMX export are expressed in the selected character's transformed world.

For Blender mmd_tools, import the original PMX and both VMD files with scale **0.08**, current frame **0**, and frame margin **0**. Enable **Use Property Driver** on the model root so display and IK keys are evaluated. The package does not contain character models or textures. Camera-only gaps cannot be represented as a camera visibility track in VMD/FBX; consult the sidecar rather than interpreting interpolated player frames as valid estimates.

## Verification boundary

The Jimingyue P1 fixture was checked in the visible Blender 5.2.1 window with installed mmd_tools. The original full intrinsics and all valid body frames remain unchanged. Odette PMX was exported in the browser and imported with its camera. At 53 shots × first/middle/last, all 159 raster silhouette bounds fit the measured analytic full-intrinsics-to-VMD approximation plus 2 source pixels. Six gap boundary renders are empty. Camera-only import errors are below 0.002 source pixels for both VMD and FBX.

This is evidence for that fixture and character. The approximation calculation includes focal anisotropy and principal point limitations; it is not a universal bound derived from integer FOV rounding alone. Whole-film visual assessment remains separate. Cold SAM runs can differ numerically even on the upstream baseline, and newly written FBX files can differ in bytes despite matching animation content. See [validation notes](product-validation.md) for the repeatability checks and their limits.
