# GeoCalib dependency and quality contract

GeoCalib supplies image-gravity evidence for recovered source cameras. Basic
motion generation remains available without the checkpoint, but fallback camera
results must be identified as degraded. Body reconstruction, contact correction
and the camera fitting/gating algorithms are unchanged by this integration.

## Installation and models

- Inference source: the unmodified subset in `third_party/geocalib_inference`,
  pinned to `cvg/GeoCalib@1dedb29f94fc11b896c5f572005978aeebccd67f`. Its license,
  provenance and file hashes are included. Research checkouts are not searched.
- Runtime: `kornia==0.8.1`, with `kornia_rs==0.1.9` in the Windows lock file.
  Installation checks import the actual bundled implementation. Release staging
  rejects missing or mismatched dependencies and requires them in the environment
  being copied; a developer PYTHONPATH supplement is not a complete release input.
- Model: `models/geocalib/pinhole.tar`, 116,074,121 bytes, SHA-256
  `86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac`.
  Source: https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar.
  Models supports normal public download, verification and folder import of
  either `pinhole.tar` or the official `geocalib-pinhole.tar`. This is a PyTorch
  checkpoint, not an archive that the user should unpack. The command-line
  default/perception download path includes it too.
- The model is optional for motion, explicitly labeled for source-camera ground
  tilt. Inference checks the pinned digest before loading and always uses the
  local checkpoint; it does not invoke upstream automatic weight downloading.
- Portable source selection includes the vendored source. Existing program
  updates replace `third_party` and the base runtime's `Lib`, including Kornia;
  model weights and device environments remain separate managed data.

## Completion and degradation

`camera_state=ready` means the camera was saved successfully. It is not a quality
guarantee. Camera status, the job response, viewer camera JSON and exported
reports include `quality`: `status`, `degraded`, `reason`, `fallback_shots` and
`total_shots`. Shot identifiers are the report's original IDs, not list offsets.

New recovery reports also retain `gravity`: the component revision, model hash,
cache reuse, availability, failure category and technical failure detail where
applicable. Reasons distinguish missing/corrupt weights, missing runtime,
unsupported device, inference failure, static-camera mode and unstable image
measurements. Cancellation still cancels. Cache reuse requires the same source
revision and checkpoint and does not bypass readiness checks.

The task completion summary and source-camera viewport display persistent,
localized warnings. Older reports derive the warning from their recorded
`gravity_source`; reading them does not rewrite stored jobs. A report with no
quality evidence is marked unverified. Shared-scene cameras retain their
separate shared-world contract. Installing GeoCalib does not retroactively
recompute an existing camera. Static-camera mode retains its previous body/ground
leveling behavior and explicitly reports that image gravity was not used.

## Verification on 2026-11-16

- 179 automated tests passed, including missing/same-size-corrupt checkpoints,
  loading errors, cancellation, official import names, CLI download, completed
  task/HTTP/MCP/export warnings, and historical result reads without mutation.
- Seven UI languages / 1,005 keys and placeholder checks passed; JavaScript
  syntax and fractional-frame/gap/cut sampling checks passed.
- Fresh official checkpoint download and local folder import passed with the
  published size and SHA-256. Fresh installed Kornia wheels were used in an
  isolated product supplement, without changing the shared legacy environment.
- Real fresh GeoCalib inference and camera recovery ran on a frozen 3,056-frame
  motion (2,950 valid frames). Of 53 fitted shots, 52 used image gravity and one
  used the existing stability fallback. Missing and truncated weights in separate
  model directories each produced explicit full fallback. All three motion
  files stayed byte-identical; this was camera-only replay, not new motion inference.
- Real UI checks confirmed model readiness and distinct missing, damaged,
  unstable and historical warnings. Installation checks reported available
  image gravity and passed CUDA matrix multiplication/NMS.
- The upstream NMF module initializes random bases. The fresh unseeded run is not
  bit-identical to old cached gravity. With seed 31417, original and vendored
  source gave exactly equal gravity vectors at source frames 90, 1596 and 2838.

This is integration validation, not a clean-machine release sign-off. Building
and installing the complete portable release and exercising a real old-to-new
update remain release gates. CPU camera recovery, ROCm and XPU quality have not
been newly validated. Earlier visual/numerical acceptance conditions stay open.
