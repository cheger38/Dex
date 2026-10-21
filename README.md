# Dex

Dex is a camera-based hand-tracking playground. It estimates hand motion and
pinches with MediaPipe, calibrates a webcam, and lets you build a grid of
perspective-projected cubes by pinching and dragging. The playground is an
experiment: hand depth is inferred from apparent palm width rather than measured
by a depth camera.

## Set up

Use Python 3.14 and a webcam that can provide MJPEG 1280×720 at 30 FPS. From the
project root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The included `models/gesture_recognizer.task` is the gesture model supplied by
[Google's MediaPipe project](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python).
Camera-specific calibration files are kept outside the public Git repository.

## Calibrate a camera

Print a 10-by-7-square checkerboard with 23 mm squares (9 by 6 inner corners).
After setup, run from the project root:

```sh
.venv/bin/python src/calibrate_camera.py --camera-index 0 --camera-name webcam
```

The calibrator and hand preview require MJPEG 1280×720 at 30 FPS. They stop with
an error if the camera negotiates another mode or returns a different frame size.
The calibrator automatically attempts a capture every two seconds, keeping only
views where it recognizes the board. Move and tilt the checkerboard to capture
varied views.
Set a different interval with `--capture-interval SECONDS`. Space pauses or
resumes capture, `R` clears the views, and `C` calibrates after at least 15 valid
views. Review the live raw and undistorted images and RMS reprojection error,
then press `S` to save, `R` to restart, or `Q`/Escape to quit without saving.
The example writes `data/calibrations/webcam-1280x720.yaml`; use `--force` to
replace an existing file. Calibration is specific to the physical camera and
frame size. The hand preview loads this file by default. You can also run the
preview without calibration; it will use an assumed field of view.

## Run the preview

After setup, run from the project root:

```sh
.venv/bin/python src/preview_hand_observation.py
```

To use a different camera or model:

```sh
.venv/bin/python src/preview_hand_observation.py --camera 1
.venv/bin/python src/preview_hand_observation.py --model /path/to/gesture_recognizer.task
.venv/bin/python src/preview_hand_observation.py --no-palm-filter
.venv/bin/python src/preview_hand_observation.py --calibration /path/to/camera-1280x720.yaml
```

The preview undistorts each frame before mirroring it for hand tracking and
display. It uses the saved focal lengths and principal point for camera-space
estimates and cube projection. The default calibration is
`data/calibrations/webcam-1280x720.yaml`; `--calibration` selects another file.
If the file is missing, malformed, or for another resolution, the preview warns
once and continues with uncorrected video and its assumed 60° field of view.

In the preview, press `Tab` to switch between hand diagnostics and the action
playground, `i` for image coordinates, `w` for world coordinates, `1` or `2` to
select a hand, and `q` to quit. The playground starts with a 5 cm cube centered
in the image, 0.5 m from the camera, with its front face parallel to the camera.
Move the fingertip midpoint near a cube at roughly the same depth to highlight
it. Pinch while highlighted and drag to add cubes along one of its six face
directions. Dragging toward an occupied row removes cubes one at a time. The
chosen direction stays fixed until release, and moving back shortens the live
preview. Opening the fingers commits the edit; losing tracking cancels it.
Released cubes remain selectable for further edits. Swipes do not change the
scene. If depth becomes unavailable during a pinch, the preview pauses. The
playground also updates while viewing diagnostics. Physical depth remains
approximate because the estimator uses an assumed palm width; camera geometry
uses an assumed field of view when calibration is unavailable.
Palm landmarks are filtered for motion and orientation estimates by default;
`--no-palm-filter` shows estimates from raw landmarks for comparison. The
landmarks drawn in the preview and listed in the panel are always raw.
Recognized actions appear for two seconds in either view.

To tune smoothing in code, pass `OneEuroPalmFilter(OneEuroSettings(...))` to
`PalmMetricsEstimator(palm_filter=...)`. Pass `None` to bypass it, or supply
another object with `apply(timestamp_ms, hand)` and `reset()` methods.

## Recognize actions

Feed each metrics snapshot to an `ActionRecognizer` after updating the estimator:

```python
estimator.feed(timestamp_ms, hands)
recognizer.feed(estimator.metrics())
for event in recognizer.events():
    handle_action(event)
```

A `Swipe` reports its peak 2D palm velocity in normalized image coordinates per
second (positive x is right, positive y is down). Its magnitude is the speed of
that peak. `events()` returns the events from the latest feed, so reading it
again does not consume them.

`PinchStart` and `PinchEnd` report when the thumb and index fingertip gap closes
and opens, using different thresholds to avoid flicker. `PinchLost` reports that
tracking stopped during an active pinch. Each pinch event includes the fingertip
midpoint in normalized image coordinates. Pinch and swipe events can occur in
the same update.
`active_pinch()` returns the current confirmed pinch location after each feed,
or `None` when no pinch is active. Consumers can use it to follow a held pinch
without treating every position update as an action event.

## Check the code

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

This repository has no software license.
