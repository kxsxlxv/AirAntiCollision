# Research log — primary-source audit, 18 October 2026

## Published claims and dataset semantics
- [Martin, Riseley & Ford (2022)](https://arxiv.org/html/2112.02735v1),
  [QUT dataset](https://open.qcr.ai/dataset/aircraft-collision-course/): actual RGB
  grounded stationary camera, 15 approaches, additional aircraft in M sequences.
  Local label readme explicitly labels approaching versus non-near-course aircraft.
  Accept as encounter-category proxy, not a collision/airborne-ownship benchmark.
  [Mirror](https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset)
  lists BSD-3-Clause. Inspect detailed capture timing before producing MP4s.
- [Bauer et al. (2019)](https://onlinelibrary.wiley.com/doi/10.1155/2019/7191839):
  image position and apparent size support normalized encounter geometry under
  constant velocity, straight flight and stable target shape; least-squares fits.
  Published flight validation does NOT validate this new implementation. Their
  tuned thresholds and calibrated camera are not transferable by assertion.
- [Yu, Sharma, Beard & Taylor](https://www.et.byu.edu/~beard/papers/preprints/YuSharmaBeard__.pdf):
  bearing-only observability depends on motion; unknown range cannot be recovered
  by assigning a detector class a typical physical size. Supports abstention.
- [ViSafe, RSS 2025](https://theairlab.org/visafe/): real flight demonstration,
  multi-camera hardware, multi-view fusion, ownship state and control barrier
  functions. Not a ready drop-in for an uncalibrated single downloaded RGB stream.
- [AVOIDDS (2023)](https://github.com/sisl/VisionBasedAircraftDAA): verified README
  describes 72,000 synthetic detection images plus X-Plane downstream simulation.
  Rejected as primary evidence: static images lack temporal encounters, simulation
  conflicts with mission's no-simulator scope. No simulator operated.
- [Karampinis et al. (2024)](https://arxiv.org/abs/2405.06749): relies on AOT for
  training/evaluation. Excluded under the explicit dataset prohibition.
- [CARE (CoRL 2025)](https://github.com/airlab-sogang/CARE): depth/APF ground-robot
  module, unsuitable evidence for tiny flying targets and aircraft maneuverability.

## Selected hypothesis (not an established performance claim)
Use actual detected centroid drift and apparent span expansion, with uncertainty
and explicit camera-motion validity. Constant bearing plus credible growth supports
a potential warning independent of screen center. Reliable large drift relative to
growth may support an apparently non-threatening image-space pass. Tiny targets,
aspect changes, turns, missing detections and unobservable growth require abstention.
No metric CPA, physical range, probability of collision or assured avoidance is
observable in general. Candidate direction illustration must remain conditional.

## Verified repository behavior / measurements
See `MISSION_STATUS.md`, `reports/environment_initial.json` and subsequent experiments.
Supplied ONNX metadata read directly; TensorRT supplied engine rejected by runtime
with serialization version mismatch. Only av==18.1.0 has subsequently been added
without dependencies; custom CUDA OpenCV was verified afterward.

## Scope limitations
This is an independently assembled research demonstrator, no novelty claim.
No AOT, no 3D simulation, no substantial detector training. Do not equate successful
image tracking, apparent approach or a central screen position with collision truth.

## Further leads inspected, not promoted to threat benchmarks
- [Air2Land](https://github.com/micros-uav/micros_air2land): existing synthetic
  stereo sequences, landing guidance, GPS and gimbal/pose metadata (CC-BY stated
  by authors). Local left RGB/meta files inspected using sibling existing SciPy;
  actual meta contains keypoints, boxes, relative poses, UAV/PTU Euler angles.
  `airport2_crash` is not by itself a verified collision with the observing camera.
  Not used as event truth without deriving camera/object geometry and timestamps.
  No simulator operated or imported.
- Supplied SkySegDINO checkpoint (~299MB) was actually run on M1/S2. A sky-masked
  proposal plus neural zoom experiment improves S2 observation coverage, but the
  separate contribution of semantic masking remains unproven. It does not solve
  expansion/aspect ambiguity. End-to-end two-pass timing is recorded separately.
- Existing FastVision LoRAT224 assets were rebuilt locally and run through a
  read-only adapter. Isolated and integrated trials show mixed continuity gains
  and high-score drift; tracking scores are not visibility or threat probabilities.

## Experimental exact projective hypothesis
When span is stable, `q=1/s`, `p=(u-u0)/s` are linear in time for constant relative
velocity. At the estimated depth-plane time `tau=-q/q_dot`, `p+p_dot*tau` gives
the lateral intercept in span units; arbitrary coordinate origin cancels at q=0.
An implemented joint fit propagates fitted cross-output covariance into sensitivity
intervals. It is not a new published method or calibrated probabilistic estimator.
Exact analytical tests pass; initial real development comparisons give mixed/no
clear performance gain. Kept as `--geometry projective` for reproducible comparison,
with the tested simple default retained. See generated ablation reports.

## Important range-label audit
`reports/range_audit.json` checks every locally supplied target CSV. M3 nominal
approach range rises overall3562→4203, with1071positive differences in2140steps.
This conflicts with treating its entire labeled target interval as a monotonically
closing physical encounter. M2/M5 also include late increasing ranges (possible
pass/recession), so even an event category cannot become per-frame collision truth.
Do not repair or reinterpret these labels by guessing. Keep nominal category metrics
and geometry-valid subsets distinct; do not report trueCPA/impact-time scores.
The target-info.csv mirror was fetched and retained; it provides no intrinsics.


## Additional geometry-data audit (18 October, 02:45 local)
[Vigil Autonomy public trajectories](https://huggingface.co/datasets/vigilautonomy/vigil-public-data)
advertise ten real video streams with timing, position and orientation, CC-BY-4.0.
The actual Hugging Face API reports gated=manual; anonymous raw access returned401.
No access request, account change or gated-data bypass was attempted; no files were
used. This could support geometry validation after ordinary authorized access.
[MMAUD official repository](https://github.com/ntu-aris/MMAUD) provides synchronized
RGB and Leica ground truth, with calibration links. The listed V1 collections are
11.1–19.7GB per drone sequence and ground-based; not used as collision truth here.
[GAREUD official repository](https://github.com/coconut-xia/GAREUD) provides real
RGB/event A2A/G2A detections, timestamped processed RGB and optional stereo calibration
metadata. Its listed real labels are normalized2D boxes, not verified collision/CPA
labels. These leads do not repair the current lack of real metric encounter truth.

## Calibration as an explicit additional assumption
The optional calibrated estimator uses the pinhole equations to fit relative
position and velocity in constant projected-target-span units. It addresses the
wide-bearing depth-plane/CPA distinction in ideal analytical tests. It requires
explicit, resolution-matched square-pixel calibration and rectified fixed-camera
RGB; no QUT focal length is guessed. Aircraft aspect, box bias and nonconstant
relative motion still violate the reconstruction. This is a standard geometric
alternative, with no novelty or real calibrated-flight accuracy claim.


## Background homography is not ownship rotation
[OpenCV5 homography tutorial](https://docs.opencv.org/5.0/tutorials/features/homography/homography.html)
describes plane-induced mappings containing rotation, translation and plane depth,
and multiple decomposition solutions. Our analytical test is an inference from
those projection equations: removing finite-depth background motion can remove
or alter genuine relative target motion. A successful similarity fit alone cannot
validate the physical bearing basis. Actual controlled KLT testing reproduces the
false-pass indication; conservative auto-mode pass states are now withheld.

## Additional existing AIR2LAND measurements
The [authors' repository](https://github.com/micros-uav/micros_air2land) describes
synthetic landing imagery, pose metadata and cameras on pan-tilt units. Our added
experiment is limited to 317 existing paired images and supplied boxes across four
predetermined variant IDs. It does not run their simulator or assume source timing,
intrinsics or pose units. Detection coverage, visibility failures and aspect/range
inconsistency are reported separately from the real QUT encounter-category scores.
