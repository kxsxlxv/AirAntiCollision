# Stage 2 results — 22 October 2026

This stage delivers reproducible comparisons and an opt-in live demonstration.
It **does not establish a physical collision classifier for uncalibrated moving-camera RGB**.
Independent published algorithm families are implemented across the analytical
and recorded-data tiers, but three complete, comparable RGB conflict classifiers
on the normal recordings were not obtained. AirTrack supplies a cue, NASA requires
metric state, and the RV cone is a partial observer diagnostic. These are not
counted as three deployable RGB classifiers. Acceptance is partial and the PR
remains a draft.

## Findings that change an engineering decision

Six complete NPS clips contain 6,644 original frames. Each detector processed every
frame with genuine neural inference. Development clips are 001/008/032; the frozen
evaluation clips are 020/034/043. The user confirmed that all targets in these clips
are fixed-wing UAVs and that neither supplied detector was trained on NPS. These
are user confirmations, not independent training-manifest audits. The CSV revision,
original video, DogFight geometry and source timestamps were checked separately.

At the predeclared .03 thresholds, detector B produces far fewer unmatched boxes,
but loses substantial recall on Clip 020: **64.27% versus A's 91.55%** at IoU .5.
The operating points are not calibrated to equal probability. Unmatched predictions
are not individually proven false positives: annotation completeness was not
independently established for every visible object.

| Frozen evaluation condition | IDF1 | HOTA | Identity switches |
|---|---:|---:|---:|
| A / original tracker | 25.90% | 24.50% | 77 |
| A / TrackTrack | 37.00% | 30.66% | 33 |
| A / PuTR | 19.38% | 15.42% | 124 |
| B / original tracker | **82.59%** | **54.72%** | **5** |
| B / TrackTrack | 77.60% | 50.31% | 14 |
| B / PuTR | 34.60% | 25.71% | 106 |

TrackTrack improves the A condition. With B, the original tracker performs better
on these reserved clips. PuTR loses considerable fresh-observation coverage.
There is no evidence here for a universal neural tracker winner. Official pinned
TrackEval HOTA, Identity and CLEAR implementations are used; emitted MOT boxes and
fresh raw detector boxes are distinguished. TrackTrack's A-condition metric gain
coexists with lower fresh-observation coverage and almost unchanged dominant-ID
coverage. Existing learned checkpoints were not retrained for aircraft. Full detection, localization,
fragmentation, continuity and size/confidence strata are in
[the NPS findings](../reports/stage2_nps_results.md) and
[the machine-readable metrics](../reports/stage2_detection_mot_s1.json).

## Collision methods and truth boundaries

[The evidence inventory](../research/stage2_selection.md) extends the previous
survey with primary equations, code/checkpoint availability, license status and
missing-input conditions. [Method provenance](../reports/stage2_method_provenance.json)
pins author revisions, paper digests and implementation deviations.

- Bauer: full model-based 3D disc/ellipse extension, including the published virtual
  vertical-disc approximation. Verified intrinsics and trajectory/model assumptions
  are required; distance is in target-diameter units without a known target size.
- Lai–Lin: published image-risk factors and weights, with detector rectangle area
  replacing the original silhouette. It is an image-risk adaptation.
- AirTrack: exact published expansion/depth-plane time cue. No independent complete
  collision classifier is invented around it.
- NASA WCV TAUMOD: independently reproduced future-horizon metric-state predicate.
  Only the analytical oracle tier supplies its required state.
- RV-C3BF: nominal calibrated feature-flow/point-cone diagnostic. Full robust
  observer, uncertainty certificate, finite footprint and control solver are absent.
- Existing log, projective, bounded and calibrated methods remain custom baselines;
  growth-only warning is an explicitly trivial baseline.

The [analytical sweep](../reports/stage2_falsification.md) tests **1,599 cases**,
123 independently specified physical trajectories × 13 observation variants.
These are mathematical projections, not recorded flights or simulator evidence.
Bauer has zero critical false-safe in the two ideal-disc variants, with 64 near
encounters each. Across 123 exact disc/ellipse projections, absolute CPA error is
median 0.01000 and maximum 0.09243 target diameters. Its failures under other shapes,
aspect changes and camera perturbations remain in the report. Lai–Lin has 430 raw
critical false-safe cases. Zero false-safe for an always-abstaining cue, or for NASA
with privileged state, does not establish equal-information superiority.

[The geometry review](stage2_geometry_findings.md) explains assumptions,
counterexamples and the reproducible historical checkpoint. All raw failed cases
remain in local artifacts. The runtime adapter withholds Lai low-index clearance
and finite-aircraft clearance from a nominal point cone; raw research outputs
remain inspectable. No method is promoted as a physical RGB collision solution.

## Recorded-video evidence

The normal NPS matrix freezes A/B × original/TrackTrack/PuTR × raw/background
similarity observations, across all six clips. Every assessment method receives
the same acquired timestamps, predicted IDs and actual detector-measured extents
within a condition. Shared confirmation is .24 s for warning and 1.2 s for
nonconflict. Separate oracle-ID and oracle-box+ID experiments use only clips
032/043 and are explicitly privileged offline controls.

The [NPS report](../reports/stage2_nps_results.md) records warning/uncertain/untracked
coverage, raw nonconflict candidates, episodes per camera hour and onset relative
to first/last visible annotation. They are **not collision recall or warning
precision**. Missing intrinsics, target feature flow, attitude and metric state
remain missing. Background-plane KLT and ORB/homography fits test an alternative
image motion model; neither identifies ownship rotation versus translation.
Pixel offsets are not bearing angles without calibration.

With exactly the same A/original-tracker observations, Lai warning samples for
raw/KLT/ORB are **0/0/19** on development Clip 032 and **99/7/23** on evaluation
Clip 043. Thus an alternative camera model changes the apparent evidence; a good
background fit alone cannot select which warnings reflect a physical encounter.

[The QUT comparison](../reports/stage2_qut_results.md) covers M1/M2/M6/M7/S1/S2 at
full rate and strides 2/5, using frozen historical genuine detector outputs and
fresh causal association. The camera diagnostics remain frozen at full rate.
QUT categories have influenced prior development and are proxies; approach does
not prove collision and other does not prove safety. Per-method, per-sequence
counts and annotation-relative warning timing are retained in
[the complete JSON](../reports/stage2_qut_comparison.json). These six recordings do
not reproduce the previous stage's complete fifteen-sequence score.

## Practical delivery and reproduction

The five-class detector A and original geometry remain supported defaults.
`--detector-variant B` selects the validated fixed-wing P2 export; its actual
input is **800×1440**, despite training image size 1440. The supplied engine's
serialization was incompatible, so the exact ONNX was rebuilt locally in FP32.
Ten development frames have identical accepted counts/classes against the original
ONNX; maximum matched coordinate difference is 0.001343 px.

[Runtime instructions and measured results](stage2_runtime.md) give actual CLI
commands, final source/engine digests, separate startup/steady-state timing,
render/export variants, decoded-video and PTS checks. Caches are not counted as
neural end-to-end performance. Jetson Orin Nano Super is unmeasured.

The source recordings, supplied weights, isolated upstream repositories and bulk
traces are local excluded assets. Commands resolve the audited local paths;
reviewers need those same assets and pinned upstream revisions. No absent media
is linked as a GitHub demo. Representative videos can be regenerated using the
runtime commands. The default runtime never opens an identity CSV.

```powershell
# Read-only data audit creates a new protocol; preserve the committed frozen one.
# Detector caches must be regenerated sequentially on the same original sources.
.venv/Scripts/python.exe scripts/stage2_observations.py --detector A
.venv/Scripts/python.exe scripts/stage2_observations.py --detector B

# Existing audited cache paths are the default tracking inputs.
.venv/Scripts/python.exe scripts/stage2_campaign.py
.venv/Scripts/python.exe scripts/stage2_score.py
.venv/Scripts/python.exe scripts/stage2_assessment_campaign.py --oracle --workers 2
.venv/Scripts/python.exe scripts/stage2_camera_ablation.py
.venv/Scripts/python.exe scripts/stage2_camera_threat_control.py
.venv/Scripts/python.exe scripts/stage2_write_nps_results.py
.venv/Scripts/python.exe scripts/stage2_qut.py
.venv/Scripts/python.exe scripts/stage2_write_qut_results.py
.venv/Scripts/python.exe scripts/stage2_falsify.py
.venv/Scripts/python.exe scripts/stage2_provenance.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/stage2_verify_environment.py
# After all jobs and report generators finish:
.venv/Scripts/python.exe scripts/stage2_artifact_manifest.py
```

Run regeneration in a fresh checkout or archive the original local artifacts
first. Detector generation refuses completed output directories by default.
See each script's `--help` for its output switches. Replays validate source
hashes and chronological acquisition times; predicted boxes cannot become fresh
measurements. Reports retain exact source and implementation hashes.

The final regression run contains 116 tests with the optional NVENC test skipped;
the media suite was then run with `AIRAVOID_TEST_NVENC=1` and all six tests passed,
including the actual NVIDIA variable-PTS encode. Results are in
[the regression log](../reports/stage2_tests.log) and
[the hardware integration log](../reports/stage2_nvenc_tests.log).
An additional [historical default regression](../reports/stage2_default_regression.json)
replays the same actual A observations through pre-Stage-2 commit `6f7a076`.
All 150 default decision frames agree, excluding the added null display-label
field. This verifies behavior on that sample rather than asserting universal
equivalence from argument-parsing tests.
[The environment fingerprint](../reports/stage2_environment_final.json) verifies
the exact original CUDA OpenCV path, version, build and upload/download result.
[Final consistency checks](../reports/stage2_validation.json) validate reported
trace and runtime source hashes; [the artifact inventory](../reports/stage2_artifact_manifest.json)
records source/report/local-evidence digests. Passing those checks does not make
the unmet scientific criteria complete.

## Acceptance accounting

| Requirement | Result |
|---|---|
| Extend ≥10-method inventory, ≥5 conflict approaches | Delivered with primary-source and input/availability distinctions |
| Research gate before redesign | Committed at bbdd520 before comparative implementation |
| ≥3 independent published algorithm families | Delivered across input tiers; NASA uses analytical oracle state |
| Three complete comparable RGB collision classifiers on normal recordings | **Unmet**; cue-only, partial and privileged-state methods are separated |
| Physical analytical falsification and raw failures | Delivered, 1,599 cases; full disc-model reproduction and shape/camera failures |
| Independent moving-camera ID/box truth adapter | Delivered; canonical user-reviewed revision, read-only audit and source checks |
| Controlled A/B, original/TrackTrack/PuTR | Delivered at full NPS cadence, fixed operating points and reserved videos |
| Oracle identity/box and alternative camera controls | Focused 032/043 controls; not the entire factorial product |
| Real-video physical CPA / collision accuracy | **Unavailable**; NPS/QUT lack required truth and sensor inputs |
| QUT full/2/5 cadence | Delivered on six explicitly listed recordings |
| Deployment and real GPU timing | Delivered as opt-in research adapters; see measured configuration limits |
| Jetson runtime / flight-ready avoidance | **Unmeasured / not delivered** |
| Preserve custom CUDA OpenCV and original defaults | Verified in the final environment and regression reports |

The user's later instruction revoked the original quota cap and authorized three
subagents for one hour, ending at 16:30 UTC. The account-wide usage record is in
[stage2_quota.json](../reports/stage2_quota.json); concurrent account usage cannot
be attributed to this study. [The ledger](stage2_ledger.md) preserves corrections
and negative findings. No package installation, detector retraining, Amazon AOT
evaluation or flight simulator was used.
