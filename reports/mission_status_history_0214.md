# Mission state

Started 2026-10-18 approximately 01:11 Europe/Istanbul (2026-10-17 22:11 UTC).
Persistent Codex goal registered. Weekly quota baseline: codex primary 10080-minute window,
usedPercent=0, resetsAt=1792876221. Authorized additional budget: 35 percentage points.
Check actual account usage at milestones; do not substitute token counts or time.

The goal references a missing 10H specification; the provided 5H-named specification
is the available authoritative technical document. Its wall-clock titles do not limit execution.
No AGENTS.md found in workspace or ancestor locations. Workspace is not a git repository.

## Verified assets / environment
- `.venv/Scripts/python.exe`: Python 3.12.13; Torch 2.12.0+cu130; RTX 4060.
- OpenCV 5.0.0 from external custom CUDA build, one CUDA device. Full initial
  fingerprint in `reports/environment_initial.json`. Do not install OpenCV wheels.
- Supplied detector ONNX: YOLO26m, static 1x3x416x704, end-to-end output 1x300x6,
  five specified classes. Native TensorRT engine cannot deserialize: version 243 vs 245.
  Use direct ONNX Runtime CUDA first; optionally rebuild TRT locally.
- Local QUT RGB sequences and centroid/range labels available on F:. Source explicitly
  stationary grounded camera, approaching target versus additional non-near-course aircraft.
  These are encounter-category proxy labels, not physical collision truth.

## Milestone (approximately 01:30 Europe/Istanbul)
- Actual weekly tool reading now 2% used, +2 percentage points from baseline (35 authorized).
  Recorded in reports/usage.jsonl. Mission remains ACTIVE; do not stop at minimum acceptance.
- Direct TensorRT local engine built successfully in 110.09s. ORT CUDA failed cuDNN
  sublibrary mismatch; no packages modified. Custom CUDA OpenCV still imports unchanged.
- Runnable CLI `python -m airavoid`: actual per-object drift/growth risk, conservative
  tracker, camera diagnostic/stabilization, Russian H.264 explanatory renderer and traces.
- 15 QUT clips prepared (last 35s each); exact IDs and SHA256 manifests in data/qut.
- Actual demos: demo/M2_explained/annotated.mp4 (two simultaneous states around15.5s),
  demo/M1_tiled_initial/annotated.mp4 (missed tiny negative),
  demo/formation_explained/annotated.mp4 (airborne turns cause abstention).
- Inspected actual frames and H.264 codec/duration. Inset obstruction corrected with
  separate zoom strip; newer renderer canvas1800x1140. Need inspect latest outputs again.
- M1 tiled original run 10.15 FPS incl neural inference+render+export; cached replay
  FPS is NOT inference throughput. Final clean sequential benchmark still needed.
- Development M1/M2 used for architecture; initial heldout M3-M5 v1 results retained.
  M4 has five FALSE NON-THREAT frames from detector span contraction. Never conceal.
- Added temporal confirmation (.24s warn,1.2s nonthreat), 13 tests pass. M4 replay
  removes false-clear episode but reduces negative decision coverage. This is diagnosed
  data, not untouched validation. v1 source frozen artifacts/source_v1; v2 frozen
  artifacts/source_confirmed_v2. New run reports hash Python sources.
- 1200 analytic 2D pinhole cases show high abstention; no false-clear in that small
  sweep. These are analytic tests, not real-video collision evidence.
- Oracle GT-centroid/identity + detector-span ablation and offline range-timescale
  comparison in reports/oracle_analysis.json. GT never enters the live CLI.

## Latest milestone — 01:37–01:42 Europe/Istanbul
Weekly account usedPercent=4, additional4 percentage points (goal ACTIVE, target35).
Quota is account-wide; attribution precision is integer percentages. Tool clock at
01:37:08 was22:37:08UTC on17Oct. Start01:11 on18Oct. Current tests19 pass.

Completed confirmed heldout M6,M7,M8,S1..S5:8/8 approach events warned>=.8s,
0 false-clear frames.3504 approach opportunities1144threat/1623uncertain/737untracked.
Only2 negative events:394opportunities173nonthreat/217uncertain/4untracked,0warnings.
Do not inflate event denominators or describe labels as physical collision truth.
S6/S7 still reserved, unexamined.

New work:
- Experimental exact joint inverse-span estimator `--geometry projective` added;
  analytical intercept/tau/focal invariance pass, no clear real-video gain over log.
  Default remains log. Dev ablations in reports/projective_window_ablation.json.
- Camera2D stress exposed stabilization reference bug. Fixed with evidence reset
  and new coordinate epoch when camera validity is lost/reacquired; regression tests.
  Corrected stress reports/camera_stress_v2.json (all8cases complete). Auto stabilized
  pan median error0.825px within each valid reference epoch; high abstention. Old
  stress report's errors used initial gauge across unobserved gaps and are not a
  directly fair numeric comparison. Actual stale evidence mixing was nevertheless a bug.
- Corrected arbitrary aspect ratio handling: camera resize preserves aspect; renderer
  letterboxes source and maps all target overlays consistently.
- Conversion check18frames:14/14detections match ONNX CPU vsTRT, median center0.14px.
- Adaptive scheduler v1 reduces network calls to3.08 but destroys negative coverage.
  Memory v2 scheduler retains2 nearby crops for10calls and scansall every5calls;
  mean4.56(M1)/4.69(M2) vs7dense. M2 matched negative nonthreat80 (same dense),
  detector134 vs137; M1 negative detections27 vs30. Keep experimental until more
  validation. `--adaptive-tiles`. Source snapshot artifacts/source_adaptive_v1.
- Actual confirmed neural+render M2 demo: demo/M2_confirmed/annotated.mp4, screenshot
  simultaneous.png at15.52s. Throughput4.2FPS was during heavy parallel CPU stress,
  not a clean hardware benchmark. Earlier dense actual10.15FPS also exists.
- Growth-only evaluation baseline now matches confirmation policy. M2 confirmed:
  approach258 baseline warnings vs210selected; negative122baseline vs0selected.
  Baseline negative event gets sustained false warning. Rerun reports as needed.
- Source TIFF audit found gaps inM5,S1,S2,S3; no gaps in the last35s clips. Full
  preparation repeats preceding frame at a missing source ID and records it,
  preserving25FPS timeline and source-aligned GT. Report qut_source_audit.json.
- AIR2LAND meta inspected via sibling existing SciPy, no deps installed. Pose data
  exists but crash folder is not a verified camera collision; don't use as event truth.

## Running background work
1. Full M1–M8 TIFF→MP4 preparation: exec session46361,
   `python scripts/prepare_qut.py --sequences M1 M2 M3 M4 M5 M6 M7 M8 --full
    --output data/qut_full`, log reports/prepare_full.log. Source I/O is slow.
2. Full evaluation waits for each completed manifest, then dense model/confirmed
   policy/stride5, `python -m scripts.full_campaign`, log reports/full_campaign.log.
   Aggregates artifacts/full_campaign/full_summary.json. Avoid changing inference
   modules during this campaign: each sequence starts a new subprocess. Copy/freeze
   source if an algorithm change must run concurrently. Do not stop preparer.

## Latest milestone — approximately01:50 Europe/Istanbul
Account usage now6% (+6 points of35 authorized), goal ACTIVE. Full preparation and
full dense evaluation COMPLETED; background sessions46361/8130 finished. Source
26,684 timeline frames /1,067.36s in8Msequences;5,340 sampled frames(stride5).
8approach events all sustained warning,0 false-clear frames.13other events:
0sustained warnings selected vs9growth-only baseline;2brief warning frames onM6.
2128approach target-frame opportunities441threat/1319uncertain/368untracked;
3273negative175nonthreat/1839uncertain/1257untracked/2threat. Truecollision/CPA
labels unavailable. Source split/proxy/visibility caveats remain.

`reports/results_overview.png` and PDF are real Matplotlib figures, Russian primary;
generated with sibling anti_collision_system env (already has matplotlib), no package
changes. Inspect latest Russian chart again if necessary. Charts use actual JSON.

Adaptive memory validation M3–M8 COMPLETE in artifacts/adaptive_memory_validation:
2628approach902threat/1653uncertain/73untracked;1458negative203nonthreat/647uncertain/
608untracked/0threat. Compare paired dense runs before changing default.

Analytical wide-bearing test found16false-clear examples across both log/projective
methods at80–85deg with perfect resolved pinhole observations. This disproves any
unqualified claim of physical non-threat from plane-intercept. Documented GEOMETRY.md
and demo guide. Do not hide this failure behind the successful QUT proxy scores.

LoRAT exploration:
- Existing FastVision engine also TRT243 vs245 incompatible. Rebuilt supplied
  FP16 ONNX as artifacts/lorat_local.engine in38.59s. No dependency installation.
- `scripts.sot_trial` uses oracle-selected MODEL seed only for isolated SOT test,
  not end-to-end results.4casesM1/M2 approach/other. M2 other:194/194 visibleGT
  frames within20px, median3.22px; M1 other drifts(p95160px). Existing external CUDA
  adapter imported read-only; source hashes in reports/sot_trial.json.
- Added optional GT-free `--appearance-bridge --adaptive-tiles`, detector every3
  processed frames, max3activeSOTs, rawscore>=.5 (uncalibrated). Actual YOLO boxes
  stored separately from SOT predictions/combined observations; source labelSOT/det
  visible and in traces. External code dependency only for this optional variant.
- Initial bridge duplicate bug: same object changed bird→airplane class, creating
  duplicate SOT/det observations and repeated ambiguity. Diagnosed actual cache;
  class-agnostic spatial merging fixed it, regression test added. Do not interpret
  class as harmlessness. Nonfinite predictions rejected.
- Corrected bridge dev M2:231threat/192uncertain/15untracked approach vsdense210/167/61;
  negative83nonthreat/71uncertain/45untracked vsdense80/89/30. M1negative135tracked
  uncertain vsdense35; approach113warnings vsdense118. Mixed, not universal win.
- Optional `--detector-scale-only` uses appearance centroids but only independent
  detector span samples, preventing tracker-box growth from inventing expansion.
  Tests demonstrate distinction. Dev M2 warnings172/negative23 (more conservative),
  M1warnings107/negativealluncertain. Not selected default; precision vscoverage
  tradeoff unvalidated. Default (flagabsent) geometry behavior retained.
- Current25 tests pass. Main entry `main.py` now invokes CLI.

## Currently running
`python -m scripts.batch_evaluate --sequences M3 M4 M5 M6 M7 M8 S1 S2 S3 S4 S5
 --adaptive-tiles --appearance-bridge --root artifacts/bridge_validation`,
execsession64221, log reports/bridge_validation.log (aroundS3atlastcheck).
Keep variant configuration stable until completion; optional-independent-scale
addition does not apply to these runs. S6/S7 remain untouched final samples.

New playable current-policy demos: demo/M2_confirmed/annotated.mp4,
demo/S2_difficult/annotated.mp4, demo/formation_reference_reset/annotated.mp4.
S2early.png visually inspected: visible glint missed, panel truthfully says no
detection is not safety. All Russian layout readable, zooms outside source view.

Next priorities: bridge/full-encounter paired assessment; separate scale fusion
scientific tradeoffs; additional genuine airborne/helicopter qualitative clip;
wide-angle/calibration sensitivity without claiming unknown intrinsics; final
untouched S6/S7; clean sequential render+inference benchmarks; polished gallery,
programmatic playback/trace checks, final OpenCV fingerprint. Continue research
and experiments toward actual35% usage budget; do not stop merely at this milestone.

## Next highest-value experiments
1. Inspect latest corrected video frames and create Russian gallery preview.
2. Finish heldout confirmed evaluation and report denominators/coverage/false-clear.
3. Compare adaptive memory scheduler on additional existing sequences, reserveS6/S7.
4. Investigate wide-angle calibration sensitivity: plane-crossing proxy differs from
   trueCPA for large lateral velocity; can produce falsely reassuring negative
   decisions at large bearing. Build analytic2D sensitivity test before changing rule.
5. Broaden full-sequence negative event coverage (preparation/campaign running).
6. Improve tiny-target continuity if evidence warrants; no classic-only sky project.
7. Clean sequential runtime benchmark after CPU/GPU workloads are idle.
8. Rigorous video validation, dependency final fingerprint, complete docs.

Exact commands are in README.md and scripts. Resume by reading this file,
EXPERIMENTS.md and logs; do not replan from scratch or pretend quota exhausted.
