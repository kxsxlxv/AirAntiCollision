# Stage 2 experiment ledger

2026-10-22: User approved the Stage 2 handoff. Work starts on
`stage2-collision-research`, from current main. The handoff contains a pre-existing
user edit; preserve it. Weekly account meter at start: 45% used, 10080-minute
window, reset Unix 1793083556. Stop at 75% used; reserve capacity for tests and PR.
This account-wide delta includes any concurrent usage.

Initial hypotheses, before experiments:

- Full Bauer geometry may resolve off-axis failures of image-plane intercepts
  under its calibrated, fixed-attitude/constant-shape assumptions. Unknown focal
  length and ownship orientation remain missing measurements on NPS.
- Published image-region and looming cues can be reproduced as observables, but
  their apparent-risk semantics need explicit separation from physical CPA.
- Better persistent association should improve evidence continuity; predicted
  tracker boxes must not create new scale observations.
- Detector B may improve tiny fixed-wing recall, with higher input/runtime cost.
  Its training independence is unverified. No held-out detector-B claim allowed.

First checkpoint: verify primary equations and annotation/model contracts, freeze
video-level splits and an executable plan, then commit before estimator redesign.

Equation check: the initial gate note mistakenly described the reference time of
AirTrack's pairwise formula. An independent Z=20−t unit test gives 16 seconds at
t=4, proving the published expression is already referenced to the later sample.
Corrected before any comparative method results; retain this negative check.

2026-10-22 15:30 UTC: User explicitly authorized subagents, revoked the 30-point
usage cap, and imposed a new one-hour execution deadline (16:30 UTC / 19:30
Europe/Istanbul). Account-wide meter then read 50% used. Three subagents own
geometry/falsification, NPS comparison, and integration/runtime; neural GPU runs
remain serial and final timings wait for CPU quiescence.

Full-rate actual NPS inference finished for A and B: 6,644 original frames each,
with checked original PTS. B's provided engine is incompatible with installed
TensorRT (serialized244/runtime245); rebuilt exact supplied ONNX as a local FP32
engine, without package changes. B ONNX contract is800×1440, one exported class
`item`, interpreted only as user-supplied fixed-wing UAV semantics. Ten development
frames show equal accepted output counts/classes versus exact ONNX, maximum
coordinate difference0.0013428px and score difference0.0008156.

Independent detailed JIRS reading recovered the virtual vertical-disc construction
without assuming a known physical height/width ratio. Full 3D Bauer is implemented
and tested; its model-specific approximation and shape failures remain explicit.
The earlier horizontal summary is preserved and its committed source reproducible.
The earlier instantaneous-WCV summary predates the future-horizon repair and is
not claimed to be identical to later results.

NPS held-out identity finding (official TrackEval, fixed .03 operating points):
B+custom outperforms B+TrackTrack/PuTR; A+TrackTrack outperforms A+custom/PuTR.
This rejects universal promotion of the Stage1 neural shortlist. No threshold is
retuned on held-out data. P2 loses recall on Clip020 despite fewer unmatched
predictions overall, so A remains supported five-class default.

2026-10-22, during final integration: the user confirmed P2 was not trained on
NPS and that every target in selected clips001/008/020/032/034/043 is a fixed-wing
UAV. Treat these as explicit user confirmations, not an independent audit of
training manifests or a subtype column that the CSV does not contain.

Predeclared NPS stride1/2/5 plan was narrowed to complete full-rate NPS ID and
threat comparisons plus QUT stride1/2/5. No held-out threshold was tuned. Six QUT
recordings were rerun; this does not reproduce the earlier 15-recording score.
Independent ORB/homography controls use only development032/evaluation043.

The final study distinguishes delivered independent algorithm families from the
unmet goal of three complete RGB-only collision classifiers on uncalibrated
moving-camera recordings. AirTrack is cue-only, NASA has privileged metric state,
and the RV cone is a partial observer diagnostic. None is counted as a third
complete RGB threat architecture.

Final corrected matrix:88 normal/oracle cached replays (94,336 assessed frames),
plus2 independent ORB controls (1,826 frames). Recomputed compensated histories
after fixing valid→invalid camera transitions that had kept the same reference
epoch; raw comparisons were unaffected. All90 replay input/output/source hashes
match current code. A/custom Lai warnings raw/KLT/ORB:032=0/0/19,
043=99/7/23. These are observation-model sensitivity results, not threat accuracy.

Final actual runtime:14 fresh-process runs,2,100 live detector updates and16,050
network calls. All1,500 encoded frames fully decoded with sourcePTS/duration QA.
At1080p A serialNVENC9.80FPS and bounded-threadNVENC11.81FPS; B serial11.55FPS.
These150-frame samples measure offline throughput, not sustained streaming or
per-frame completed encoding latency. Both threaded controls preserve exact
detection/decision hashes. Ten-frame originalONNX/localTRT agreement is functional,
not proof of numerical equivalence on all possible images.

Final verification:116 tests (oneoptionalNVENCskip), then6 media tests with actual
NVENC enabled, allpass; the exact original customCUDAOpenCV version/path/build,
CUDAdevicecount andupload/download equality are preserved. Three subagents have
finished. No package installation/retraining/AOT/simulator occurred.

Default-preservation control: extracted pre-Stage2 commit6f7a076's complete
airavoid source from Git into an excluded directory and replayed150 identical
actualA observations on originalClip032 RGB. Every decision field agrees with
the final default pipeline after excluding only the added null class_display
schema field. No source/model asset was modified; no replayFPS claim is made.

2026-10-22, final delivery checkpoint before Git upload: weekly account meter
73%used, versus45%missionstart and50%at the one-hour override. The28/23point
deltas are account-wide and include any concurrent account work. The user's
quota-cap revocation remains authoritative. All scientific execution and tests
finished before the16:30UTC deadline; acceptance gaps remain explicit.
