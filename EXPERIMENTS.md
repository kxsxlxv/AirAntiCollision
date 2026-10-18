# Experiments — observed results only

## E00 Environment and model adapter
- RTX 4060; OpenCV 5.0.0 custom CUDA build unchanged. No packages installed.
- Supplied TensorRT engine cannot deserialize in TRT 11.4 (serialization version mismatch).
- Rebuilt supplied FP16 ONNX directly with TensorRT 11.4 in **110.09 seconds**.
  Local engine `artifacts/yolo26m_local.engine`; source weights untouched.
- ONNX Runtime CUDA encountered `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
  Do not represent that backend as working; direct TensorRT run is the chosen path.

## E01 Geometry sanity checks
` .venv/Scripts/python.exe -m unittest discover -s tests -v `
8 tests passed: off-center constant bearing/growth, crossing with growth, stable size,
unknown camera, tiny target, recession, gap/association ambiguity, advisory abstention.
These are analytical checks, not evidence of real encounter accuracy.

## E02 Initial real RGB video
M1 development, last 35 s available locally, actual TIFFs encoded at published 25 FPS.
Original frame IDs and hashes in `data/qut/M1.manifest.json`.
First 240 processed frames at stride 2: `demo/M1_trt_initial/annotated.mp4`.
Inspected preview at 12 s: target aligned, Russian text readable, size instability
correctly causes uncertainty; normal full-frame detector fails to recover the tiny
other aircraft and fragments the approaching target. No successful discrimination
claim yet. Benchmark JSON records actual stages and total throughput.

## Next experiments
1. Tiled supplied detector vs full-frame, same M1 development footage.
2. Diagnose growth instability and tracking gaps without weakening abstention blindly.
3. Offline centroid proxy evaluation with fixed 20 px tolerance, 10/30 sensitivity.
4. More sequences reserved (M3-M8, S1-S7) once method/config are frozen.
5. Independent airborne-camera qualitative footage; unknown-camera stress cases.

## E03 Tiled vs full-frame on M1 development
Same 438 sampled RGB frames (35s, stride2), tolerance20px approximate centroid match:
approach detector matches431 tiled vs379 full; negative30 vs0. Tiled maintains
one associated approach track, fullframe two. Tiled threat121 matched frames vs123
full; no false-nonthreat on approach in either. Other aircraft remains mostly
untracked (403/438 tiled;438/438 full). **Not a detector recall benchmark with GT boxes.**
Tiled actual original full render/export throughput10.15FPS; stage means in JSON.
Fullframe no-render timing is not comparable end-to-end rendering performance.

## E04 M2 discrimination and baseline
438 approach-frame opportunities:219 threat,158 uncertain,61 untracked (raw rule).
199 negative-frame opportunities:110 nonthreat,59 uncertain,30 untracked,0 threat.
Simultaneous red/green states at15.28–15.84s confirmed by actual rendered frame.
Growth-only baseline incorrectly warns on many negative frames. Baseline script
now applies the same size/quality gates; rerun final counts before reporting.
These are encounter-category proxies, not physical collision precision/recall.

## E05 Initial heldout M3–M5, raw v1
1314 approach opportunities:451 warnings,828 uncertain,30 untracked,**5 false
apparently-non-threatening frames** on M4. All3 approach events get >=.8s warning.
1064 negative opportunities:110 nonthreat,367 uncertain,587 untracked,0 warnings.
Those negatives comprise four aircraft events (not1064 independent encounters).
M4 size contraction at15.44–15.76s caused false recession evidence; actual boxes
were small/interior, so truncation does not explain this failure.

## E06 Temporal confirmation ablation (diagnosed data)
Confirmation0.24s warning /1.2s nonthreat; no hold through unknown/lost observations.
M4 replay: false nonthreat5→0, threat156→144; negative nonthreat54→7.
M2 replay: approach threat219→210, negative nonthreat110→80. No false-safe.
This explicitly trades decision coverage/latency for rejecting brief evidence.
Not an independently validated fix. Source snapshots preserve both versions.

## E07 Analytic and oracle isolation
1200 fixed-seed 2D pinhole trials (200 each: ideal, centroid noise, aspect changes,
low pixels, dropouts, unobservable pan). Many abstentions, no false-clear samples.
See reports/analytic_sweep.json for actual per-bin denominators; avoid a misleading
single accuracy score. Actual geometry point CPA is independently calculated.
Oracle centroid/identity with detector spans leaves detector misses and size errors;
M1 approach131 threat /300 uncertain /7 undetected; M2 219/139/80. Negative M2
92 nonthreat /45 uncertain /62 undetected. Offline range-derived approach timescale
median error2.00s(M1,n131) and6.95s(M2,n225), correlated frames. This reference is
**not time-to-impact or CPA truth**, and this oracle ablation is not online results.

## E08 Airborne qualitative clip
Local 2_Planes_Formation_[xfXn0j9gnLo].mp4, source45–65s, scaled1280x720,
29.97FPS preserved, stride3. RGB source provenance is local footage; no encounter
GT or flight geometry verified. Auto camera diagnostic:194 unreliable,5 unobserved,
1 usable fit still in cooldown;200 analyzed frames. All237 tracked-object frames
uncertain. Tests an honest failure/abstention mode, not successful airborne risk.

## E09 Confirmed policy on eight further sequences
M6,M7,M8,S1,S2,S3,S4,S5, fixed35s selection,438 sampled frames/sequence.
3504 approach opportunities:1144 threat /1623 uncertain /737 untracked;2746
detector centroid matches. All8 approach events had >=.8s warning;0 false-nonthreat
frames. Only two negative aircraft events visible in these chosen subsets:
394 opportunities,173 nonthreat /217 uncertain /4 untracked;0 warning frames.
This is encouraging proxy evidence on a small, stationary-camera distribution.
It is not eight verified imminent collisions or a comprehensive false-alert rate.
S6/S7 remain reserved. Results and per-sequence events in artifacts/heldout_confirmed.

## E10 Backend spot check
18 fixed development frames,14 FP32 ONNX CPU detections,14 TRT detections,all14
matched. Median center error0.140px, max individual coordinate error0.232px,
max confidence difference0.00196. Not a full detector accuracy equivalence study.

## E11 History window comparison (development only)
Windows4/6/8/12s, same cached detections and confirmation policy. Longer windows
improve M2 negative coverage80→95 frames, but M1 warnings118→113/96/63 because
scale model residual grows. Rejected as an unconditional default improvement.

## E12 Actual M4 pixel inspection disproves a tempting explanation
GT-centered QA crops only (never online input), actual neural proposals13–17s.
Both full-frame and tile predictions contract together from~20 to~14px. Thus,
switching tile views is **not** the main cause of the observed false recession.
Do not claim box fusion solves it. Actual visual appearance/exposure/shape and
detector span measurement are central limitations. Contact sheet saved.

## E13 Camera stress exposed a reference continuity bug
Known2D pan/zoom/roll applied to real M2 frames, cached boxes transformed identically.
Not a flight simulation or actual airborne validation. Original auto stabilizer
could skip invalid motion intervals then resume with an incomplete transform,
mixing coordinate references in target history. Added reference epochs and evidence
reset on loss/reacquisition; regression test checks preservation of ID with history
discarded. Re-running stress separately as v2; preserve original v1 reports.
Wrong fixed-camera declarations deliberately remain an unsupported assumption in
pan tests. Default auto does not silently assume that the camera is stationary.

## E14 Exact projective alternative
Joint inverse-span and coordinate/span fit,256 deterministic covariance sensitivity
samples. Analytical intercept,depth-plane time and focal-invariance tests pass.
Development M1,4s:118 warnings (same as log); M2,4s:210 warnings and72 nonthreat
(log210/80). No demonstrated accuracy win; alternative kept optional, no novelty or
probability-calibration claim. Longer windows remain mixed. Current19 tests pass.

## E15 Complete multi-aircraft source coverage
All available M1–M8,26,68425FPS timeline frames(1067.36s),5,340sampledframes atstride5.
M5 has one missing TIFF, transparently repeated to preserve source-ID timing. All
source manifests/hashes saved.13other-aircraft events visible in available files;
M4_1's label interval ends before available M4RGB and is not counted.
8/8approach events sustained warnings>=.8s;0false-clear approach frames.
Other events:0/13sustained warnings versus9/13growth-only baseline, with identical
quality/confirmation gates. There were2brief warning frames on labeled M6other;
do not claim zero false-warning frames. High abstention: positive441threat/1319
uncertain/368untracked of2128;negative175nonthreat/1839uncertain/1257untracked/
2threat of3273. Actual RGB only, labels are encounter category proxies, no trueCPA.
These full sequences overlap earlier short development/validation samples, so are
expanded coverage and failure analysis, not an independent new test split.

## E16 Wide-bearing falsification
216analytical cases acrosslog/projective, sixbearings0–85deg,threeinitialdepths,
sixplaneintercepts.16false-clear outcomes with true2D pointCPA<1.5targetspans and
plane miss>5. Perfect resolved observations, very wide analytical image plane.
These are mathematical counterexamples, not flight videos. Even exact projective
plane fitting cannot establish trueCPA without additional camera/geometry constraints.

## E17 Adaptive crop scheduling
V1nearestpreviouscrop reduced calls7→3.08 but M2negative nonthreat80→8, rejected.
V2two-nearby-crop memory10calls /all-crop discoveryevery5calls:4.56–4.69calls/devframe.
M2negative80nonthreat retained;M1negative detections30→27. Sixfurther lateMclips:
2628positive902threat/1653uncertain/73untracked;1458negative203nonthreat/647uncertain/
608untracked,0warnings. Accuracydefault staysdense pending a complete paired analysis.

## E18 LoRAT SOT isolation
Existing external source imported read-only; pretrained FP16ONNX locally rebuilt
forTRT11.4 in38.59s. Engineoldformat mismatch again. GPUOpenCV crop/normalize and
TensorRT pointer bindings actually run; dependencies/environment unchanged.
Fouroracle-selected model seeds,M1/M2approach/other,250frames/case. GT selects the
intended seed once, never supplied to SOT after initialization; therefore NOT
end-to-end detection evidence. M2other194/194visibleGTframes within20px,median3.22px,
p959.00px. M1other152/250within20px,p95159.8px (drift). RawSOTscore>=.5 is not
calibrated visibility. Measuredupload+SOT~6.4–9.7ms withpossibleGPUcompetition.

## E19 GT-free appearance bridge
Optionalbridge runsactualdetector every3processedframes, up to3LoRATtargets between
observations, independent detector corrections. Observations/source/rawYOLO outputs
are separate in traces. Initialclass-consistent suppression duplicated bird→airplane
switches on the same target; actualcache diagnosis confirmed overlap, fixed with
class-agnostic spatial/scale suppression. Regression test added; labels never imply
harmlessness. CorrectedM2approach231warning/192uncertain/15untracked (dense210/167/61),
negative83nonthreat/71uncertain/45untracked (dense80/89/30). M1negative135tracked
uncertainframes vs35dense, but approach113warnings vs118dense. Mixed results.
11further lateclips completed:4818approach1419threat/2567uncertain/832untracked,
all11events warned,0false-clear.1458negative187nonthreat/620uncertain/651untracked,
0warnings. Compare against paired dense inputs before selection. Throughput34.3FPS
on M2 without renderer, not full rendered demo throughput. External code hashes saved.

## E20 Independent scale measurements
Experimentalflag `--detector-scale-only`: SOT centroid continuity, only actual detector
span measurements for growth. Avoids treating tracking-box expansion as independent
loom evidence. Tests explicitly construct false tracking-box growth with constant
independent spans and confirm abstention. M2warning172,negative23vs231/83mixed-size
bridge; M1warning107vs113. Scientificconservatism reducescoverage; not default.
25tests currently pass. All old traces retained.

## E21 Warning-event sensitivity
Both complete-sequence variants warn on at least one frame of1/13knownother events.
Densemaxfalse pulse0.4s,bridge0.6s. Atpredefined>=.8s, both0/13; do not hide these
short pulses behind the persistence threshold. Positive8/8nominalevents remain
warned atdurations0–2s. Source reports/event_sensitivity.json includes all counts.
Offline annotation range at alert is diagnostic context, not an online range output.

## E22 Range annotations are not unquestionable event truth
M3 nominal approach range3562→4203,1071/2140increasing steps,globalfitpositive12.06
range-units/s. M2andM5alsohave late increasing ranges; range minima occur before
their final target label. This reinforces treating published approach labels as
nominal event categories, not per-frame imminent-collision truth. M3must be flagged
in any geometry-based subset. Units/alignment not independently calibrated.
Do not use M3 for time-to-impact or range-timescale accuracy claims. See range audit.

## E23 Actual helicopter qualitative footage
UAV-Anti-UAV_Test_000252 first14s,30FPS,856x480,RGB only, no GT supplied online.
Author repo identifies academic research use only; local manifest stores source hash.
Actual detector classes113helicopter/25airplane trackedframes.137uncertain/1nonthreat,
no warning. Background similarity fits135of140frames, but cannot establish actual
3Dcamera/target geometry. Inspected frame at7s: aligned helicopter, size-instability
reason, camera diagnostic visible, avoidance refusal. No collision accuracy score.

## E24 Appearance confidence is not calibrated visibility
M1other isolatedSOT: of140GT-visibleframes accepted atscore>=.5,77havecentroiderror
>20px. At>=.7,24/60wrong;>=.8acceptsonly4(allwithin20px). Thus merely raising the
threshold does not yield robust coverage. Independent detector correction and
threat-model abstention are necessary; no calibratedtracking probability claim.

## E25 Timestamp correctness and dependency protection
Installed only av==18.1.0 with --no-deps after a one-package dry run; custom CUDA
OpenCV path/version/device preserved in reports/environment_after_av.json. Default
export now preserves actual presentation timestamps with PyAV/libx264. The real-RGB
VFR trial has 70 frames with mixed 0.08/0.20s intervals: trace error <9e-16s, output
PTS error zero, duration 7.84s. Stream/container duration metadata was inconsistent;
independently decoded packet PTS+duration established the source presentation end.
No-render and rendered timelines now share the same source clock. See vfr_trial.json.

## E26 Correlated-error uncertainty sensitivity
A fixed-seed AR noise sweep tested actual fitted-slope coverage of the nominal
2.5-SE margin. IID:97.4%; rho=.8:57.8%; rho=.95:42.6%. Optional time-based Bartlett
HAC at 0.5s improves those correlated cases to72.8%/56.4%, still inadequate for
calibrated confidence. At1s rho=.95 coverage58.8%. This is an optional sensitivity
setting, not a default or a claim of probabilistic safety. See correlation_sweep.json.

## E27 Conditional bounded-error geometry
Exact set-membership linear slope intervals contain the true slope in500/500
trials at each of rho0/.8/.95 under explicitly clipped +/-2px errors. All available
plane-miss intervals contain truth in1800 controlled constant-shape trials; no
infeasible case by construction. This only checks the conditional math.
On real cached M1/M2 detections, assumed +/-2px and +/-10% span bounds reject353/438
and226/438 nominal-approach windows. M1:36warning/393uncertain/9untracked;
M2:74warning/303uncertain/61untracked. M2other:48nonthreat/121uncertain/30untracked,
0warnings. Both nominal approach events retain >=.8s warning, but substantially
less coverage than log geometry. No main-default change justified. Reports bounded_*.

## E28 Calibrated CPA alternative (in progress)
Optional --geometry calibrated requires explicit source-resolution intrinsics,
rectified square-pixel pinhole RGB and --camera fixed. It fits [X,Y,Z]/W via
[(u-cx)/s,(v-cy)/s,f/s] and estimates future CPA in constant projected span units.
No metric range or physical span is supplied or inferred. Covariance sensitivity
excludes calibration/shape bias. Exact wide-bearing counterexample unit test
recovers true CPA and does not falsely clear; broader sensitivity sweep pending.
41 unit tests pass. QUT has no verified intrinsics, so no calibrated real-video
accuracy claim and no invented QUT calibration file.


## E29 Calibrated geometry sensitivity
756 analytical conditions were evaluated (108 trajectories each under ideal,
focal x0.5/x2, constant span x0.5, smooth aspect growth/shrink, correlated noise).
Ideal max CPA reconstruction error3.33e-13 target spans;10/58 close cases warn,
48 abstain;13/23 far cases clear,10 abstain. No close-case false clearance in these
conditions, but aspect growth causes3 far-case warnings. This is conditional
analytical validation, not real calibrated encounter accuracy. Wide-bearing
sensitivity is large and uncertainty is deliberately broad. No main-default change.

## E30 Clean rendered runtime, RTX4060
Actual M2 1280x720,438 processed frames,stride2,TensorRT batch1,FP16-derived/mixed
engine,1800x1140 Russian render,libx264/PyAV PTS export. CPU preprocessing9.32FPS;
GPU preprocessing12.37FPS;adaptive GPU14.80FPS. Mean detector59.57/29.85/23.10ms;
render18.70/20.47/17.12ms;encode/write17.78/18.21/16.79ms;networkcalls7/7/4.69.
These are separate serial runs, not a universal speed guarantee; per-run p95 and
startup are retained in artifacts/runtime_final/*/benchmark.json. GPU dense began
before a source edit and its end-of-run hashes can reflect later disk contents;
this provenance defect was corrected by capturing source hashes at process start.
A final repeated benchmark after freeze remains required. Decode stage originally
counted only selected-frame decode; now accumulates stride-skipped decode too.

## E31 Match-tolerance sensitivity
Archived full dense M1–M8 traces at5/10/20/30/50px tolerance retain8/8 nominal
approach events and0/13 sustained-other warning events. Positive warning frames
423/438/441/442/445 of2128; other nonthreat135/167/175/175/175 of3273. Negative
untracked1586/1321/1257/1255/1255. Thus event categories are stable here, while
frame-level tracking coverage depends on approximate centroid matching. These are
archived legacy-policy traces, not fresh independent tests or physical CPA truth.

## E32 Higher-resolution coverage and source provenance
Native crop grid now covers every pixel with25% requested overlap at arbitrary
resolution, preserving exact1280x720 grid. Tests cover small/tall/4K rasters and
60px targets near crop boundaries. Native real formation1920x1080:17 neural calls
per frame,200processed frames,9.40FPS rendered,581uncertain track-frames. It is the
same45–65s source encounter as the previous1280x720 clip, not independent evidence.
Visual inspection found partial-target duplicates from internal crop boundaries;
optional --tile-edge-margin rejection is being evaluated, default remains0.
New runs record input SHA256. Replay checks digest when available and explicitly
marks legacy caches unverified; timestamp mismatch and invalid observations fail.
49 tests pass before latest rendering-only adjustment.


## E33 Crop-guard results and partial-box audit
Actual neural edge0/2/8px comparisons complete on M1/M2/native formation. At2px,
M1positive matched detection431unchanged, warnings117→127; M2detection358→360,
warnings209→213, negative78nonthreat unchanged. Native formation raw detections
436→289, trackIDs33→13, all decisions uncertain. At8px nearly identical but not a
justification to choose a larger rejection margin. Optional2px tested beyond
these development clips in the full paired campaign, still running. More warnings
alone is not an accuracy improvement. Unmatched new M1/M2 warnings occur near
large targets with20–105px centroid deviations; no verified false-alert count.

## E34 Unmatched-warning visual review
Archived full dense campaign has7unmatched track-warning episodes/35samples at
20px centroid matching. All7selected mid-episode source crops visually contain
an aircraft, often a partial-wing model box displaced from subjective GT. The
longest is2.2s; nearest-GT distances in previews reach117px. The review does not
establish trueCPA or expert labels, but demonstrates why unmatched!=false alert.
See reports/unmatched_warning_review.png and unmatched_warning_audit.json for exact selection.

## E35 Source-boundary geometry guard
A source image boundary truncates apparent span, independently of internal crop
boundaries. --image-edge-margin1 is currently opt-in; resets geometric history
and abstains while fresh boxes touch the source boundary. M2cache replay leaves
209positive warning frames unchanged and reduces other green78→73, replacing5
with explicit clipped uncertainty. Consequently the old15.5s red/green overlap is
not defensible as simultaneous reliable geometry. The new demo shows green+yellow
at14s and later red. Raw neural observations unchanged; replayFPS is not inference.
Full-source ablation and final default selection pending.54tests pass with actual
NVENC integration enabled. Intentional cache timestamp mismatch correctly marks
run failed, removes stale benchmark and closes resources without cleanup errors.


## E36 Complete-source paired crop validation
All15 local sources prepared:55,531actualTIFF/55,535timelineframes,2221.4s (37.02min).
Four missing source IDs repeat the previous frame (M5,S1,S2,S3). Local counts differ
from the paper's55,521 total; manifests/audit preserve the actual local endpoints,
counts and gaps rather than silently forcing the published total.
BothactualGPUdense runs processed11,113samples, stride5, original1280x720. Atedge0:
4031approach opportunities789warning/2073uncertain/1169untracked;3273other:
7warning/170nonthreat/1840uncertain/1256untracked. Atinternal-edge2:
838/2027/1166approach;otherunchanged. Both15/15nominalevents warned>=.8s,0positive
false-clears;1/13other eventswarns1.4s (M6). Unlike historicalCPUfull8M, GPUresize
extends the false-warning pulse; paired18frame preprocessing equivalence cannot
establish all-trajectory decision equivalence. Source category!=trueCPA (M3caveat).
No untouched split: full extensions overlap previously inspected clips.

## E37 Correlated-error and clipping counterfactuals
Fixed-camera source transforms are identity. Rebuilding all15source histories from
actual cached observations/camera validity exactly reproduces states/reasons and
numerical geometry (maxdifference0). NoRGB/NNruntime claim for this isolation.
Baseline edge0:789positive warning/170othergreen/7otherwarning. Sourceclip1:
764/159/7;plusHAC0.5s:752/151/2;plusHAC1s:751/151/0. All15nominalevents retain
>=.8swarnings,0positivefalse-clear. The last2other warningframes atHAC0.5 form0.4s.
Raisingminspan10doesnot removeM6'spulse. Extendingwindow6gives801positivewarnings,
359othergreen but retains7otherwarnings. No calibration/physical safety is claimed.
Withinternalcropedge2+sourceedge1+HAC1s, the4s-window candidate has801positive
warnings/2064uncertain/1166untracked and151othergreen/1866uncertain/1256untracked,
zerootherwarningframes. With6swindow:830warnings/330green but3otherwarningframes.
Keep4s as the stronger conservative candidate. Selection is post-hoc development,
not unbiased generalization evidence. Reports geometry_campaign/profile_comparison.

## E38 DINO-conditioned pixel extent refinement rejected
Cached neural sky masks conditioned local background eligibility; actual RGB
contrast components refined YOLO-seeded boxes, noGT read. M1:425componentsaccepted,
mean7.57ms refinement/sample. M1positive warnings117→92,untracked9→7;S2warning
129→75,untracked251→263. Components often describe a body/wing fragment rather
than a stable physical span. No main integration or gain claimed. Raw neural
observations preserved separately. See scripts.silhouette_trial and actual caches.

## E39 Detector confidence is not a safe/unsafe threshold
Actualneural low-conf0.03 runs on diagnosedM1/M2/S2 atfixedgeometryHAC1s,clip1,crop2;
post-NMS thresholds0.03/.08/.12/.18/.25/.4/.6 then isolate tracking/geometry while
holding source camera validity. At0.03 vs0.18,approach untracked198vs320 of1314,
other345vs433 of637;warnings422vs445,othergreen81vs73. At0.4other592/637untracked,
45uncertain,0green. No other warning at anythreshold in these3clips. Unmatched
warning13samples unchanged; these are not verified false alerts. Lowerthreshold
has more nuisance tracks (M2freshmean2.39,max6 at.03 vs1.64,max4 at.18). S2at10s
stillhasno fresh track at0.03: lowering confidence does not solve continuous tiny
object recall. Full-source0.03 actualNN+threshold counterfactual campaign running.

## E40 Advisory consistency and ordered export
New advisory compares a common image flow of1/64rasterwidth/s, corresponding to
parallel lateral translation only under equal camera-depth assumptions. The score
is minimum change in drift/observedspan across quality-eligible targets. It excludes
invalid aspect/tiny/lost/clipped geometry, explicitly counts unresolved targets,
remains invariant under input resizing and always refuses physical recommendations.
Three new tests cover multi-target worsening, resizing and bad-extent exclusion.
Optional --async-render has one ordered worker and atmost2pending frames, explicit
error propagation and metadata snapshots. VFRpts/duration/order and workerfailure
tests pass. Actual pipeline smoke/performance and final default selection pending.
59tests pass with real NVENC enabled.


## E41 Full confidence campaign and default selection
Actual low-threshold0.03 TensorRT inference covered all15complete sources and11113
samples. Post-NMS filtering at.18 exactly equals the separately executed.18boxes
and camera-validity flags on every sample (0differences). Conservative.03:
approach802warning/2300uncertain/929untracked of4031;other156pass/2345uncertain/
772untracked of3273,0warning. All15nominalapproach events warned,0approachpasses.
Compared with the same guards at.18,237more approach and484moreother tracked
samples; gains persist at5/10px matching. Unmatcheduncertain3722vs1026, more
nuisance/possiblyunlabeled observations; no exhaustive precision or physicalCPA.
Default profile now.03+denseCUDA+cropedge2+sourceedge1+HAC1s/window4. Reference
profile preserves earlier settings. This is development selection on reused data.
Growth-only samequality/policy baseline warns10/13other events,476othersamples.

## E42 Ordered render thread, actual inference equivalence
Actualselected M2,438samples,stride2,NVENC:serial14.8436FPS vs thread21.3467.
Startup/preflight-inclusive13.8295vs19.2356 (timer insidepipeline,notPythonstartup).
Detection and decision files are byte-identical; all438decoded outputframes are
pixel-identical,0PTSdifference,both35.000s,H264yuv420p. Queue bounded2. Source
artifacts/async_trial and reports/async_equivalence.json. Not cache replayFPS.

## E43 Render process and RGB encoder-input hypothesis
ActualsameM2 with separate renderprocess:24.0883processingFPS,21.1179including
pipelinepreflight/startup. Childstartup0.393s; samecustomOpenCV5path,12CVthreads.
All438decodedframes,decisions,neuraldetections andPTS matchserialexactly. See
process_equivalence.json. No physical streaminglatency claim. Controlled worker
failure/VFR/cleanup tests pass. Optional mode, not a silent codec fallback.
Isolated240frameNVENC probe: CPUYUVinput106.04FPS vsBGR0/GPUconversion105.51FPS,
write9.15vs9.32ms. BGR0 yields no speed benefit, between-modeRGBPSNR46.60dB,
RMSE1.19,maxchanneldifference58. Keep YUVinput; no CLIpromotion. Codec microbenchmark
uses20repeatedpreviouslyannotated frames, not application or encounterFPS.

## E44 Parallax/ownship-translation falsification and scope restriction
324analytical trajectories per method, all trueCPA0;297 meet configuredsmall
similarity-motion bounds assuming sufficient matched planar features. ExactBG
stabilization yields96log/HAC1,78projective,6bounded apparent passes amongthose297.
These are adversarial constructed projections, not error prevalence or realRGBdata.
ActualKLT/RANSAC on translated2Dtexture plus analyticaltargetboxes:62/63fitsvalid,
22confirmed apparent-pass samples while raw targetwouldwarn. New camera-scope
restriction returns63uncertain. It withholds ALL auto-mode passes, so this is
abstention, not solvingparallax. Fixedacquisitionassumptions remainexplicit; potential
warnings stillconditional. The research override/reference profile restoresoldscope.
68tests pass with actualNVENC enabled. Rendercircles nowuseoriginal-to-display
scale andexplicit4–50pixel cosmeticlimits, withsourceviewportclipping.


## E45 Optional methods under the selected conservative profile
Actual NN comparison: M1/M2/S2last35s plusFULLM4, allstride2,1997approach and1872other
opportunities. Dense:555warning/1090uncertain/352untracked approach,138pass/1367unc/
367untracked other. Adaptive:555/1066/376,other138/1360/374. Appearance bridge:
569/1083/345,other175/1468/229. Independent-detector-span bridge:559/1093/345,
other48/1595/229. No approachpass or knownotherwarning in anyofthese4variants.
No-render actualper-sourceFPS:dense29.5–31.1,adaptive36.3–44.4,appearance41.1–46.2,
independent42.0–46.9. Cadence and model calls differ; this is actual inference,
not rendered throughput or independent unseen validation. SOTscore remains
uncalibrated; historical drift counterexamples still apply. Dense remains default
until broader independent evidence; optionalbridge can improve speed/coverage.
Full-framerate regression on studied M1/M2/M4/M6/S2 is now running to test default
stride1 rather than assuming strided results generalize to denser observations.

## E46 Source/mask identity and warning timing
The optional sky workflow now verifies CFR timestamps before expensive inference,
rejects source/mask frame-count mismatches, and binds generated masks to source and
checkpoint SHA-256 hashes. Legacy masks remain explicitly unverified. Failed
workflow reruns remove stale success timing. Targeted real-codec tests pass.
Warning episodes are independently reconstructed with annotation-gap resets and
explicit sample-hold support. All15 nominal approach events in the selected
stride5 campaign have a >=2s episode,14have>=4s; none of13knownother events warns.
First qualified episode onset is3.4–22.2s before the last annotated sample. This is
retrospective episode onset, not decision latency or ground-truth impact lead time.
Source-provided range at onset is offline context only; M3 remains contradictory.
The full-rate cadence experiment and matched sky-region control are running.

## E47 Full-rate regression exposes a residual other-aircraft warning
All19370 source frames on M1/M2/M4/M6/S2 were actually processed at stride1 with
current conservative settings. Approach:7010opportunities =1487warning +3607uncertain
+1916untracked,0passes. Other:6098 =26warning +434pass +4776uncertain +862untracked.
All5nominal approach events warn. One of5other events warns: M6 at96.16–97.16s,
26samples/1.04s. All26match the approximate other-aircraft point within20px.
This contradicts extending stride5's zero-other-warning result to default stride1.
The source crop and fit show a growing detector span with low drift; label is an
encounter category, not verified physical CPA. See reports/M6_fullrate_other_warning.*.
Counterfactuals on exactly the same actual M6 observations: projective increases
other warnings26→42 (approach303→325); window6 gives33 (approach318). Bounded yields0
other warnings but approach72, with597other/807approach infeasible windows. Residual
cutoff.1 gives0other warnings and208approach, by more abstention. None promoted from
this isolated failure. All15source residual sensitivity is queued. Default retains
its explicit limitations; M6 recorded-decision failure excerpt is being produced.

## E48 Sky causality, isolated runtime and mask-free zoom
With current low-confidence dense inference on S2, known approach opportunities
are 136 warning / 124 uncertain / 178 untracked out of 438. The actual DINO-mask
proposal path gives 147 / 242 / 49. The all-pixel proposal-region control has
identical detections.jsonl and decisions.jsonl SHA-256 hashes. No independent sky
contribution is demonstrated on this scene. The full two-pass workflow takes
48.7886 s (8.9775 processed FPS), assessment alone 15.9492 FPS. Source and checkpoint
hashes, CFR preflight and both process startups are included in the full timer.
The first rerun failed with cuDNN sublibrary mismatch: importing custom cv2 for
preflight prepended OpenCV/CUDA13.3/cuDNN9.24 directories to inherited PATH.
Capturing the incoming environment before importing cv2 and passing it explicitly
to the isolated SkySegDINO child fixes the actual rerun. Neither external environment
was modified. Failed status/log retained at reports/sky_failed_dll.log.
Optional --zoom-proposals now exposes the all-pixel proposal/YOLO path without
segmentation. It is not a classical sky detector; every added observation requires
the supplied neural detector. Crop-boundary filtering is shared with native tiles.
Its current actual video and equality check are queued after the bridge extension.

## E49 Lower confidence and stricter quality are tradeoffs
Actual M1/M2/S2 inference at 0.005 followed by post-NMS threshold counterfactuals
0.005/0.01/0.03/0.18 gives approach untracked counts 110/124/198/320 of 1314;
warning counts 421/448/422/445. Other untracked:277/312/345/433 of637. At0.005
there are no other conditional passes, due to increased uncertainty/association.
Unmatched uncertain track samples:2703/1529/771/297; these are not verified false
positives. Defaults remain0.03 because lower thresholds lack full-source validation.
Residual cutoffs0.08/0.10/0.12/0.16 on all15sources retain all15nominal warned events
but give653/685/729/802approach warning samples and124/135/151/156other pass samples.
No physical safety calibration or untouched validation follows from this sweep.

## E50 Existing synthetic detection and aspect diagnostic
317 paired RGB/metadata images from predetermined AIR2LAND variants000/012/024/035,
including all available pairs and explicitly listing1missing color image. Full-frame
inference matches310boxes atIoU>=0.5 and emits314boxes; native crops match312and
emit388. All matching class predictions are airplane. Visual audit includes heavy
obscuration and very poor visibility; GT boxes may describe occluded content.
No verified source FPS/intrinsics/native-pose units, so no temporal threat or metric
CPA scores. Existing sibling SciPy exports metadata; main environment unchanged.
First-to-last range norms change only about0.8–2.3%, while GT span falls from147/148
to101px. Other intermediate variation is larger (range max/min about1.17–1.18).
This is evidence against constant projected-span assumptions in these existing
rendered data, not a real-flight accuracy result. No simulator was operated.

## E51 Matched full-source appearance extension
Actual full M6/S1/S3/S5 at stride2, four approach events and one other event.
Dense approach514warning/1136uncertain/924untracked of2574; other12warning/12pass/
441uncertain/30untracked of495. Mixed-size appearance442/1142/990, other13/0/446/36.
Independent-detector-size appearance450/1134/990, other0/5/454/36. All4nominal
approaches warn in every method; no approach pass. No-render FPS ranges dense
29.69–31.41, mixed35.91–52.48, independent38.42–52.42. Independent sizes remove
M6 other warning at this cadence, but overall approach coverage worsens. Dense
remains default. These are studied sources and category proxies, not collision GT.

## E52 Small-raster duplicate inference removed
A derived real M2 clip at640x360 exposes an identical full-image/native-crop pass.
The native tile coverage function is unchanged; Detector skips the redundant pass
only when both dimensions already fit704x416. Actual438-frame before/after runs
have identical detection and decision SHA256 hashes. Model calls2→1 per frame;
no-render processing73.8768→99.3421FPS, pipeline-startupinclusive54.8941→68.0041FPS.
Single paired run, same resized encounter, no annotated-FPS or generalization claim.
This branch is unused by the primary1280RGB and native1920RGB results.

## E53 Final video and environment checks
75tests pass including realNVENC. All8final videos decode fully,3000frames total;
allPTS/framecounts/durations andH264/yuv420p1800x1140 verified. Maximum timestamp
error7.1e-15s;3source-to-render samples/video maxlossyRMSE4.06/255. This checks
export consistency, not threat truth. Actual gallery playback advances to19s/35s
in the in-app browser over loopbackHTTP; responsive layout and poster verified.
The app blocks file-URL navigation; no bypass attempted. Temporary QA tab/server
closed. Local MP4/HTML/Markdown artifacts remain the delivery outputs.
Customcv2version/path/fullbuild/devicecount exactlymatch initial fingerprint;
actualCUDAupload/download exact. No dependency changes after av18.1.0.

## E54 Fresh-process end-to-end timing
9actual rendered runs,3each M2/S2_zoom/nativeformation. Parent timer includes child
interpreter/imports, model startup, processing/render/export/flush and process exit.
Every detection+decision trace matches its final gallery reference exactly.
Median external FPS:M2=19.8845(range19.392–20.389);S2zoom=14.4324(14.410–14.604);
nativeformation=10.9183(10.917–10.937). Sequential freshprocesses on warm machine;
OS/GPU/file caches uncontrolled, no cold-boot or streaming-latency claim. Full-stage
metrics and exact commands in reports/external_benchmark.json.

## E55 All-source default-cadence extension (running)
M6's cadence-specific category warning justifies processing all15complete sources
at stride1. The5matching complete runs are reused after checking configuration,
inputSHA, framecount and completion;10remaining sources run actual neural inference.
No algorithm tuning. scripts.complete_cadence writes reports/cadence_all_sources.json.
A queued offline GT-centroid/identity + independent actual-span ablation separates
association/centroid limits from span/model limits; it is NOTcamera-only performance.

## E56 Completed default-cadence validation and final correctness repair
All 55,535 timeline frames / 15 sources completed actual neural processing.
Before clipped-history exclusion: approach4,221warning/11,876uncertain/4,058untracked
of20,155; other40warning/1,200pass/12,294uncertain/2,827untracked of16,361.
Excluding the invalid clipped seed changes only one matched S1 warning to uncertainty.
No approach passes. All15nominal approaches warn; other warning frames occur inM6
(26/1.04s) andM7(14/0.56s). Do not hideM7 behind the0.8s event threshold.
340 unmatched warning track-samples are not verified false alarms: labels incomplete.
Offline GT-centroid/identity ablation gives4,516approach warnings/15,639uncertain,
31other warnings/1,291pass/15,039uncertain, with zero untracked by construction.
Actual detector sizes exist on15,519approach and12,492other opportunities; no invented
sizes, range or class risk prior is passed to geometry. This is not camera-only performance.
All canonical videos refreshed after the narrow fix; current M2 actual neural output
exactly matches repaired-code replay on438frames. Earlier benchmark references archived.
76tests pass, all8videos/3,000frames decode/PTS/codec/duration checked again. Final
CUDAOpenCV fingerprint and real CUDA round-trip remain identical to initial build.
Final measured usage34% at02:23:13UTC, baseline0, cap35; integer/account-wide.
