# Active mission — resume here

Start: 2026-10-18 01:11 Europe/Istanbul (17Oct22:11UTC).
Latest checkpoint: 18Oct03:34 (18Oct00:34UTC). Persistent Codex goal ACTIVE.
Weekly quota baseline0%, latest actual account reading18%, additional18 percentage
points of35authorized. Account-wide integer reporting; no token/time substitution.
Do NOT stop at minimum acceptance. Continue productive experiments toward budget.
No subagents requested/used. Workspace notGit. Missing10H spec reference resolved
by supplied task/AIR_COLLISION_AVOIDANCE_5H_AGENT_SPEC.md. No AGENTS found.


## Latest delta — read before older notes (02:56 local)
- Actual weekly quota14% used, baseline0; goal ACTIVE, continue to35 or no defensible
  work. No subagents used. Code is runnable; do not stop at minimum acceptance.
- Optional calibrated CPA geometry requires real source-resolution pinhole
  calibration JSON, square pixels/rectified RGB and fixed camera. No QUT focal
  length guessed. 756 analytical cases, ideal CPA reconstruction error<3.4e-13
  spans,10/58 close warnings/48uncertain,13/23far clear/10uncertain. No false clear
  in this limited sweep; aspect-growth causes3far warnings. NOT real flight validation.
- Native tiling now covers arbitrary rasters (1280 unchanged7calls;1920x1080=17;
  3840x2160=50). Tests cover all pixels and60px objects across boundaries.
- Crop-edge guard experiment0/2/8px, real neural runs: M1raw640→583,IDs17→12,
  approach117→127warnings, matched detection431 unchanged; M2raw746→727,
  IDs20→17, approach209→213warnings, detected358→360, other78green unchanged.
  Formation native1920raw436→289, IDs33→13, alluncertain. Fewer boxes alone is
  NOT precision proof; guard remains opt-in --tile-edge-margin2, default0.
- Image-edge geometry guard is a separate NEW experiment --image-edge-margin1,
  currently opt-in default0 so running paired campaign remains unchanged. Clears
  span history on physical image-boundary boxes. M2positive209warnings unchanged,
  other78→73green; old brief red/green overlap involved clipped target and disappears.
  New honest demo should show green+uncertain earlier and later red; do not advertise
  old border red/green as validated simultaneous geometry. Select default after
  full-cache ablation; main full paired campaign has guard0 in all runs.
- Renderer: short nonoverlapping labels,2s histories, stale boxes hidden on source,
  stale zoom uses placeholder instead of unrelated current pixels. A misplaced
  stale label found in visual QA was fixed. Latest checked reports/new_ui_checked.png.
- Added explicit source SHA256 and cache identity/timestamp/box validation. Legacy
  caches marked unverified. RunSession writes running/failed/complete status and
  removes stale benchmark before rerun, cleans decoder/encoder on exceptions.
  Intentional timestamp mismatch fails with no stale benchmark or cleanup errors.
- NVENC optional --encoder h264_nvenc actually works, VFR PTS/duration test passes.
  M2rendered GPU dense14.85FPS withNVENC vs12.37libx264; settings are QP20 vsCRF20,
  not quality-matched. CPU-preprocessing libx2649.32,adaptiveGPUlibx26414.80FPS.
  Final repeated clean performance benchmark still pending after freeze. New reports
  include whole_run_wall_s, input hash time; processingFPS excludes preflight/startup.
- 54 tests pass WITH AIRAVOID_TEST_NVENC=1. Default portable suite skips NVENC test.
- Match tolerance5/10/20/30/50px: archived fullM1–M8 event8/8nominal positives and
  0/13sustainedother remain. Frame counts/coverage vary substantially. No CPA truth.
- Unmatched-warning audit:7episodes/35samples in archived full dense traces;
  inspected reports/unmatched_warning_review.png shows actual aircraft/partial-wing
  boxes in all selected7 examples, often20–117px from subjective/interpolated GT.
  Thus unmatched warnings are NOT verified false alarms. No expert GT claim.
- Vigil publicdata manually gated (API verified), no access bypass/request. MMAUD
  OneDrive zip+GT return403. Old commented NTUDOI actually NTUVIRAL, not MMAUD;
  retained metadata only, no files downloaded. GAREUD real2D labels not CPA truth.

## Running jobs at this checkpoint
- Session3913: full S1–S7 TIFF preparation to data/qut_full. S1/S2 manifest ready,
  remaining sources encoding. Explicit split=extension_of_previously_inspected_sequences.
- Session88332: scripts.full_paired_campaign, actual GPU dense edge0 vs2 on all
  15 complete sources, stride5. M1–M6 both done, M7 running. Waits for S manifests.
  Outputs artifacts/full_gpu_default and full_gpu_edge2, full_summary.json per root.
  No fresh untouched split; M3 range caveat. Runtime competes with preparation,
  so do not use these as the clean performance benchmark.
- Session51467 clipping M2 no-render completed (poll if needed).
- New demo/M2_clipping_guard render running (session from latest tool): raw cache
  SHA verified, clipping guard1, current UI, NVENC. Cache replayFPS not inference.

## Next high-value work
1. Finish full paired campaign; analyze per-target coverage, false clears, brief
   warning pulses before selecting crop guard. Full-cache image-edge ablation.
2. Update demo guide NOW: old M2 red/green overlap is a clipped-boundary limitation.
   Inspect new video14s (green+uncertain),18s(red), refresh canonical gallery later.
3. Confidence threshold sweep on actual low-conf neural caches (planned, notrun).
4. Pixel silhouette size-refinement with cached DINO sky and YOLO seeds: investigate
   partial-wing boxes/unstable size, if useful; no classical-only sky detector.
5. Improve advisory comparison to a common pixel flow under explicit equal-range
   assumption; current per-target1span/s cannot correspond to one known maneuver.
   It always abstains, but explain this physical inconsistency; no recommendation.
6. Final-source benchmark, VFR/full-code smoke, docs/research evidence, environment
   fingerprint and video QA. Freeze risky work near30additional quota points.

## Latest delta — 03:34 local (18Oct00:34 UTC)
Actual account reading18% at00:22UTC, baseline0. Goal ACTIVE. Continue productively
toward35, freeze risky changes near30. No agents, no new packages after av18.1.

All preparation/full paired/geometry/confidence jobs HAVE COMPLETED. All15local
sources:55,531actualTIFF,55,535timelineframes (4 missing IDs repeated),2221.4s.
The paper says55,521; actual local endpoints/counts recorded without silent repair.
- Actual GPU reference15full:11,113sampled frames atstride5. Nominal approach4031:
  789warn/2073uncertain/1169untracked;other3273:7warn/170green/1840uncertain/1256untracked.
  15/15nominaleventswarn>=.8s,1/13other eventwarn1.4s(M6),0approachgreen.
- Crop2+clip1+HAC1,threshold.18:801warn/2064uncertain/1166untracked positive;
  other151green/1866uncertain/1256untracked,0warnings. Allnominaleventsretainwarnings.
- FULL actual low-conf.03 campaign nowcomplete:15sources,11113actualNNsamples.
  Matchedpositive3097detections,802warnings/2300uncertain/929untracked of4031.
  Other2497detections,156green/2345uncertain/772untracked of3273,0warnings.
  15/15nominaleventswarn>=.8s,0positivegreen,0/13knownotherwarnings. Unmatchedwarning
  37samples is NOT falsealert count. Unmatcheduncertain3722vs1026at.18; more nuisance
  observations. Freshtracks mean.84vs.53,p95both2,max5vs3. More coverage mostlyuncertain.
- Match-sensitivitygain persists at5/10px, notonly20:positive detected2964vs2752 at5;
  other2106vs1697. All15nominaleventcoverage stable. Data reused/development,NOT
  unbiased test or physical collision recall. M3rangeconflict remains!
- Higher-threshold filtering EXACTLY reproduces actual.18raw detections and camera
  validity on ALL11113samples (0diff frames acrossall15). Lowest.03reassessment
  source verification alsoexactlymatchesstates/numericalgeometry. See confidence_full.
- Conservative profile is NOW CLI DEFAULT:conf.03,tiledTrue,nativeGPUpreproc auto
  forTRT+CUDA,internalcropedge2,imageedge1,HAC1forlog,window4,confirmedpolicy,
  no contractionclearance. --profile reference restoresconf.18,no crop/defaultCPU,
  zeroedgeguards/HAC. Explicitflags override. GenericConfig/ThreatEstimator defaults
  stay reference for analytical scripts. Other geometry usesHAC0.
- Pixel silhouette refinement rejected: M1warning117→92,S2 129→75 andS2untracked
  251→263; DINOcontext+RGBcomponents notstablephysicalspan. No main integration.
- New common-flow advisory, input-width/64 px/s, equal-depth hypothesis; excludesbad
  extent/geometry,countsunknown,alwaysrefusesphysicaldirection. Resize invariance
  andmulti-targetworsening tested. Old source snapshot source_before_common_flow.
- Optional --async-render usesoneorderedworker,atmost2pendingframes; copyouter
  metadata before main pops histories, errorpropagation, cleanup/status. Actual
  cached120frameNVENCsmoke complete; performance is cache replay notNN.
- Source duration nowrelativefirstdecodedPTS (handles streamoffsets); nonzero5s
  offset integrationtestpasses. Fixed origin_ms unit bug (origin storesseconds).
- Renderer newbounds row, shortnonoverlappinglabels,2s histories,staleboxhidden,
  stalecropplaceholder,class/scoreerrorscaption,green'ПРИЗНАКИ ПРОХОДА МИМО'.
- 63tests pass WITH AIRAVOID_TEST_NVENC=1 (portable default skipshardwaretest).
- New serial ACTUAL conservative M2,438frames,stride2,NVENC:14.8436processingFPS,
  13.8295whole-runFPS. Next asyncactualbenchmark running, session54178,
  artifacts/async_trial/async, reports/async_trial_async.log. Only thisGPUjob active.
  Serial source artifacts/async_trial/serial. Comparetraces anddecodedframe hashes
  before claiming asyncbenefit. No heavyCPU/GPUjobs duringcleanbench.

Next:
1. Finishactualasync vsserial, decision/detection/videoPTS/frameidentity checks.
2. Run scripts/plot_full_extension.py using sibling anti_collision_system env AFTER
   active benchmark (matplotlib notinstalledmain); inspect resultingRussianPNG.
3. Clean README/RESULTS/evidencegallery for selectedprofile. AlreadynewM2_clipping_guard
   video showsgreen+yellow14s,laterred, butoldcovariance/threshold/advisory version;
   final canonicalcurrentprofileliveNN rerenders stillrequired.
4. ExploreNVENC BGR0 input and/or renderingprofile only ifactualbenefit; no falseFPS.
5. Newdefaultqualitative FPV/helicopter,actualtwo-passSky comparison vscheaplowconf,
   detector/geometry generalization limits,environmentfingerprint,finalvideoQA.
6. Profile integration: batch_evaluate now compares existing effectiveargs and hashes
   to prevent accidental stale reuse; historical full scripts explicitlyreference.
   Primary CLIprovenance nowcachehash+upstreamsummary; noNeuralreplayFPSdeception.

## Latest delta — 03:47 local / 00:47 UTC
- Conservative profile defaults are implemented and parse-tested. 65tests pass
  with AIRAVOID_TEST_NVENC=1. Generic Config default remainsreference; CLI resolves
  selectedconf.03/dense/GPUauto/cropedge2/imageedge1/HAC1s. Batch nowvalidates
  existing effectiveargs/inputhash to prevent stale reuse; per-run actualargsreported.
- ACTUAL live NN M2 serialvsrenderthread complete:14.8436vs21.3467processingFPS;
  startup/preflightinclusive13.8295vs19.2356. Exactdecision/detectionfileSHA equality,
  all438decoded outputframes pixel-identical,0PTSdifference,35.000sduration,
  H264yuv420p. reports/async_equivalence.json. Final currentgallerycopy:
  demo/M2_conservative/annotated.mp4 (source actual artifacts/async_trial/async),
  copy_manifest recordshashes. Previewdifferent_states.png at14s checked.
- RESULTS.md nowreadableEnglish evidence summary. New Russian figure
  reports/results_full_extension.png checked (minor3%labelthreshold changed after
  render; rerun plotlater). Sourcescounts55,531actual/55,535timeline/37.02min.
- Growth-only baseline on selectedsameactualtracks/quality/policy warns10/13known
  other events,476other samples. Geometric conservative has0knownother warnings;
  no physicalCPA/truecollisionclaim. Fullconfidence.03 gains persist5/10px matching.
- New optional --render-process implemented with bounded2framequeue, readiness/
  orderedacknowledgments,30s watchdog,childcleanup,workerCV2fingerprint. Testsverify
  actualVFRPTS/duration,correctvenvOpenCVpath,workererrorandexit. Not selecteddefault.
- Currentlyrunning actual full M2process-render benchmark (session from latesttool),
  artifacts/async_trial/process, reports/async_trial_process.log. NootherGPUjobs.
  Afterfinish compareagainstserial/threadfile hashes+decodedframes beforeclaims.
  Next: evaluatebenefit,maybeNVENCBGR0inputexperiment,clean3repeatbench,finalcurrent
  M1/S2/FPV/helicopter demos,updatedSkyworkflow,docsandQA. Datasetexperimentsall
  completed; do not launch duplicates blindly. Actualquota last18 at00:22UTC,
  baseline0, target35; freeze near30. No new packages, noagents, goalACTIVE.

## Latest delta — 04:08 local / 01:08 UTC
- Actualweeklyquota22 at00:58:28UTC, baseline0, target35; activegoal. Freeze risky
  work around30. Readnewestdeltas first; olderblocks arehistorical. Noagents, no new
  packages afterav18.1, customcv2still5.0externalpath/CUDA1.
- Separate renderprocess ACTUALM2:24.0883processingFPS,21.1179preflight/startup
  inclusive (insidepipeline timer). All438decodedpixels,det/decisionfileSHA andPTS
  EXACTLYmatchserial,35s,H264420. reports/process_equivalence.json. Thread21.35,
  serial14.84. WorkerCV2pathmatchesparent,12threads,startup.393s. Optionalprocessmode.
- Importantnewgeometryscope: ideal backgroundsimilarity canremoveownshiptranslation
  andcreatefalsepassesontruecollision EVENatimagecenter. 324analyticcases/method,
  297meet small-motionbounds; falsepasses96logHAC1/78projective/6bounded among297.
  No prevalenceclaim,noRGB/flight simulation. ActualKLT/RANSAC2Dtextureprobe:
  62/63validfits,22confirmedpasssampleswhiletrueanalyticalCPA0/rawwouldwarn.
  NewdefaultwithholdsALLpassesaftercameraauto(backgroundR/Tunresolved),potential
  warningsstillconditional. --allow-similarity-pass/referenceprofile restoresold
  researchassumption. Fieldsimage_rule_candidate_state/camera_motion_assumption;
  cameravalidityscopeexplicit. FixedQUTdecisionmath unchanged, notrerunallcorpus.
- RGBNVENCinputprobe REJECTED: isolatedYUV106.04FPSvsBGR0105.51,9.15vs9.32ms,
  PSNR46.60dB betweenoutputs. No CLIpromotion, constructorarg forprototype only.
- Renderercoordinatesnowcliptoactualsourceviewport, residualcircleproperlyscaled
  native-to-display andexplicit4–50pxdisplaylimits. FinalvideosneedcurrentUIrerender.
- 68tests pass withactualNVENCenabled beforelatest smallUIchange. No moremajor
  refactorsneeded. CurrentvideoM2_conservative remainsactualoldrenderthreadgallery
  copy; currentfixeddecisionprofilematches,butnewscope/clipdrawingUIneedsfinalrerender.
- RUNNING: scripts.selected_methods,actual4methods onM1/M2/S2last35s +FULLM4,
  allstride2/newconservativeprofile. Dense/adaptive/appearance/independentdetectorsize.
  M4fullpreservesknownfailurecontext. Outputsartifacts/selected_methods,
  reports/selected_methods.json/log. No otherknownGPUjob. PYTHONDONTWRITEBYTECODE1
  setforread-onlyexternalFastVisionimports. WaitbeforeotherGPU/performancejobs.

Nexthighvalue:
1. Readselectedmethodcomparison,don'tpromoteSOTwithoutcoverage/falseclear evidence.
2. RefreshactualselectedprofileFPV/helicopter/M1/S2/M2canonicalvideos, inspectwhy/
   alignment/currentcamera-scope, properPTS/codec. No GTonline. UseprocessNVENCfastpath.
3. ActualSkyworkflowvscheaplowconf, mask-vs-allsky proposal contribution ifvaluable;
   skysemanticmask currentlyunproven benefit inallskyQUT. NeverclaimcachedNNFPS.
4. Threecleanrepeatedfinalbenchaftercorefreeze;strides/resolutions/stagepercentiles/
   startup,externalprocesswall ifneeded;stagesoverlapnotadditive.
5. RewriteREADME/RESULTS/demo guide readable selectedprofile, addnewauto-scope
   limitation, updateprocess24FPS,baseline10/13othergrowthwarnings, confidence
   intervalnotcalibrated. FinalfigurePNG exists/checked; rerun min3%label change.
6. Finalenvironmentfingerprint,artifactQA/tests/bytecode/sourcehashes,metadataindex,
   orphan/zero-ownedartifactcleanup. Stop at35additionalpoints or truly no further
   defensibleexperiments; don'tstopatminimum. Keepactualquotareadings.

## Environment invariant
Use `.venv/Scripts/python.exe`, no `uv sync` / OpenCV wheels. Custom cv2 5.0.0 at
C:/Data/DEV/Libs/opencv_cuda_5/install/python/cv2/__init__.py,CUDAdevice1 preserved.
RTX4060 8GB, driver610.88, CPU i5-12400F6core/12thread. Torch2.12+cu130,TRT11.4.
Only av==18.1.0 installed, --no-deps after a one-package dry run; lock updated
with only av added. `reports/environment_after_av.json` confirms custom cv2 intact.
`reports/environment_initial.json` full fingerprint. Mainenv lacks scipy,matplotlib,transformers,einops,yaml,timm. Existing sibling
anti_collision_system env used forMatplotlib/SciPy read-only; SkySegDINO ownenv
used forsegmentation. Never overwrite either environment or external source files.

## Working code / entry
`python -m airavoid VIDEO --camera auto --output demo/run`
`python main.py` sameCLI. Default local TRTengine whenpresent. Supplied engine
incompatible243vs245; locally rebuilt YOLOin110.09s,LoRATin38.59s,artifacts/*.engine.
ORTCUDA failedcuDNN mismatch;ONNXCPU used onlyconversion comparison.

Implemented: five-class suppliedYOLO26m,dense/adaptive crops,conservativeMOT,
KLT/RANSACcamera similarity withreference epochs/history resets,drift/growth risk,
uncertainty +temporal confirmation,explicit avoidance refusal,RussianH.264videos,
actualJSONL observations/decisions/stage benchmarks. NoGTonline. NoAOT/3Dsimulation.

Importantflags:
- `--tiled`: seveninferences at1280x720, accuracybaseline.
- `--gpu-preprocess`: native CUDAOpenCV upload/letterbox directly intoTRTdeviceinput,
  validated. Faster. CPU/CUDAresize differs for tinyobjects; recordbackend.
- `--adaptive-tiles`: experimental memory2nearesttiles10calls,fulldiscoveryevery5;
  reducescalls~7→4.6,roughlysimilar coverage,notunconditionalbest.
- `--appearance-bridge`: optionalexistingFastVisionLoRAT adapter(read-onlydependency),
  max3SOTtargets,detector every3processedframes,SOTscore>=.5uncalibrated. SourceSOT
  vsdetexplicit; independent corrections,class-agnostic duplicate merging.
- `--detector-scale-only`: bridgecentroids butonlyYOLOspanhistory,moreabstention.
- `--geometry projective`: experimentaljointinverse-span fit; exact2Dtests pass,
  no clearrealaccuracygain. Defaultlogspan4s.
- `--camera fixed` is DECLAREDacquisitionassumption forQUTtripod,neverautoapplytoFPV.
- DefaultNOW rejects contraction-only clearance (`shrink_ambiguous`). Legacy
  `--allow-recession-clearance` reproduces oldstable-shape hypothesis. Thisdefault
  changed AFTERfullbridgecampaign found3false-clearframes onM4. Oldsource archived.
- `--sky-mask PATH`: optionalDINOsky-conditioned bright/darktinyproposal+YOLO96px
  cropinferenceevery5processedframes, originalpixels preserved. Masks source-aligned.
  `python -m scripts.run_sky_demo VIDEO --camera fixed --output DIR` executeswhole
  DINOprocess+actualdetector+render workflow,measures bothpasses/startups/export.
- Default --video-timing pts now uses actual source presentation timestamps and
  PyAV H.264 output. VFR real-RGB trial:70/70 frames, zero output PTS error,
  7.84s packet presentation end preserved. Source stream/format duration metadata
  was inconsistent; packet PTS+duration independently verified. CFR legacy optional.
- --correlation-seconds .5: optional Bartlett HAC sensitivity; correlated AR noise
  remains poorly calibrated, so not a default or confidence guarantee.
- --geometry bounded: experimental set-membership inverse-span fit with ASSUMED
  centroid +/-2px/span +/-10%. Controlled containment passes, real M1 353/438
  approach windows infeasible; only36 warning frames. Not an accuracy upgrade.

## Scientific limits / never overclaim
Imageplaneintercept is NOTtrueCPA. `scripts.bearing_limits`:2162Dpinholecases,
16false-clear examples at80–85deg evenperfectresolvedobservations, bothestimators.
Unknownintrinsics/shape/ownship motion prevent metricphysicalseparation guarantees.
Green is conditionalimagepass, never safe command. Advisoryalwaysabstains; scores
comparehypotheticalcamera-referenceimageflows,notknownbodyaxis maneuvers.
M3nominalapproach label CONFLICTSwithrange3562→4203,1071/2140increasingsteps.
LocalM3CSVidenticaltoauthorsHFmirror(SHAinreports/M3_label_provenance.json).
Keepnominalcategoryscores distinctfromgeometrytruth. M2/M5include late rangeincrease.
No truecollision/CPAlabelsavailable. Do notcall8/8verifiedimminentcollisionrecall.

## Data / verified evidence
15QUTlast35sclips preparedwithpublished25FPS,exactIDs/hashesin data/qut/*.manifest.
FullM1–M8:26,684timelineframes/1067.36s,5340samplesatstride5,13negativeeventIDs.
M4_1RGBnotavailable(labelintervalbeforefirstsourceframe),excluded denominator.
Missing TIFFs transparentlyrepeatedinfullpreparation(M5etc),noneinlast35ssubsets.
Sourceauditreports/qut_source_audit.json,rangeauditreports/range_audit.json.

Denseconfirmed OLDrecessionrule fullcampaign (`artifacts/full_campaign`):
8/8nominalapproacheventswarned>=.8s;0approachnonthreatframes.
13other events:0sustainedwarningsvs9growth-only, BUT1other eventhas2warningframes
(maxpulse0.4s). Positive2128opportunities441threat/1319uncertain/368untracked;
negative3273opportunities175nonthreat/1839uncertain/1257untracked/2threat.
Theseoverlapdevsubsets,notnewindependenttest. Figure reports/results_overview.png/pdf
Russianprimary,needsM3rangecaveataddedtofiguretext.

FullbridgeOLDrecessionrule (`artifacts/full_bridge_campaign`) COMPLETE:
8/8nominalevents warned;positive428threat/1299uncertain/398untracked/3nonthreat(M4).
Negative190nonthreat/1809uncertain/1271untracked/3threat(maxfalsepulse0.6s),0>=.8s.
Thus NOTaccuracydefault. Recessioncounterfactual `scripts.recession_ablation`
removesM4falseclear3→0,noobservationchanges;defaultnowshrinkuncertain.
11lateclipsbridgevalidation completed,all11events warned,0falseclearinthatwindow,
negative187nonthreat/620uncertain/651untracked,no warnings. Mixedcoverage.

Reservedend-to-end S6/S7 testedaftercurrentno-recessionselection:
2/2warnings,0falseclear;876opportunities244threat/605uncertain/27untracked,
852rawdetectormatches. `artifacts/final_reserved_dense`. RangeCSV metadata was
previouslyaudited;notusedtotunethresholds. Don'treuseandcallfuturemethoduntouched.

GPUpreprocessing paired18frames:singlecrop7.63→3.80ms;dense42.79→22.29ms,
29/29detectionsmatched,median0pxdense(maxconfdiff.0054). Singlecropmedian.64px,
maxconfidence.1245,soresamplingcanmatter. M1/M2GPUfullnorender~30FPS;warnings
117/209vsCPUpipeline118/210,negativeM278vs80nonthreat,no newfalseclear.
Fullrenderedneuralend-to-end cleanbenchstillpending. CachedreplayFPSisnotinference.

Sky/crop recovery:
ActualSegDINOprovidedcheckpoint144x256,batch4,875framesM1/S2,losslessmasks cached.
No skyaccuracyclaim. AllskycaseM1gainmodest30→39rawnegativematches,tracked35→42.
S2baselineuntracked251/438→113,croppeddetector208vs166matches;moreuncertain
observations,warning129unchanged. Mainrecallgainfromzoom,notprovedskysegcontribution.
Completeactualworkflow `demo/S2_sky_enhanced/workflow_benchmark.json`:
50.205sfor438processedframes =>8.724FPSincludingbothprocessstartups/seg/export;
assessmentprocessing13.745FPS. No streaminglatencyclaim. S2wasdiagnoseddata,notheldout.
Need inspectS2enhancedvideo/earlypreview andcompareunmatchedfalseproposals.

RendererfontglyphLRU512entries:paired190framescached22.71msvsuncached36.27ms,
actualrenderedcachedframeinspectedreadable. NotapplicationFPS. Camera/statusaxesnow
Russian;zoomstripoutsideactualsourceview. Sourceaspectletterboxed.

## Artifacts / commands
Primarydemo guide `demo/README.md` Russian; technical README,RESEARCH,GEOMETRY,
EXPERIMENTS recordprotocols/limitations. SomeREADMEflags/docsstillneedlatestupdates.
Actualvideos: demo/M2_confirmed/annotated.mp4(simultaneousstates~15.5s),
demo/M1_tiled_initial/annotated.mp4(tinynegativemisses),demo/S2_difficult/annotated.mp4,
demo/formation_reference_reset/annotated.mp4(turns,abstention),
demo/helicopter_252/annotated.mp4(137uncertain/1nonthreat,nothreatGT),
demo/S2_sky_enhanced/assessment/annotated.mp4(newrecallworkflow).
LoRATisolatedoracle-seedtrial isNOTe2e;M1negativehighscoresstilldrift(77/140accepted
frames>20pxerror). No calibratedtrackingprobability. Externalcodehashesrecorded.
Hardware/fingerprints/benchmarksallJSON. Tests35pass:
`.venv/Scripts/python.exe -m unittest discover -s tests -q`.
Sourcesnapshots artifacts/source_v1,source_confirmed_v2,source_adaptive_v1,
source_bridge_before_recession_gate preserveoldbehavior. No gitrepo.

## Current running processes / next work
Current clean actual CPU dense rendered M2 benchmark running (session73021),
output artifacts/runtime_final/cpu_dense, reports/runtime_cpu_dense.log. Run GPU
dense/adaptive sequentially afterward to avoid GPU contention. As ofcheckpoint
no other known pending experiment except any tool started after this file write;
S2skyworkflow,fullcampaigns,bridgevalidation,skyproposalcachescompleted.
Nextvaluablework:
1. InspectactualenhancedS2video andupdategallerywithfairbefore/aftercoverage.
2. Cleanactualneural+renderbench CPU/GPUdense/adaptive,stagepercentiles,codecplayback.
3. Testtemporalcorrelationuncertainty (ARnoise / HAC/blocks);plannednotimplementedyet.
   Statsmodelsofficialcov_hacdocsread(primary);don'tclaimcalibratedbounds.
4. CorrectlyhandleVFRtimestamps orrejectwithdocumentedpreflight;ordinaryCFR works.
5. Camera/parallax/turnstress,highresolutiontilingcoverage(arbitrary>1280gridcurrently
   only3x2locations,fullframedetectorstillruns;nativegridcoveragecouldbeimproved).
6. Skyproposalablation onnegative/crossing/clutter withoutclaimingmaskcausality;
   quantitativeunmatchedwarningepisodes andlargermeasurementuncertaintyanalysis.
7. MoreRGBexamples/classesifvaluable,GTsemanticsstrict. NoAOT,no3D,noheavytraining.
8. FinalartifactQA,dependencyfingerprint,reprocommands,Englishsummary/Russianguide.
Keepworkingthroughremainingusagebudget;writefreshstateatmilestones/~hourly.
