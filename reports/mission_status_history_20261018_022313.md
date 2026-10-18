# Active mission — authoritative resume state

Checkpoint 2026-10-18 04:10 Europe/Istanbul / 2026-10-18 01:10 UTC.
Goal ACTIVE. User budget35 additional weekly allowance percentage points, baseline0;
latest ACTUAL account reading22% at2026-10-18 00:58:28UTC. Integer/account-wide,
not isolated to this chat. Do not infer quota from tokens/time or stop at minimum.
Freeze risky work around30; finish by35 or when no defensible experiments remain.
Mission start2026-10-18 01:11Istanbul /17Oct22:11UTC. No subagents authorized/used.
Not a Git repo, no AGENTS. Goal's missing10H reference resolved to supplied5H spec;
wall-clock limit superseded by quota. User forbids questions, AOT, building3Dsimulator.

## Environment invariant
Use .venv/Scripts/python.exe directly. NO uv sync or OpenCV wheels. Custom cv2
5.0.0 at C:/Data/DEV/Libs/opencv_cuda_5/install/python/cv2/__init__.py, CUDAdevice1,
RTX4060 ~8GB,driver610.88,i5-12400F. Torch2.12+cu130,TRT11.4. Only av18.1.0 added,
--no-deps after1-package dryrun; lock addedonlyav. Fingerprints initial/after_av JSON.
Mainenv lacksSciPy/Matplotlib; sibling anti_collision_system env usedread-only for
those. SkySegDINO ownenv usedforsegmentation. Never overwrite external sourcefiles.

## Current implementation and defaults
Main: python -m airavoid VIDEO --camera auto --output DIR (main.py same).
Default --profile conservative:conf.03,dense native crops,GPUpreprocessingauto for
TRT+CUDA,internalcropedge2px,sourceedge1px,HAC1s forlog,window4s,confirmedpolicy
(.24swarn/1.2spass),contraction-onlypass disabled. Explicitflags override.
--profile reference:conf.18,full-frame CPUpreproc,zeroedges/HAC,allows similaritypass;
legacycontraction requires --allow-recession-clearance separately.
--full-frame/--cpu-preprocess override auto; adaptive/SOT optional, notselected.
Sourcegeometry uses image drift/growth and depth-plane miss in targetspans, NOT
trueCPA. Nativegridcoversarbitraryraster:1280x720=7calls,1920x1080=17,4K=50.

Cameraauto: KLT/RANSAC2Dsimilarity withreferenceepochs/resets. NewdefaultWITHHOLDS
allpassstates afterauto evenvalidbackgroundfit: rotation/translation/parallax
unresolved. Potentialwarningsremainconditional. --allow-similarity-pass restores
researchhypothesis. Camera fixed is DECLAREDacquisitionassumption, never silently
applytoairbornevideo. Newreasonparallax/image_rule_candidate_state recordsactual
imagecandidate. Calibratedgeometry requiresrealJSONintrinsics,rectifiedsquarepixels,
matchingresolution andfixedcamera; noQUTfocalguessed. Projective/bounded experimental.

AdvisoryalwaysNone/refusal. Compares commonimageflow width/64px/s under equaldepth
hypothesis, minimum changeof drift/span acrossquality-eligibleobjects, unresolved
count andlimitingtrackIDs. Bodyaxes/ranges/dynamicsunknown; numbersnotflightcommands.
RussianUI1800x1140,sourceaspectpreserved,Why3targets,actualbounds,2s histories,
staleboxhidden/stalezoomboxplaceholder. Sourceviewportclips; residualcirclescales
native-to-display,cosmetic4–50pxlimits explicitlycaptioned. Class/scoremaybewrong.

SourcePTSpreservedviaPyAV/libx264 orNVENC; VFR70frameactualRGBtrial zerooutputPTS
error,7.84sduration. Sourcebadstreamduration7.6 nottrusted: packetendrelativefirst
actuallydecodedframe. Nonzero5sstarttested. Corrected origin_ms units (originis_s).
RunSession recordsrunning/failed/complete,removesstalebenchmarkonrerun,cleansowned
resources. InputSHA/cacheSHA/upstreammetadata; digest/timestamp/malformedcache guards.
Legacycacheunverified explicitly. ReplayFPS NEVERneuralFPS.

## Model assets
Supplied ../anti_collision_system YOLO26m fiveclasses. ONNX1x3x416x704 ->1x300x6.
OldTRTengine243vsruntime245 incompatible. Locallyrebuilt artifacts/yolo26m_local.engine
110.09s; FP16-derived/mixed,IOFP32. ORTCUDAcuDNNmismatch leftunrepaired; directTRTworks.
ONNXCPUvsTRT18frame14/14matches. GPUvsCPUpreproc18frames dense29/29median0px,
42.79→22.29ms; smallresamplingchanges CANalterwholetrajectorydecisions.
LoRAT suppliedFastVision read-onlyadapter,3slots,rawscore>=.5uncalibrated,YOLOevery3
processedframes. Engine artifacts/lorat_local.engine rebuilt38.59s. SOTdriftknown:
M1other77/140acceptedGT-visibleframes >20pxwrong. No GT seeds inactualbridge.
Detector-scale-only usesindependentYOLOspans. Prototype silhouette refinementrejected.

## Data and strongest measured evidence
All15QUTfullsources data/qut_full:55,531TIFF /55,535timelineframes,2221.4s (37.02min),
4missingIDs repeatedprevious (M5,S1,S2,S3). Paper55521 differs; actualcounts/endpoints
preserved inmanifests. RGBgroundedtripod,published25FPS. No truecollision/CPAGT.
M3nominalapproachrange3562→4203 conflict,CSVbyteidenticalauthormirror; donotfix/ignore.
M4_1negativeintervalbeforeavailableRGB excluded =>13other eventIDs. GTapprox/interpolated
centroids,20pxmatch; reportmisses/uncertainties. Allsourcespreviouslystudied; NOTnew
untouchedtest or detector-training-independent. S6/S7oldreservedalreadyconsumed.

SELECTEDactualGPUlowconf.03fullcampaign artifacts/confidence_full/neural_low and
assessment/0.03,11113processedframes,stride5:approach4031opportunities=
802warning/2300uncertain/929untracked; other3273=0warning/156pass/2345uncertain/
772untracked.15/15nominalevents warning>=.8s,0approachpass,0/13knownotherwarnings.
UNMATCHEDwarning37 NOTverifiedfalsealerts. Samequality/policygrowth-onlybaseline:
10/13other eventwarnings,476otherwarningsamples. Sourcecase!=physicalcollisiontruth.

Conf.18 withsameguards:positive801warn/2064unc/1166miss;other151pass/1866unc/1256miss.
Lowconf gains237positive+484othertracked samples,mostlyuncertain. Gainpersists5/10px.
Filteringlow.03at.18 EXACTLYmatchesactualseparate.18boxesANDcameravalidity acrossall
11113samples (0diff). Nuisance/unlabeleduncertain3722vs1026;freshtracks mean.84vs.53,
p95both2,max5vs3. Thresholdcurve.03/.08/.12/.18/.25/.4/.6 recorded,posthocdevelopment.

ReferenceGPU.18/noedge/HAC15full:789warn/2073unc/1169miss;other7warn/170pass/1840unc/
1256miss. M6otherpulse1.4s; selectedHAC1removesitbywideningmargins. CPUhistorical8M
had2otherwarningframes/.4s (same29/29pairedpreprocessmatchesdidnotproveallequivalence).
Fullfixedgeometrycache reconstruction matched ALLstates/reasons/numericalmetrics
exactly(maxerror0). HACcorrelated-noisecoverage stillpoor (rho.95~58.8% nominal2.5SE),
NOTcalibratedCIs. Boundedassumed2px/10%mathenclosurespassbutrealM1353/438infeasible.
Projective/calibrated exactanalyticalmathdoesnotestablishactualworldmodelvalidity.

Newparallax_limits:324analyticalcases/method,297within smallsimilaritymotionbounds;
falsepasses96logHAC1/78projective/6bounded among297 withperfectbackgroundcorrection.
ActualKLT/RANSACsynthetic2Dtexture+analyticalboxes:62/63validfits,old22confirmedpass
samples,trueCPA0/rawwouldwarn; guardreturns63uncertain. NotRGBencounter/NN/prevalence.
No3Dsimulatorbuilt. scripts/camera_parallax_probe andtests regression.
Widebearingtestalso16falsepasscases evenperfectobservations; uncalibratedplane miss
notCPA. Optionalcalibratedmodel onlyanalyticalvalidation,requiresknownconstantspan/K.

## Performance, videos, reports
ActualconservativeM2,438samples/stride2,NVENC:serial14.8436FPS,thread21.3467,
process24.0883. Pipelinepreflight/startupinclusive13.8295/19.2356/21.1179 (notPython
importtime). Separateprocessstartup.393s,workerCV2pathsame,12threads. ALL438decoded
pixels,det/decisionfileSHAs,PTS matchserialexactly;35.000sH264yuv420p. Queuebound2.
artifacts/async_trial/*,reports/async_equivalence.json,process_equivalence.json.
LatestUIclipping/camera-scope addedafterthese; final3repeatcurrentbenchstillneeded.
RGBNVENCinputprobe rejected:isolated106.04YUVvs105.51BGR0FPS,no speedgain,PSNR46.6dB
betweenoutputs. Constructorprototypeoptiononly,no CLIpromotion.

Currentgallery demo/M2_conservative ishash-recordedcopyofactualthreadrun with14s
preview,green+yellowthenred. OldM2_confirmed briefred/green overlap involvedclipped
object; NOTdefensiblesimultaneousgeometrysuccess. FinalcurrentUIliveNNrerendersneeded.
Otherworkingoldclips:M1_tiled_initial,S2_difficult,formation_reference_reset,
helicopter_252,S2_sky_enhanced/assessment. PrimaryguideRussian demo/README.md,
readableEnglish RESULTS.md, technical README/GEOMETRY/RESEARCH/EXPERIMENTS.
reports/results_full_extension.png checked; runplotusing siblingenv,notmainMatplotlib.

SkyactualDINO144x256,batch4 provided299MBcheckpoint. M1/S2cachedmasksartifacts/sky.
OldS2sky/crop workflow actualbothpasses/startups50.205s/438=8.724FPS,main13.745FPS;
untracked251→113,warning129unchanged. MaskcontributionvszoomNOTisolated,allskyM1
minimalgain. Newselectedcheap.03stillmissingfreshS2targetat10s. Needcurrentcomparison.
SilhouetteDINOcontext+pixelcomponentchanges reducedM1warnings117→92,S2129→75,
S2untracked251→263; rejected,notmain. Sourceclasspredictionscanbewrong; no bird/
balloon/parachuteindependentthreatvalidation. Realhelicopter/FPVqualitative,nocollisionGT.

## Running work and next priorities
RUNNING session92425: scripts.selected_methods (PYTHONDONTWRITEBYTECODE1), outputs
artifacts/selected_methods +reports/selected_methods.json/log. Actualdense/adaptive/
appearance/independent-spans onM1/M2/S2last35s andFULLM4, allstride2/currentprofile.
NootherknownGPUjob. WaitbeforeGPU/performancework; don'tduplicatecompletedcampaigns.
68tests passed WITH AIRAVOID_TEST_NVENC=1 beforelastsmallUIclip changes; portablesuite
skipsonlyNVENChardwaretest. MPworkeractualtestsincludeVFR/error/ownedchildexit.

Next: readoptional-methodresults; currentactualSkyworkflowandmask-vs-allskyifvaluable;
finalliveNNM1/M2/S2/nativeFPV/helicoptervideoswithprocessNVENC/currentUI, inspect WHY/
alignment/parallaxscope. Clean3repeatcurrentbenchaftercorefreeze, maybe stride1/nativeres;
refreshdocs/plots/artifactindex/environmentfingerprint, tests andcodec/PTSQA. Freeze
near30actualadditionalweeklypoints; finishby35. Neverclaimreplay/codecprobeFPSasNN.
NoAOT/3Dsim/heavytraining. AdditionalrealgeometryleadsVigilmanuallygated,MMAUDlinks403;
oldcommentedNTUDOIactuallyNTUVIRALnotMMAUD. No accessrequests/bypasses attempted.

## Checkpoint — 2026-10-18 01:23 UTC
Reliable account usage remains24% (baseline0, mission cap35additional points).
Selected-method campaign finished; E45 and reports/selected_methods.json hold all16runs.
Running full-rate cadence PID15732 (wrapper)/18056, now M6 after M1/M2/M4 completed.
All3 completed sources have0approach passes/0knownother warnings. M2 earliest
qualified-warning onset119.4s atstride1 versus139.2s atstride5, so cadence matters.
reports/warning_timing.json/log analyzes sampled episodes with explicit gap resets,
retrospective onset and later support qualification; source annotation end is NOTCPA.
Full15stride5:15/15nominal approaches have2s warning episodes,14/15have4s; earliest
qualified onset is3.4–22.2s before last GT sample (notimpact), M3rangeflag retained.

New changes: SkyProposals verifies FPS/framecount and optional source+mask SHA256
sidecar, records legacy_unverified when absent. --sky-all-control is a matched
whole-image proposal-region ablation, not standalone classical sky segmentation.
run_sky_demo now CFR-preflights before segmentation, writes identitymanifest, accepts
NVENC/process rendering and owns failure status; lifecycle supports workflow report.
5targeted checks passed; full suite71passedwithNVENC before two new timingtests,
which separately pass. Current total expected73. No new dependencies.

Queued final production exec session59464 waits ONLYonPID15732, then runs
scripts.final_production sequentially: current liveNN M1/M2/S2/formation_native/
helicopter videos in demo/*_final, two M2repeats, actual S2DINOworkflow, all-region
control and confidence-floor .005/.01/.03/.18 experiment (latter neverdefault).
No other GPUjobs should be started until this finishes. Perjoblogs
reports/final_production/*.log and reports/final_production.log; cadence in
reports/cadence_fullrate.log and artifacts/cadence_fullrate/*.log. Lost execsession
can be monitored via Get-CimInstance; do not duplicate jobs.
Next readactualcadenceresults, inspectfinalvideoUIandtimestamps/framealignment,
compare current sky/control and lowconf nuisance/recall, rewrite concise readable
README/RESULTS/demo gallery, provenanceindex and finalenvironment/tests. Corefreeze
~30points; final~5points forQA/handoff. Mission ACTIVE, do notstopat24points.

## Checkpoint — 2026-10-18 01:46 UTC (supersedes earlier running-job notes)
Reliable account meter: 28% used, baseline 0; cap 35 additional percentage points.
Full-rate validation and bridge extension have COMPLETED. Do not rerun them.
19,370 actual frames on M1/M2/M4/M6/S2: approach1487warn/3607unc/1916miss/0pass
of7010; other26warn/434pass/4776unc/862miss of6098. M6 other warning96.16–97.16s
is real category mismatch, 26frames/1.04s, unlike stride5 zero-other result.
Audited image/report and demo/M6_failure (12s recorded-decision excerpt) checked.

Complete bridge extension on FULL M6/S1/S3/S5, allstride2:
Dense approach514warn/1136unc/924miss of2574; other12warn/12pass/441unc/30miss of495.
Mixed SOT442/1142/990; other13warn/0pass/446unc/36miss.
Independent-detector-size SOT450/1134/990; other0warn/5pass/454unc/36miss.
All4nominal approach events warned in every method; no approachpass. FPS ranges
30ish dense,35.9–52.5mixed,38.4–52.4independent (no render, exploratory throughput).
Keep dense default; optional independent-size bridge has tradeoffs, not broad win.

All final core liveNN videos generated: demo/{M1,M2,S2,formation,helicopter}_final.
M2current3runs:21.805/22.698/23.433FPS; pipeline-startupinclusive19.329/20.057/20.463.
Native1920FPV11.994/10.793FPS; alluncertain. Helicopteralluncertain. M2decoded14/27s
and M6failurepreview visually checked; UI readable/aligned, no misleadinggreenclip.
Full suite last71passed before4newtests (expected75now); targeted sky/zoom5passed.

Current conservative S2 sky vsallregion BEFOREnew tiny-crop edgeguard has exactly
identical detection and decision SHA256. Dense136warn124unc178miss vs147/242/49.
WholeDINOworkflow48.7886s=8.9775FPS; stage15.9492FPS. DLLPATHfailurefixed by preserving
incoming environment BEFOREcv2import for isolatedSkychild; actualrerunpassed.
Skywrapper nowCFRpreflight/hashmanifest/failurestatus; standalone masks legacyunverified.
New --zoom-proposals supports actual all-pixel YOLO crop path withoutDINO. Lastsmall
fix applies existing native crop-edge margin to zoom crops as well. No main-path
logic changes. Core should nowfreeze except concrete QAfixes by~30points.

RUNNING exec84429 waits completedbridgePID6396 then liveNN zoom:
demo/S2_zoom_final, reports/S2_zoom_final.log (likelyfinished soon).
RUNNING exec25938 scripts.finalize_sky waits zoom benchmark, then CURRENT guarded
DINOworkflow demo/S2_sky_current, allregion artifacts/sky_control_current/S2_all_region,
reports/sky_comparison.json. Do not startotherGPUjobs until itcompletes.
Oldsupplement exec76350/comparison done; originalfinalproduction session59464 failed
atoldSkyDLL step, corevideos allcomplete. Resumedsupplementcompleted E48–50.

New diagnostics complete: residual quality sweep .08/.1/.12/.16 retains15/15nominal
warned events butapproachwarn653/685/729/802; no defaultretuneonM6. confidencefloor
.005/.01/.03/.18 tradeoffs; postfiltered.03 EXACTr awboxes/camera-valid/state-reason
matches current actual.03 on all1314samples (reports/confidence_floor_verification).
AIR2LAND317pairedimages fourIDs,1missingcolor; fullframe310/317IoU.5 vsnative312,
314vs388predictions. Visibility/occlusionlimits; noFPS/K/poseunitsguessed, nosimulator.
reports/air2land_detection.json/preview and aspect.json; allsourcehashes recorded.

README/RESULTS/demo/README rewritten/updated readable; build_gallery.py awaiting
S2_zoom_final complete. Russian warning_timeline.png checked, all15statehistory.
scripts/final_video_qa.py prepared butNOTYETRUN: checks8finalvideos/currentSky,
allactualdecodedframes+PTS/codec/duration/sourcehash,3source-render samples/video.
Need run it aftercurrentSky, fix any actual issues; then entire75test suite WITHNVENC,
check_environment.py exactcustomcv2initialbuild+CUDAfunctional check, artifactindex,
currentthree-waySky figures/docsstatupdates/gallery generation and visualHTMLcheck.
Then timestamps/usage finalreport andgoalcomplete onlywhenallrequiredworkachieved.
Potential worthwhile small-video optimization found: Detector currentlyduplicatesfull
pass when nativecrop covers entire source (<=704x416); not implemented/measured yet.
Do not spendremainingbudgetonpadding. Final leadplayablevideos,Russianvisualoverview,
then plainEnglishsummary/docs, limitations, actualintegeraccountmeter/timestamps.

## Checkpoint — 2026-10-18 02:11 UTC — FINAL QA, 32% used
Actual weekly meter32%, baseline0, cap35. Algorithms frozen except a concrete
clipped-history correctness repair now being audited. Do not start new experiments.
Remaining sources in all-frame extension: S5/S6/S7; others complete. Actual rootjob
scripts.complete_cadence exec5023, reports/cadence_all_sources.{json,log}, source logs
artifacts/cadence_fullrate/SEQ.log. M7 also has14other warningframes at63.8–64.32s,
0.56s (below0.8s event criterion); M6has26frames/1.04s. Do NOT hide M7 by event cutoff.
M7source audit PNG/JSON saved; it hasNOclipped history, so repair will not fix it.

Full-source oracle_centroid+ID analysis queued exec57564, scripts.oracle_fullrate;
reports/oracle_fullrate.json/log. Clearly OFFLINE GT ablation, not camera-only score.
Clipped-size exclusion audit queued exec94780 scripts.clip_history_audit, waits for
allNN15sourcecomplete then geometry-only replays allactual observations. It uses
ExcludeClippedTracker subclass; it clears geometry histories AFTERsuper._observe
on clipped observations, preserving raw history. Core is NOTPATCHEDYET.
Bug: current _observe clears history then immediately appends clipped sample,
contradicting intended exclusion. After fullNN/auditfinish, apply narrow fix:
append tr.history and tr.scale_history ONLYif nottr.clipped; raw_history always.
Add behavior test that2s valid history excludes earlier clipped frame (expected76tests).
Wait for audit/oracle tofinish BEFOREediting core or running clean performance.

scripts/final_clip_refresh.py PREPARED, NOTRUN. After patch and completed audit:
archives8canonicaldirs under artifacts/pre_clip_history, archives QA/reports, updates
historical external-benchmark reference paths to copied exact traces, then refreshes
6actualneuralclips, current DINO/control via scripts.finalize_sky, M6recordedexcerpt
from corrected audit trace, and exactreplay→freshM2 consistency check.
If run, do not rerun fromscratch after partialfailure because archivealreadyexists.
Need inspect output and finish/recover onlyremainingsteps iferror. It expects
reports/clip_history_audit.complete. New report reports/final_clip_refresh.json;
reports/clip_current_equivalence.json should0mismatches. It deliberately doesnot
repeat confidence/residual/appearance campaigns; those are historicalrecordedphases.
Afterrefresh regeneratevideo PNGs/HTML, runfinal_video_qa again (8videos/3000frames),
then fullsuite76withNVENC (newfixjustifiesrerun). Update build_artifact_index assertion
75→76. environment_final already exactinitialcv2+CUDAfunctional (02:49? actual01:49UTC).

Current final export QA completed75tests/8videos3000frames, PTSmax7.1e-15s,
3source/render samples each maxRMSE4.06. Browsergalleryplayback advanced19s/35s;
app fileURLpolicyreject (no bypass), temporarytab+serverPID18404closed, port62152gone.
CustomOpenCVbuild unchanged. externalbenchmark9freshprocesses completed, alltraces
exactlymatch current(beforeclippingrepair)canonicals. FPSmedians19.8845M2,
14.4324S2zoom,10.9183nativeFPV; laterarchiveoriginalreferencesifrefreshing.

Current Sky aftertinycropedgeguard: dense136warn124unc178miss vs147/242/49for
zoom/sky/allregion, all438opportunities. ALL3proposalpaths have identical raw
anddecision hashes. Current fullDINO48.5415s=9.0232FPS; stage15.9422. Zoom16.2184
processing/14.8344pipelinewhole (imports excluded). Reports/sky_comparison.json.
Main cliprefresh will regenerate these reports/timings for repairedcore.
Small640 duplicate fixdone: actual438before/after exacttracehashes,2→1calls,
73.8768→99.3421no-renderFPS.75tests passedafterthatfix.

README/RESULTS/demoREADME updated but pendingfinal all15cadence counts/cliprepair
andactualcurrentrefreshmeasurements. scripts/build_gallery.py createdHTMLgoodbrowser
layout; preview S2zoomframe10s.png nowexistsbutcard stillusespreview.png unlessfixed.
scripts/build_artifact_index.py PREPAREDNOTRUN: requires DELIVERY.md and
reports/mission_final.json, allcompletecadence+oracle, QA, tests countthen75(update76).
Needfinalsummarydocuments, source/reporthashindex, recordedUTCstart17Oct22:11Z,
actualfinalUTC/usage (integer/sharedaccount). Do notfabricate35; stopbybudget35.
Finalgoalcompleteonlyverifieddeliverable/bestworkingprototype achieved. Finallead
playableMP4links/Russianvisualoverview, then conciseEnglishsummary, limitations,
Englishdocs. NoAOT/3Dsim/heavytraining; neverreplaceCUDAOpenCV.
