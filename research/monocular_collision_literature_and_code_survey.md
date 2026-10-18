# Airborne RGB Collision-Threat Assessment — Literature, Code and Reproduction Survey

**Research date:** 2026-10-18  
**Project:** [kxsxlxv/AirAntiCollision](https://github.com/kxsxlxv/AirAntiCollision)  
**Scope:** independently implemented or reproducible algorithms for airborne target-specific **collision threat assessment**, not merely detection, tracking, optical-flow estimation or attractive video overlays. 

**Bottom line:** Reproduce **Bauer 2016/2019**, **AirTrack's looming/tCPA cues**, and **Lai & Lin 2024** against the **same frozen YOLO26m/MOT observations**. Compare separately with **Opromolla & Fasano** when synchronized ownship navigation is available and with the **ColANet** learned temporal classifier if the data/weights support a fair run. The existing project's multiple geometry estimators are *not* independent published method replications. Ground-fixed QUT results are *not* evidence of working moving-ownship collision discrimination.

## 1. Evidence labels and permitted reuse

- **A — public implementation:** author's/research group's repository available. This **does not** mean a complete RGB-to-threat package is ready or that weights are published.
- **B — published method:** paper provides derivation/algorithm sufficient to attempt independent implementation; no verified official executable found.
- **C — related component/data:** useful for detection, tracking, ego motion, supervision, calibration or physically grounded validation, but **not** a monocular threat-estimator baseline by itself.
- **Code license** and **paper/dataset license** are separate. A GitHub repository without explicit reuse permission must be treated as reference-only. Always inspect the precise commit and `LICENSE` before integrating.
- **Do not use Amazon AOT** for this project. A paper previously evaluated on AOT can still be reproduced on QUT/Purdue/TartanAviation. Do **not** build/run a 3-D simulation as part of this research stage. Sensor-assisted variants are legitimate but must be benchmarked in a separate category from RGB-only.
- **Monocular limitation:** calibrated cameras and stable target dimensions can supply relative range up to scale; unknown target dimensions, arbitrary attitude/translation, asymmetric aspect variation and short noisy tracks prevent general metric 3-D CPA or unconditional *safe-to-pass* inference. An abstention state is mandatory. Keep physical collision truth, dataset encounter categories and image-plane risk proxies distinct.

## 2. Primary threat/avoidance methods — proposed direct comparators

| Priority | Work and authoritative link | Original author code / weights | Required observations; actual output | Evidence and reproduction class |
|---|---|---|---|---|
| **P0** | **Bauer, Hiba, Vanek, Zarandy, Bokor (MED 2016)**, [*Monocular Image-Based Time to Collision and Closest Point of Approach Estimation*](https://doi.org/10.1109/MED.2016.7535965), [author manuscript](https://www.researchgate.net/publication/306113671_Monocular_image-based_time_to_collision_and_closest_point_of_approach_estimation) | **Official implementation/weights not verified** | Time series of target centroid and apparent size; inverse-size and image-projection fits; projected TTC, normalized CPA/decision quantities. Explicit geometric assumptions. | **B.** Full independent math reproduction before developing another custom estimator. |
| **P0** | **Bauer, Hiba, Zsedrovits (2019)**, [*Real Flight Application of a Monocular Image-Based Aircraft Collision Decision Method*](https://doi.org/10.1155/2019/7191839) | **Official software not verified**; original publication has [supplementary real flight videos](https://doi.org/10.1155/2019/7191839). | Extends preceding collision-decision geometry; compares estimated CPA/TCPA-type quantities with flight-trajectory-derived values; studies threshold selection, detection and real airborne encounters. | **B.** Strongest directly applicable published flight-validation comparator. Extract equations, all thresholds, degeneracies and target/ownship geometry; never claim full reproduction from a vaguely similar inverse-span method. |
| **P0** | **Ghosh, Patrikar, Moon, Hamidi, Scherer (ICRA 2023)**, [*AirTrack: Onboard Deep Learning Framework for Long-Range Aircraft Detection and Tracking*](https://arxiv.org/abs/2209.12849), [CMU publication](https://publications.ri.cmu.edu/airtrack-onboard-deep-learning-framework-for-long-range-aircraft-detection-and-tracking) | **No verified official complete AirTrack model/checkpoint repository.** [Flight video](https://youtu.be/H3lL_Wjxjpw); supplementary exposition in [TartanAviation 2025](https://doi.org/10.1038/s41597-025-04775-6). | Full-resolution two-frame alignment, cascaded aircraft detection/filtering, Kalman tracks, range/angular-rate/tCPA-style states; flight tests on Cessna/controlled helicopter encounters. | **B/C.** Independently reproduce *area-growth/looming and angular-state threat cues* from already available boxes; learned frame-alignment module is a separate unverified reproduction task. Published tracking probability is **not** collision-decision precision. |
| **P0** | **Lai & Lin (Remote Sensing 2024)**, [*Vision-Based Mid-Air Object Detection and Avoidance Approach for Small UAVs with Deep Learning and Risk Assessment*](https://doi.org/10.3390/rs16050756) | **No official full-stack implementation or weights verified.** | Image position/bearing, apparent area change, defined conflict/risk regions; avoids based on model-dependent risk. Real visual detection and simulated avoidance are separate evidence. | **B.** Independent *risk/classification* decision rule on frozen detections; compare with Bauer rather than fusing prematurely. Extract exact paper equations and parameters; do not invent unavailable ones. |
| **P0/P1 (navigation-assisted)** | **Opromolla & Fasano (Aerospace Science and Technology 2021)**, [*Visual-based obstacle detection and tracking, and conflict detection for small UAS sense and avoid*](https://doi.org/10.1016/j.ast.2021.107167), [institutional record](https://www.iris.unina.it/handle/11588/862741) | **No official code/weights verified.** | RGB detection and Kalman tracks; **ownship navigation** for stabilized line-of-sight and LOS rate; per-track conflict classification. Tested using real low-altitude dual-UAS imagery. | **B.** High-value IMU/attitude-assisted comparator; not equivalent to an uncalibrated image-only algorithm. |
| **P1 (learned alternative)** | **Pedro et al. (2020)**, [*ColANet: A UAV Collision Avoidance Dataset*](https://doi.org/10.1007/978-3-030-45124-0_5), [dataset site](https://colanet.qa.pdmfc.com/) | [**Authors' public project**](https://github.com/dario-pedro/uav-collision-avoidance): dataset preprocessing, training notebooks, temporal classification/escape-vector experiments. **Top-level repository license not verified**; published ready-to-run pretrained weights not established. | Video-level/temporal collision-relevant labels and learned warning/escape tasks; inspect obstacle types, splits and leakage before using. | **A (research code)/C (domain transfer).** Run only if data and preprocessing can be reproduced. Do not compare learned output with aircraft CPA labels as though task definitions matched. |
| **P1 (reference DAA software)** | **Stanford SISL AVOIDDS / VisionBasedAircraftDAA** [authors' source](https://github.com/sisl/VisionBasedAircraftDAA) | **Public source, MIT**; models/dataset code, X-Plane-dependent downstream encounter module. | Published downstream controllers use **3-D ownship/intruder or encounter state** (e.g., tau, relative altitude/velocity). Not a monocular box-only conflict classifier. | **A/C.** Read and reuse algorithm interfaces, policy/state conventions or controlled detector tests; do not present its simulator controller as camera-only. No simulation needs to be launched. |
| **P1 (physical-truth policy reference)** | **NASA DAIDALUS**, [official C++/Java/PVS repository](https://github.com/nasa/daidalus), [guide](https://nasa.github.io/daidalus/) | **Public source** under NASA Open Source Agreement (review `LICENSES`, not generic MIT). | Requires estimated **3-D relative position and velocity**, configuration of well-clear volumes and aircraft dynamics; outputs alerting, maneuver bands and guidance. | **A/C.** Use as *state-ground-truth/physical-policy benchmark* only where trajectory state exists, **not** on YOLO bbox alone. |
| **P1 (research direction)** | **ViSafe (RSS 2025)**, [authors' project](https://theairlab.org/visafe/), [website repository](https://github.com/castacks/visafe) | Public repo inspected is **website/assets**, not verified runnable perception/avoidance code or checkpoints. | Real demonstrated high-speed visual DAA with integrated sensors/perception/control and safety reasoning; different observability/computation assumptions from ordinary downloaded single-camera RGB. | **C** until usable implementation and complete input specifications verified. |

### Why the first four are not equivalent

**Bauer** estimates geometric encounter quantities under explicit projection/target-size assumptions. **AirTrack** provides an onboard detection/tracking system with angular/looming-based encounter states but does **not**, by its tracking benchmark alone, validate general CPA classification. **Lai & Lin** defines an image-driven risk/avoidance decision framework. **Opromolla & Fasano** explicitly uses ownship navigation to stabilize LOS and estimate LOS rate. Their *input evidence*, *decision semantics*, and *validation data* must be annotated individually. Do not invent one shared physical-metre CPA metric for all.

**Important AirTrack provenance correction:** a later paper by AirTrack authors, [TartanAviation (Scientific Data 2025)](https://doi.org/10.1038/s41597-025-04775-6), states that AirTrack auto-labeling emits target tracks, 2-D Kalman state, range, angular rate and tCPA. The paper's range may involve estimated quantities and external ADS-B for post-filtering; **never treat those range outputs as independently validated RGB-only 3-D distance without checking their derivation**. The TartanAviation public repository contains **dataset scripts**, not a verified release of AirTrack inference weights.

## 3. Moving-camera perception, tracking and open software

These are **not competing collision-decision algorithms**. Use them as independent observation producers and control experiments.

| Reference | Source | Available contribution | Limitation for this project |
|---|---|---|---|
| **Li et al., IROS 2016 / IEEE TETC 2021** *Fast and Robust UAV to UAV Detection and Tracking from Video* | [Official Purdue author source](https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking), [paper DOI](https://doi.org/10.1109/TETC.2021.3104555) | Real onboard-camera detection, background/camera-motion compensation, tracking; older **Keras** implementation; repo reports **BSD-3-Clause** | Excellent comparison for moving-camera target continuity; not ground-truth collision classification; older CUDA/dependencies need isolation. |
| **TrackTrack (CVPR 2025)** | [Official code](https://github.com/kamkyu94/TrackTrack) | Tested high identity continuity in this project's MOT campaign; pedestrian FastReID may not represent tiny aircraft well | Must evaluate downstream warning/false-clear effect; cannot substitute smoother tracker predictions for independently observed target size. |
| **PuTR (2024/2025)** | [Official code](https://github.com/chongweiliu/PuTR), [paper](https://arxiv.org/abs/2405.14119) | Transformer association using externally provided detections; already benchmarked | Requires proper online causality and fixed observation streams. |
| **BoT-SORT (2022)** | [Official code](https://github.com/NirAharon/BoT-SORT) | Third detector-fixed association baseline, camera-motion and appearance logic | Featureless sky / target size / pretrained person embeddings can degrade transfer. |
| **Roboflow trackers** | [Repository](https://github.com/roboflow/trackers) | Convenience wrappers for standard MOT controls | Not an aircraft threat model. Preserve baseline separation. |

**Already done in AirAntiCollision:** `MOT_RESEARCH.md` and `MOT_RESULTS.md` benchmark TrackTrack, PuTR, BoT-SORT, OC-SORT, DiffMOT and TrackSSM with the provided YOLO26m, including replay-vs-neural timing distinctions and extreme detector recall failures on some UAVSwarm clips. Do not run another expensive pedestrian-MOT leaderboard. Instead feed the **same** frozen detector outputs to selected association methods and compare the **threat states**.

**Reid vs geometry:** A predicted/smoothed/interpolated bbox may preserve identity but is not independent evidence of expansion. Schema must identify *measured detector span*, *observed centroid*, *predicted centroid*, *source* and *age*. Deny false growth due only to prediction.

## 4. Datasets: precisely which labels exist?

| Priority | Dataset / source | Modality and available truth | Valid claims and critical limitations |
|---|---|---|---|
| **P0** | [QUT Aircraft Collision Course](https://open.qcr.ai/dataset/aircraft-collision-course/), [paper](https://arxiv.org/abs/2112.02735) | **Real RGB sequences** shot with stationary/ground camera. 15 recorded approaches, 55,521 images reported by paper; dataset distinguishes a designated approaching aircraft from additional 'other' aircraft using approximate image centres. Some approach targets have reported range. | Excellent **encounter-category proxy** for first controlled baselines; **no verified multi-target physical 3-D CPA truth**. Current repo documents M3 range inconsistency and missing/repeated source frames. Overlap with YOLO training is unknown. |
| **P0 (moving camera; ALIAS)** | **NPS-Drones = Purdue UAV Dataset**, [original dataset/labels](https://engineering.purdue.edu/~bouman/UAV_Dataset/), [Purdue author code](https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking) | **The same 50 real flight videos / 70,250 frames, NOT two independent datasets.** The footage originated with the US Naval Postgraduate School (NPS), was annotated/released by Purdue collaborators, and is often called *NPS-Drones* in later papers. Moving on-board camera, UAV bbox labels, **no verified 3-D collision/CPA labels**. | The user already has NPS-Drones with **DogFight-reannotated** bboxes: **reuse this local video corpus; do not download a duplicate Purdue dataset**. Test camera compensation and association; distinguish label revisions. |
| **P0 (annotations, same video footage)** | **DogFight**, [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ashraf_Dogfight_Detecting_Drones_From_Drones_Videos_CVPR_2021_paper.pdf), [authors’ code and labels](https://github.com/mwaseema/Drone-Detection) | Re-annotated **NPS-Drones and FL-Drones**, correcting many oversized/loose original boxes (Fig. 8). DogFight GT is a **different label version on the same NPS video frames**, not new data and not physical CPA. Purdue also distributes its **own updated annotations**; do not assume the Purdue update and DogFight annotations are byte-identical. | Use the already downloaded DogFight labels as the canonical bbox-GT version after frame-index/coordinate-order checks. Keep dataset provenance and label version explicit; avoid train/test leakage if YOLO26m was trained on this corpus. |
| **P0 (different authors; synchronized labels)** | **TartanAviation**, [2025 paper](https://doi.org/10.1038/s41597-025-04775-6), [official repository/downloader](https://github.com/castacks/TartanAviation), [dataset site](https://theairlab.org/tartanaviation/) | **Real stationary airport cameras**, 550 image sequences, 3.1M images overall, boxes with identities and **range** per aircraft, synchronized ADS-B trajectories and camera calibration/extrinsics used during label creation. Dataset tooling supports partial downloads. | Very promising **range/association/calibration validation** and independent held-out airport scenes, but camera is **ground-fixed**, ADS-B coverage/accuracy and labels may be incomplete, and ADS-B may have contributed to label generation. Not airborne camera or ownship-motion truth. |
| **P1** | [ColANet](https://colanet.qa.pdmfc.com/), [research code](https://github.com/dario-pedro/uav-collision-avoidance) | Recorded UAV collision/escape dataset and temporal-model notebooks | Audit whether obstacles are aircraft, annotation timing, collision ground truth, licensing and split leakage before declaring air-to-air relevance. |
| **P1 (modality/ego-motion)** | [TU Delft ODA](https://github.com/tudelft/ODA_Dataset), [data DOI](https://doi.org/10.4121/14214236.v1) | RGB, IMU, event camera, radar, OptiTrack ground truth; 1369 indoor flight trials avoiding poles | Excellent for synchronized camera–IMU stabilization and obstacle avoidance; **not airborne intruder aircraft**. Repository warns some trials lack RGB and some have missing obstacle coordinates. |
| **P2 (synthetic self-state)** | [Mid-Air](https://midair.ulg.ac.be/), [author utilities](https://github.com/montefiore-institute/midair-dataset) | Synthetic RGB, depth, GNSS/IMU and aircraft states | Useful for ego-motion tests and calibration, **not verified target-on-collision-course flight benchmark**. No need to operate simulation to read recordings. |
| **P2 (physics-only)** | [MIT Lincoln Lab Terminal-Area Encounters](https://www.ll.mit.edu/r-d/datasets/unmanned-aircraft-terminal-area-encounters) | Large corpus of synthetic encounter **trajectories, no paired RGB** | Useful to test math of time/CPA and alert semantics with exact trajectories; not a video classifier benchmark. Do not build a simulator. |
| **Reference only** | [Stanford AVOIDDS](https://github.com/sisl/VisionBasedAircraftDAA) | Synthetic aircraft detection still images and simulator code | No temporal aircraft RGB track truth off-the-shelf; do not substitute simulator labels for real RGB flight validation. |
| **Excluded** | Amazon AOT | — | **Never download/train/evaluate on this dataset in this project.** |

**TartanAviation is a noteworthy additional lead** beyond the first set of papers: the combination of public videos, real aircraft tracks, aligned ADS-B range and author code lets us measure *range-change* and angular-motion accuracy from RGB more credibly than QUT alone, although 3-D relative CPA to a moving ownship remains unavailable.

## 5. Exact method-reproduction contract (rather than more bespoke variants)

Freeze detector checkpoint, neural input geometry, confidence/NMS, observations and timestamps. Use the existing `AirAntiCollision` inference cache and provenance/manifest code. Every estimator implements a **stable** observation API:

```text
FrameObservation:
  source_id; source_frame_id; timestamp_s; original_resolution
  camera_mode: fixed | moving | unknown
  K / distortion / IMU / ownship state: optional, provenance-tagged
  camera_transform + inliers + residual + motion_model + valid_time_interval
  targets[]:
    {track_id, class_id, measured_bbox_xyxy, measured_bbox_area,
     detection_confidence, measured: bool, measurement_time_s,
     source: detector | optical | tracker_prediction, association_ambiguity}
EstimatorOutput (each method, each track, each time):
  state: potential_conflict | apparent_nonconflict | uncertain | unobserved
  physical_claim: none | camera_plane_intercept | model_CPA | metric_CPA
  tau_s; proxy_miss_spans; metric_cpa_m; LOS_rate; growth; signal_validity
  evidence_age_s; method_id; assumptions; reason_codes
```

**Core comparisons, in order:**

1. **Implement Bauer 2016/2019 faithfully**, extract exact equations/thresholds including special cases, and unit-test on synthetic *analytical trajectories* (not 3-D simulation). Document all deviations from the papers.
2. **Implement AirTrack-derived looming and angular-state estimate**, explicitly label it a **reconstruction** rather than official checkpoint reproduction. Check tCPA's physical meaning under straight/oblique/rotation encounters and unstable area; benchmark calculation cost independently of AirTrack's detector.
3. **Implement Lai & Lin's published conflict/risk rules** from actual equations, with paper-specific thresholds; if full reproducibility is impossible, label the missing detail and evaluate the nearest documented variant separately (not as identical original method).
4. **Run Opromolla LOS/LOS-rate branch** only when synchronized attitude/navigation exists; treat RGB-only missing-IMU branch as abstention or explicitly separate approximate compensation experiment.
5. **Conditional learned comparator:** ColANet if downloaded data+training labels and reasonable port are verified. Do not claim a fair comparison if it learns from incompatible obstacles/task labels.
6. Add the existing project's `geometry.py` log-growth, `projective.py`, `bounded.py` and `calibrated.py` as **custom control methods**, not independent published baselines.

**Camera compensation A/B:** existing masked KLT+RANSAC similarity/homography vs an independently reproduced/Purdue-inspired flow approach. Estimate residual parallax quality, not just inliers. For pure sky/no background, **do not set compensated LOS-rate to zero**. With actual gyro/attitude, separately evaluate de-rotation, synchronization offset and rolling shutter. Physically unidentifiable conditions trigger `uncertain`. Do not restore `nonthreat` in auto mode merely to improve apparent coverage.

## 6. Falsifiable evaluation and stopping criteria

**Stage A: geometry isolated.** Oracle/manual centroid **and independently measured target extent** (QUT publishes approximate centroids, not reliable box sizes); analytical 3-D straight-line encounters give exact CPA; systematic sweeps cover off-axis bearings, varying intruder dimensions, crossing vs closing, growing/shrinking, missing frames, clipped boxes, rolling shutter-like jitter and shape/aspect changes. Classify counterexamples into **false safe**, **false warning** and **abstain**.

**Stage B: frozen YOLO26m+MOT.** Run identical estimator configurations for each of `YOLO26m → custom`, `→ TrackTrack`, `→ PuTR`, optionally `→ BoT-SORT`, using exactly identical detector observations. Split by entire encounter and source provenance, not random frames. Report both oracle-vs-model and detector-conditioned-vs-end-to-end results.

**Stage C: moving ownship.** Purdue onboard camera; genuine moving-camera videos; test stability and observability, with no fabricated CPA GT. Sensor-assisted alternatives require real calibration/navigation and time alignment. TartanAviation independently evaluates grounded camera range and multi-aircraft geometry.

**Required metrics:**

- Per-encounter warning recall on **explicit encounter-category proxies**; event-level qualifying onset and persistence, observed alert latency, earliest warning vs meaningful supplied trajectory timestamp. Do not call last annotated frame the collision time.
- **False-safe** counts + duration on nominal approaching targets; false-warning counts/duration on labeled nonapproaches; warning episodes per camera hour. Distinguish unmatched tracks from labeled negatives.
- Abstention/untracked time, coverage vs false-safe trade-off, stability under processing at 1/2/5 frame cadence and detector confidence sweep.
- Bearing-rate error, angular/scale drift error, TTC/CPA error *only* when data and camera geometry independently support those labels; calibrated metric CPA only with fully verified state.
- Track identity switches, detector misses, target size bias/clipping, camera validation failures as causal attributions to wrong risk decisions.
- RTX 4060 complete **decode → detector → tracker → method → (optional render)** frame-time p50/p95/p99, warm/cold start separately; backend/torch/cv2 fingerprints, actual source timestamp rates. For Jetson, independently measure rather than extrapolate desktop FPS.
- At least 3 published-algorithm candidates with a single tabular outcome over **exactly matched** data and warnings; method failing for lack of sensors must show precise abstention/input requirement, not fabricated success.

**Reject a candidate** if it causes catastrophic false-safe geometry in valid constructed cases, if claimed CPA truth is a dataset category without trajectories, if results depend on interpolated/predicted (unmeasured) scale, or if gains vanish on reserved encounters. Publish negative findings; no threshold tuning on final holdout. Any aviation guidance remains advisory/research-only until genuine vehicle dynamics, separation geometry and flight safety validation exist.

## 7. Specific gaps in existing AirAntiCollision

| Existing work | Missing independent comparison / action |
|---|---|
| `RESEARCH.md` cites Bauer but picks bespoke image-plane drift/expansion. | No documented faithful full published **Bauer 2016/2019 reproduction** on shared tracks/true CPA test cases. |
| `GEOMETRY.md` documents analytical off-axis and parallax counterexamples. | Keep them as mandatory adversarial regression tests for **every** candidate. |
| `airavoid/geometry.py` abstains on `nonthreat` in auto-camera mode. | This avoids some false clears but **does not solve moving-camera collision discrimination**; benchmark independently stabilized LOS and sensor-assisted alternatives. |
| `MOT_RESULTS.md` thoroughly compares six trackers. | Main threat pipeline still uses bespoke association; compare **threat errors**, not just ID consistency. |
| `RESULTS.md` evaluates 15 QUT sources. | Fixed-ground observer, encounter-category proxy, known overlap risk and untested independent held-out CPA truth. |
| `demo/*.mp4` excluded by `.gitignore`. | Publish representative video/QA artifacts as permitted release assets or external versioned links, not nonexistent repository URLs. |
| `airavoid/appearance.py` may extrapolate SOT boxes. | Keep detector-measured scale separate from predicted geometry to avoid self-manufactured looming. |

## 8. Author code to pin for later cloning

**Recommendation: keep a link/commit manifest now, not mandatory recursive Git submodules.** Scientific repos often contain very heavy files, old dependencies, large Git history, licensing differences and potentially recursive submodules. They are not needed for the first published-method math comparisons; frozen `external/` clones can be placed in isolated environments. Any later submodules should be pinned to audited SHAs, not moving branch tips.

| Candidate path (optional) | Official/research URL | Category / preliminary license status |
|---|---|---|
| `third_party/purdue_uav` | https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking | Moving-camera detection/tracking; **BSD-3-Clause** |
| `third_party/stanford_daa` | https://github.com/sisl/VisionBasedAircraftDAA | Simulator-backed DAA policy + detection; **MIT** |
| `third_party/tracktrack` | https://github.com/kamkyu94/TrackTrack | Association; **MIT reported in surveyed project**; recheck specific commit |
| `third_party/putr` | https://github.com/chongweiliu/PuTR | Learned association; **MIT reported**; recheck |
| `third_party/botsort` | https://github.com/NirAharon/BoT-SORT | Association; license to recheck at pinning |
| `third_party/colanet` | https://github.com/dario-pedro/uav-collision-avoidance | Learned collision benchmark; **license unclear; reference-only** |
| `third_party/tartanaviation` | https://github.com/castacks/TartanAviation | Official video/ADS-B dataset scripts; inspect data-specific licensing |
| `third_party/nasa_daidalus` | https://github.com/nasa/daidalus | Ground-truth state policy; **NASA Open Source Agreement** |
| `third_party/oda_dataset_tools` | https://github.com/tudelft/ODA_Dataset | Optional motion/sensor study; research-data licensing to recheck |
| `third_party/midair_dataset_tools` | https://github.com/montefiore-institute/midair-dataset | Optional synthetic recorded ego-motion dataset tools; license to recheck |

**No verified official code/checkpoint URL found in this survey** for the complete Bauer, AirTrack, Lai & Lin, or Opromolla & Fasano threat-estimation pipelines. That is a research reproducibility gap, not a claim the paper is irrelevant. **ViSafe's verified `castacks/visafe` is a *website*, not a ready training/inference library.**

**Optional submodule commands (after the next phase selects exact versions):**

```bash
# Run only in an approved, clean AirAntiCollision working tree.
# A parent repo tracks only gitlink SHA + .gitmodules, not third-party binaries.
git submodule add https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking.git third_party/purdue_uav
git submodule add https://github.com/sisl/VisionBasedAircraftDAA.git third_party/stanford_daa
git submodule add https://github.com/kamkyu94/TrackTrack.git third_party/tracktrack
git submodule add https://github.com/chongweiliu/PuTR.git third_party/putr
# Review exact commit SHAs, license files and .gitmodules, then commit only when needed.
```

## 9. Decision after replication

Do not designate a winner yet. The next milestone is **a reproducible matrix of at least three independent published threat algorithms on identical frozen observations, explicit moving-camera failure analysis, and physically grounded counterexamples**, *before* optimizing interface, neural detector or demo rendering again. A model that correctly abstains on unobservable cases can be better than a higher-coverage model that falsely declares clearance; select operating points on risk/coverage curves with mission-specific costs.

**This document is research and experiment design, not an aviation safety approval.** Source links checked against official publisher/author/repository pages on 2026-10-18; public availability and licenses can change. No external repository was cloned into the main project by publication of this document.