# Stage 2 methodological gate, 2026-10-22

This extends the committed survey rather than restarting it. No published flight
result validates our implementation. Source PDFs are read-only local references
under ignored `external/stage2`; original equations are visually checked.
Repository revisions/licenses are in `reports/stage2_official_repositories.json`.

| Approach and primary source | Reproduction class; inputs and output | Decision / missing evidence |
|---|---|---|
| [Bauer MED 2016](https://eprints.sztaki.hu/8949/1/Bauer_1168_3096377_ny.pdf) | Faithful equation reproduction possible: corrected disc projection, LS reciprocal extent and normalized position, Eq. 8; calibrated focal length and edge bearings | Implement horizontal disc cues. Original TTC is depth-plane crossing, not general physical CPA. No official executable verified. Paper is IEEE author manuscript; implement equations independently. |
| [Bauer 2019](https://real.mtak.hu/100246/1/Hindawi_2019.pdf) | Equation reproduction possible with prerequisites: Eq. 4–13 rotate to horizontal minimum then interpolate vertical crossing; trajectory-aligned camera, focal length, horizontal/vertical shape correction | Implement identifiable horizontal geometry; label unavailable vertical equal-diameter extension explicitly. Do not replace this disc correction by the old parallel-segment estimator. Flight thresholds 2 s / 20 normalized horizontal units are studied settings, not transferable safety limits. CC BY paper, no verified author code. |
| [Lai–Lin 2024](https://doi.org/10.3390/rs16050756) | Equation reproduction possible: Eq. 2–6 area since first detection, image-centre region, inward motion cosine; weights .5/.333/.167, warning >.5 | Implement exact factors and threshold. Substituting detector rectangle area for Mask R-CNN silhouette is an explicit adaptation. Zero motion makes bearing cosine undefined; do not invent it. No official code/weights verified. CC BY paper. |
| [AirTrack 2023, III-E](https://arxiv.org/html/2209.12849v3) | Reproducible observable: pairwise dt/(sqrt(Aj/Ai)−1); angular rate needs verified degrees/pixel | Implement cue separately. Pairwise formula gives time measured from the earlier sample under inverse-depth growth; subtract dt to express remaining time at j. Not an independent complete threat classifier and not a third independent risk architecture. Learned range head/checkpoints are unavailable; do not substitute size priors. |
| [Opromolla–Fasano 2021](https://www.iris.unina.it/handle/11588/862741) | Incompatible with current RGB-only measurements: ownship navigation aids LOS and LOS-rate stabilization | No navigation/attitude synchronization found yet. Institutional record has no full manuscript attached. Do not invent exact thresholds from its abstract. No official executable verified. |
| [NASA DAIDALUS WCV_TAUMOD](https://github.com/nasa/daidalus) | Reusable official source plus independent equation implementation possible; metric relative position/velocity, horizontal modified tau + DCPA and vertical TCOA | Distinct state-based comparator in analytical/verified-telemetry tier only. Normal RGB traces emit missing-state, never inferred metric input. Source pinned, NASA NOSA reviewed separately; do not vendor library. Not a third monocular result. |
| [SISL AVOIDDS](https://github.com/sisl/VisionBasedAircraftDAA) | Official MIT code, pinned; detector/simulator package and policies require encounter state | Inspect source; no X-Plane/simulator operation. Incompatible as a ready box-only threat algorithm. Detector weights are not our fixed detector. |
| [ColANet](https://github.com/dario-pedro/uav-collision-avoidance) | Incomplete deployable comparator: dataset/training and escape-vector notebooks | No top-level license or verified published inference checkpoint established. Do not copy notebook code or claim a pretrained run. Task/domain/split calibration needed. |
| [ViSafe 2025](https://theairlab.org/visafe/) / [public repo](https://github.com/castacks/visafe) | Incomplete specification for local reproduction: public repository is project website/assets, BSD-3-Clause | No inference release established. Multi-view/state/control prerequisites differ. A real-flight claim is not a callable monocular-box classifier. |
| [Purdue UAV-to-UAV](https://github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking) | Reusable official BSD-3-Clause detection/motion/tracking source, pinned | Perception comparator, not conflict estimator. Do not redownload NPS (= Purdue). Compare background motion models without adding GT. |
| [TartanAviation 2025](https://doi.org/10.1038/s41597-025-04775-6) | Recorded stationary RGB/ADS-B dataset, not collision algorithm | Calibration/timing/ADS-B provenance required before metric errors. Optional data lead; do not consume budget on another corpus while NPS is available. |
| [TrackTrack](https://github.com/kamkyu94/TrackTrack) / [PuTR](https://github.com/chongweiliu/PuTR) | Existing isolated, pinned online association adapters/checkpoints from Stage 1 | Reuse for causal observation continuity only. No GT input and no tracker-predicted scale treated as measurement. No new MOT survey. |

Five direct conflict/avoidance families are Bauer, Lai–Lin, Opromolla–Fasano,
DAIDALUS, ColANet, plus ViSafe; AirTrack is an observable-cue source.

## Selection and falsifiable plan

First compare Bauer disc geometry, Lai–Lin risk factors, AirTrack cues, and the
four existing custom estimators plus a transparent constant-bearing growth rule.
Add NASA modified-tau well-clear as a genuinely distinct state-based architecture,
explicitly confined to analytical/verified-state data. Three independent RGB
threat classifiers are **not yet demonstrated**; AirTrack does not count as one.
Investigate another reproducible monocular architecture if the initial comparison
does not supply three; report unmet scope if source/prerequisites remain blocked.

Freeze video-level development 001/008/032 and evaluation 020/034/043 before any
threshold tuning. This is a focused NPS subset, not a full-dataset leaderboard.
Use complete selected clips, canonical user-verified revision, actual source PTS
where recoverable, and shared frame/cadence/IoU contracts. Prepared JPEG nominal
FPS alone does not prove original acquisition timestamps. Both detectors have
user-confirmed NPS training independence (2026-10-22); it is not an independent
training-manifest audit. A/B compare fixed-wing only. Keep A five-class unchanged.

Execute the checked dataset/model inventory first (`scripts/stage2_audit.py`),
freeze observation caches with hashes, then run the same method list for each
detector/tracker separately. GT only joins outputs in offline scoring or explicitly
named oracle traces. Compare original/TrackTrack/PuTR association, predicted vs
oracle identities, measured extents only, raw vs similarity vs independent
homography camera diagnostics. Homography success is not rotation recovery.

Analytical projected trajectories use independent vector CPA truth, no simulator.
Sweep off-axis bearing, relative velocity, noise, cadence, missing/clipped boxes,
shape change and parallax. Separate valid-model failures from assumption
violations. Do not tune away false-safe examples or count uncertain as correct.
Real NPS has no physical threat truth: report continuity, scale, evidence and
warning/abstention coverage; no collision recall. QUT approach/other is a separate
development-influenced proxy, checked at strides 1/2/5 with frozen existing caches.

Only after comparison integrate selectable backends and a separate P2 adapter,
retaining existing defaults. Measure actual RTX 4060 decode→detector→association
→assessment (with/without encoding), initialization and latency quantiles. Jetson
is unmeasured absent actual hardware. Export commands must recreate ignored video
assets. Commit each checkpoint; open PR with evidence and remaining limitations.

Planning allocation within 30 percentage points: about 6 for source/protocol/math,
18 for controlled real-data comparisons, 6 for integration/tests/delivery. These
are targets, not subcaps; account-wide usage checkpoints govern the hard stop.
