# Autonomous Research-to-Demonstration Mission
## Monocular RGB Airborne Collision-Threat Assessment and Explainable Avoidance Advisory

**Target agent:** GPT-6.1 Sol operating as an autonomous research and software-engineering agent in a Codex-capable environment.  
**Date:** 18 October 2026.  
**Maximum active project budget:** 35% of the user's total weekly Codex subscription usage allowance, measured as additional consumption from the start of this mission. This is usage-based-budget, not a wall-clock time limit. The agent should use the available budget productively for research, implementation, experimentation, and validation
**Execution mode:** Research, implementation, experimentation, and final delivery without user intervention.

## 1. Mission and definition of success

Independently research, design, implement, and validate a **runnable demonstration** of a system that uses **one RGB video stream** to distinguish airborne objects that may threaten an ownship's trajectory from objects that appear non-threatening. The system must also make its decisions understandable to a non-programmer through **actual annotated test videos**, and illustrate **candidate collision-avoidance directions with explicit assumptions and uncertainty**.

This is a **research prototype and offline demonstrator**, not certified aircraft collision-avoidance equipment. No deployment to an aircraft or live flight control is requested.

Success requires actual executable code, actual input videos, actual generated visual outputs, and honest evidence of what worked. **A literature review, design document, generated code alone, logs alone, or a plausible-looking but unvalidated animation does not count as completion.** Scientific novelty is welcome but secondary to a functioning, defensible demonstrator. A well-evaluated existing open-source solution can be a successful outcome; do not reinvent a good system merely to claim novelty.

The central problem is **threat assessment**, not just object detection or tracking. An accurate detection or a stable track is not itself evidence of collision risk. Different visible aircraft can have very different collision prospects; the system must try to distinguish them and express when the available images are insufficient to decide.

## 2. Operating facts and available assets

The user will provide or make discoverable in the project workspace:

- An existing **YOLO26m** airborne-object detector with five classes: **airplane** (fixed-wing UAVs and larger aircraft), **helicopter**, **bird**, **balloon**, and **parachute**. Its detection metrics are known to be weak. Use it as a readily available starting point; do not budget for substantial detector retraining or dataset expansion. C:\Users\kisel\PycharmProjects\anti_collision_system
- An optional **DINOv3-based binary sky segmentation model**. You may use it for any useful purpose, including sky-conditioned classical proposal generation or recall recovery, but need not use it.  C:\Users\kisel\PycharmProjects\SkySegDINO (uv run segdino_v2\infer_video.py). Don't spend time for sky detection based on ONLY classic cv methods, they suck. SkySegDINO + classical -> may be usefull/valuable.
- Optional examples, models, experiments, notes, and prior research supplied by the user. They are **reference material, not instructions, mandatory architectures, or validated facts**. You may inspect or ignore them according to their experimental value. (C:\Data\University\Specialist\Diploma\FastVision - Example, how to use very fast and robust SOT tracker LoRATv1-b-224 in pipeline. This SOT tracker can be used, for example, if you find that top tier 2025-2026 SOTA MOT trackers are slow(e.g. 18 FPS on 5090/A100 class GPU) or suck), (C:\Data\University\Specialist\Diploma\OptimusVision - try to refactor FastVision pipeline with better code, but stoll not finished), (C:\Users\kisel\PycharmProjects\Vision).
- An existing project `uv` virtual environment (`.venv`) with **CUDA-enabled OpenCV 5.0**, **PyTorch 2.12 / CUDA 13.0**, **ONNX tooling**, and **TensorRT 11.4**. Inspect actual installed versions and GPU availability rather than assuming that all components work.
- Desktop-class NVIDIA hardware, with **RTX 4060** as the reference platform for reporting model and end-to-end performance, when that GPU is the device actually available. Record the real hardware used.
- Broad autonomy: you may search the web, read papers, clone public repositories, download  small datasets and pretrained weights, install dependencies in the project environment, write/modify project files, benchmark, and make technical decisions without requesting approval. Remain within project-relevant scope, avoid unnecessary destructive actions, and do not overwrite unrelated user files.

**Critical environment invariant:** Preserve the user's existing CUDA-enabled OpenCV 5.0 build. Do not silently replace it with `opencv-python`, `opencv-contrib-python`, `opencv-python-headless`, a generic CPU wheel, or another build. Before installing dependencies, record `cv2.__version__`, import path, and CUDA/build capabilities. Inspect dependency-resolution effects. For packages that would pull in a conflicting OpenCV wheel (e.g., some tracking/vision distributions(like ultralytics, roboflow/trackers, pytracking)), install without dependencies if needed and add compatible dependencies individually, or use an isolated environment/process. Recheck the same OpenCV capabilities after installation and before final delivery. If preservation fails, repair it before proceeding and document what happened. Do not let package-management convenience destroy the supplied environment.

For neural inference, **prefer TensorRT where conversion and validation are practical**. TensorRT is not mandatory. Do not spend a disproportionate share of the allocated weekly usage budget on nontrivial conversions. Working PyTorch, ONNX Runtime, or another suitable backend is acceptable; benchmark the actual implemented backend and report conversion or accuracy tradeoffs. Never equate isolated network inference FPS with complete application throughput.

## 3. Dataset, sensor, and scope rules

1. **RGB video is mandatory** for the primary demonstration and evaluation. Do not use Amazon's **AOT / Airborne Object Tracking dataset**.
2. Real-world video is strongly preferred. Suitable existing synthetic **RGB datasets created by others** may be evaluated as additional evidence, clearly labeled as synthetic. **Do not build, operate, or depend upon a 3D game engine or flight simulator** for this assignment. Small analytical or 2D unit tests of kinematics are allowed, but they cannot substitute for evaluation on actual video.
3. The default problem is **monocular video with no known camera intrinsics, field of view, IMU, GPS, ownship velocity, or target telemetry**—for example, video retrieved from YouTube. Do not fabricate these quantities or secretly use annotation metadata unavailable at inference.
4. When a dataset **genuinely provides synchronized RGB plus intrinsics, IMU, GNSS, aircraft poses, relative range, or trajectories**, you may exploit those signals in a clearly labeled **sensor-assisted variant**. Preserve a **camera-only evaluation path** and compare the variants where feasible. Do not describe sensor-assisted results as camera-only.
5. Search for datasets that support the **actual threat/non-threat discrimination problem**: airborne encounters, multiple aircraft, camera motion, range/trajectory/closest-approach information or credible encounter labels. Critically check label semantics. An object annotated as a “collision-course target” is not necessarily proof of a physical collision. A detection dataset without flight geometry is not automatically a threat-assessment benchmark.
6. Distinguish stationary-camera versus airborne-camera video, potential encounter versus verified collision, and ground-truth geometric CPA versus approximate or manually inferred labels. Do not merge these into one misleading score.
Available local datasets:
Airplane(fixed wing) class: 
F:\UAV_DATASETS\jasminmartin(Aircraft Collision Course Dataset)
F:\UAV_DATASETS\AIR2LAND(Synt)
F:\UAV_DATASETS\auto_annotate\Videos\Tracking  (CVAT for Video format 1.1)
X:\DATASETS\UAV\TRACKING\PreparedTracking_NPS_verified_OK_v1_jpeg95\NPS

Helicopter and fixed wing class (SOT, not MOT):
X:\DATASETS\UAV\TRACKING\RGB\UAV-Anti-UAV
## 4. Independent research: first stage, then engineering

**Research first, but time-box it and move to working code.** Conduct an independent, evidence-based search as of 18 October 2026 for approaches, open-source projects, datasets, and test protocols relevant to **monocular airborne encounter assessment and avoidance**. Search scientific literature, research institutions, relevant code repositories, public dataset indexes, and credible technical documentation. You may investigate both classical geometry and modern learned approaches, including hybrids and solutions not previously mentioned in the user conversation.

Prioritize questions that decide the implementation:

- How can a single camera determine that a tracked object is on a potentially dangerous relative trajectory rather than merely approaching or crossing the image?
- What is genuinely **observable**, and under which assumptions (camera motion, perspective, target aspect changes, scale, target range, flight dynamics, missing calibration)? Where must the algorithm abstain?
- Which already-implemented systems or repositories could be reused with minimal effort, and how strong is the supporting evidence?
- Which available RGB datasets and videos contain meaningful positive **and negative** encounter cases, multiple targets, usable annotations, and/or independently verifiable geometry?
- What limited additions to detection, tracking, motion estimation, or geometry would most improve practical threat discrimination given weak detections and scarce training data?
- When, if ever, can avoidance-direction guidance be justified from the available evidence, and how should uncertainty and other targets be handled?

Do not treat any earlier assistant research memo, example repository, particular dataset, paper, formula, or named algorithm as the prescribed solution. Independently verify leads and feel free to overturn them. Favor **early executable comparisons** over speculative novelty. Decide and document which candidate ideas were accepted, rejected, or not attempted, and why.

You can download the researchers' articles to a local folder so that you can reread them later without additional online search.

Write a concise, source-linked research note with enough technical specificity for later auditing. Clearly separate **published claims**, **verified repository behavior**, **your own measurements**, and **hypotheses**.

## 5. Build the demonstrator autonomously

Choose the architecture after inspecting the environment and research evidence. You have broad freedom over classical vision, multi-object tracking, temporal fusion, learned motion or risk estimation, camera-motion compensation, geometric modeling, optional sky segmentation, uncertainty quantification, and decision policies. Avoid training a large new detector; small calibration, fitting, or short optimization experiments are allowed if valuable.

The working system should, to the extent supported by the data:

- Ingest an ordinary RGB video file, preserve timestamps/frame rate, and process multiple airborne-object candidates with stable identities where feasible.
- Use the supplied detector when available and integrate appropriate tracking/temporal logic. Compare or replace individual components if measurements warrant it. Optional extra proposals, classical methods, or pretrained networks are permitted when their cost and failure modes are justified.
- Produce a **per-object, time-varying threat assessment** grounded in estimated relative motion or another defensible observable, rather than class identity, bounding-box position, detector confidence, or a central-image “danger zone” alone.
- Distinguish at least **potential threat**, **apparently non-threatening**, and **undetermined/insufficient evidence**. These must be meaningful states, not arbitrary confidence thresholds presented as guaranteed physical safety. A balloon, bird, or parachute must not automatically be declared harmless solely because of its class.
- Explain how camera field of view or image-space threat sectors were chosen, estimated, or assumed, and how conclusions change when calibration is missing or wrong. If a hazard region is drawn, it must have documented meaning and limitations, not merely an aesthetically appealing overlay. You decide how to display the overlay interface correctly. You can make at least a few windows in the video to display graphs and get into the truncated pyramid of the camera's field of view (frustum) or the threat area.
- Provide temporal behavior: accumulating evidence, handling intermittent detections, unstable object size, camera rotation, crossings, identity switches, disappearance/reappearance, and multiple simultaneous targets to the extent practical.
- Include an **illustrative avoidance advisory**: show how candidate lateral and/or vertical escape directions would be compared, what relative-motion and ownship maneuverability assumptions were used, and why the chosen option is preferable—or explicitly say **“insufficient information to recommend a direction.”** A single monocular video frequently cannot identify a uniquely safe maneuver. Never turn an unsupported visual guess into an asserted safe flight command. Advisory is visualization-only, with no actuator or autopilot outputs.

Implement the **smallest credible end-to-end version early**, then improve it experimentally. A robust partial system with honest uncertainty and clear visual evidence is more valuable than an elaborate untested stack.

## 6. Demonstrate decisions visually, not through logs

**Annotated video is a primary product deliverable.** Produce real MP4/H.264 (or another widely playable format) outputs from actual test clips, not only screenshots, JSON, console output, or mockups. The result must be readable by a reviewer who has **not read the code**.

Use clear visual conventions, and make their semantics visible in the video. Where supported, show:

- Per-target bounding regions or markers, track ID, detected class, and current state (**threat / apparently non-threatening / uncertain**).
- Target motion history and relevant predicted image-space trajectory or closest-approach visualization, with uncertainty or validity indication; avoid fabricating physically metric predictions from uncalibrated pixels.
- The observable evidence that caused the classification, such as **relative bearing behavior**, angular drift, apparent expansion, predicted miss/encounter geometry, estimated time-to-event, persistence, or a different justified algorithm-specific quantity. Label **measured, estimated, assumed, and unavailable** information distinctly.
- Any field-of-view estimate, camera-motion estimate, geometric danger envelope, encounter region, or risk sector actually used by the algorithm, with a small legend explaining how it was obtained.
- A compact on-video **“WHY?”** explanation for the selected threat and at least one non-threat or ambiguous target. Show changes in the decision as new frames arrive. The explanation must reflect **the algorithm's actual internal outputs**, not text invented during video rendering.
- An ownship-relative **avoidance options panel/overlay** comparing available candidate directions (e.g., left/right/up/down when meaningful), assumptions, tradeoffs, and the recommended illustrative option or an explicit abstention. Show if changing course might worsen the encounter with another visible target. Do not draw a confident safety corridor without sufficient evidence.
- A clear distinction between **live model estimates** and **offline ground-truth annotations** if GT is presented for educational comparison. GT must not influence the model's online decisions in camera-only mode.

Use readable typography and uncluttered composition: place detailed diagnostics in a side panel when appropriate, retain an unobstructed view of the actual targets, and use consistent colors and legends. Prefer a **short, high-information set of annotated clips** over hours of repetitive footage. At least one clip should reveal the system making **different decisions for different simultaneously visible objects** if the available source data supports this; otherwise document the unmet condition.

Also provide a one-page **visual results overview** (`demo/README.md`, HTML gallery, or similar) linking to the output videos and preview frames. It should tell the reviewer **what to watch, which objects were judged threatening, why, whether direction advice was warranted, and what is uncertain**. Avoid presenting a cherry-picked success clip as comprehensive validation: include a failure or difficult case if one occurred. (For results overview and UI annotation use primary Russian language)

## 7. Verify claims and benchmark performance

Do not evaluate only visually. Set up a reproducible evaluation on the best available public RGB encounters, supplementing with clearly distinguished camera-only, sensor-assisted, synthetic, and qualitative-video trials as warranted.

At minimum, establish evidence for:

1. **Geometry/threat logic in isolation**, using ground-truth tracks or other appropriate controlled inputs when available, so detector errors do not hide flaws in the threat decision.
2. **End-to-end behavior** with the supplied detector and the chosen tracker(s) on real RGB video, including the effect of false detections, missed targets, and track fragmentation.
3. **Positive and negative cases**. Measure both **missed threats/false-safe decisions** and **false warnings**; when reliable event labels or geometry are available report suitable event-level recall, false-alert rate, time-to-alert, and (only when meaningful) time-to-closest-approach or miss-distance error. Include class/sequence counts, denominators, and abstentions. If data lack true collision labels, do **not** manufacture ground-truth scores; report proxy or qualitative outcomes explicitly.
4. **Baseline comparison or ablation** where possible: at least one simpler alternative and the selected approach evaluated under comparable inputs. Record what changed and whether gains are substantive.
5. **Stress/failure analysis** covering relevant cases such as stationary camera vs moving camera, low-pixel-size targets, apparent scale changes from aspect, camera turns, featureless sky, crossing aircraft, multiple targets, missed detections, and ID switches. Use targeted tests or existing examples, and label tests that remain unperformed.
6. **Real local runtime measurements**, including detector, tracker, threat evaluator, any optional segmentation/motion stage, render/export overhead, and actual end-to-end throughput (plus resolution, batch, precision, hardware, and backend). Favor TensorRT when straightforward; use other working backends when it is not. Distinguish offline processing FPS from deployable streaming latency. Do not invent FPS.
7. **End-to-end visual checks**: open or inspect the rendered output frames, confirm target alignment, labels, legends, timing, codec playback, consistency with decisions, and any avoidance-panel explanations. A renderer that simply exits successfully is not sufficient validation.

Use honest train/validation/test separation where learning, fitting, or thresholds are involved. Avoid using test ground truth to tune and then claim unbiased performance on the same sequences. Keep source footage and processed derivatives traceable. Ensure results are repeatable with fixed configurations and commands.

## 8. Autonomous execution contract - weekly usage budget

Operate under a budget of the user's total weekly Codex subscription allowance. Do not impose an arbitrary wall-clock deadline. Track subscription usage through reliable usage indicators whenever accessible, recording the starting usage and subsequent consumption. Do not fabricate usage measurements or treat elapsed time as an accurate proxy for subscription consumption.

Continue productive research, implementation, experimentation, and validation throughout the allocated budget. Do not stop merely because the initial prototype works or the minimum acceptance criteria have been satisfied. Use the remaining budget to investigate failure cases, improve algorithms, compare alternative hypotheses, perform ablations, evaluate generalization, and refine the final demonstration.

The objective is to maximize the quality and scientific validity of the finished system within the allocated usage budget, not to minimize the number of agent iterations.

Choose your own workflow and adjust dynamically. Indicative **guardrails**, not fixed phase allocations:

- **Early:** Inspect assets/environment; independently research and select a high-value dataset and promising runnable baseline. Secure at least one video that can be processed. Record the starting timestamp.
- **Before over-investing in research:** Produce a minimal executable video-to-annotated-video pipeline, even with a provisional estimator. Never spend the whole session reading papers and leave no demo.
- **Middle:** Implement and compare the best hypotheses, evaluate geometry separately from detection/tracking, test the threat and uncertainty decisions, and render representative videos.
- **Late:** Revisit failure cases; validate a reasonable avoidance-advisory visualization; benchmark; recheck the environment; produce a polished reproducible demo and final evidence report.
- **Final ~10–15% of the allocated usage budget:** Freeze risky refactors, run smoke tests, verify that video files play and outputs are real, complete documentation and handoff.

Write durable state in small project files (e.g., `MISSION_STATUS.md`, `RESEARCH.md`, `EXPERIMENTS.md`) that record observed subscription usage when available, elapsed time, hypotheses tried, actual evidence, current blockers, next experiment, and the exact commands required to resume. **Update at each meaningful milestone and at least approximately hourly**, without spamming massive tool transcripts. On any context reset or resumed session, read these files and continue the experiment; do not replan from scratch.

**Do not ask the user clarifying questions, seek approvals for ordinary project work, or pause for a response.** Inspect the workspace; infer sensible defaults; state important assumptions in writing; continue. If one asset, download, package, method, or dataset is unavailable, try a reasonable fallback without blocking the entire project. Handle download size and disk/time constraints pragmatically: use valid subsets when full archives are too costly, and report selection and limitations.

**Do not claim that the requested subscription budget has been consumed unless reliable usage measurements support that claim.** Timers and TODO files do not independently keep an agent executing after its session ends. Use persistent Codex Goals or equivalent long-horizon execution mechanisms when supported. Continue meaningful work until the allocated usage budget is reached, the execution environment prevents further progress, or no defensible additional experiments remain. If subscription usage is not observable, explicitly report this limitation rather than inventing percentages. The final report must include actual execution timestamps, completed experiments, and measured usage where available.

Choose experiments by expected information gain, time cost, likelihood of implementation success, and relevance to the central discrimination task. Avoid repeated unproductive downloads, interminable model conversions, speculative training, and major UI work that displaces algorithm validation.

## 9. Required final artifacts

Use a clear, self-contained project structure suitable for future iteration. Adapt to the actual repository, but ensure an evaluator can find all of the following:

- **Runnable implementation** and an obvious entry point for applying it to an arbitrary RGB MP4. A CLI is sufficient; a polished interactive app is optional.
- **Environment instructions and dependency protection** (`README.md` or equivalent), including how to run with the supplied `.venv` without overwriting CUDA OpenCV; note externally provided model-weight paths and licenses.
- **Research report** with selected papers/datasets/repos, source links, observed versus published results, and explicit rejected/unverified leads.
- **Experiments and metrics** with actual sample counts, evaluation definitions, baseline comparison, performance numbers, hardware/runtime settings, and known limitations.
- **Annotated video outputs**, preferably several varied real-video cases, plus preview images and a short human-readable visual guide. Keep an easy-to-run demo path.
- **Actual model decision trace** in a structured machine-readable format if useful (CSV/JSONL/Parquet), so annotations and reported metrics can be regenerated. Do not make the reviewer read raw traces to understand the demo.
- **Avoidance advisory explanation** that states how the options were evaluated, which assumptions are necessary, what was and was not observable, and why abstention may be the only defensible result.
- **Test/smoke-test commands and observed outcomes**, including confirmation that the delivered video files exist, are playable, and illustrate the intended behavior.
- **Final summary** in plain English: what's implemented, most convincing examples, most serious failures, confidence of claims, scientific novelty (if any, carefully qualified), and the next highest-value improvement.

The final user-facing response should **lead with links/paths to playable annotated videos and a concise visual result summary**. Put technical documentation second. Provide commands and logs only as supporting evidence.

## 10. Acceptance and honesty conditions

Treat these as non-negotiable completion checks:

- The output performs **per-object collision-threat assessment**, not merely detection, tracking, or arbitrary near-center highlighting.
- It runs on at least one **real RGB encounter video**; report exactly which footage was processed and what the test can actually establish.
- Its **on-video reasoning** comes from computed internal variables and is understandable without source code. The choice or refusal of an avoidance direction is visibly justified, not decorative.
- It includes meaningful **negative or uncertain examples**, not only hand-picked positives.
- It has at least one measured evaluation or carefully justified proxy test, plus actual end-to-end throughput on available hardware; unsupported metrics are explicitly marked unavailable.
- The existing **CUDA OpenCV 5.0 installation is preserved and checked**.
- It does not present unknown field of view, distance, ownship speed, GPS, target dimensions, risk probabilities, physical miss distances, or “safe escape routes” as measured facts.
- It makes clear that this is an **offline research demonstrator, not a flight-certified or flight-ready safety system**.

If full success is impossible within the allocated subscription usage budget, deliver the **best working and verified partial system** with all produced videos, tests, blocker evidence, and a precise unfinished-items list. **Never replace missing experiments with confident claims.**

**Begin execution now. Conduct research, implement, run tests, generate and inspect actual visual artifacts, iterate, and deliver the evidence-backed result without waiting for the user.**
