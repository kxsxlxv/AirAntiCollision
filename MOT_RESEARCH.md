# Online MOT with fixed YOLO26m: research and experiment protocol

Research date: 18 October 2026. Local results are recorded separately in
`reports/mot_*` and the final comparison. Published benchmark scores below are
context, not predictions of aircraft accuracy or RTX 4060 throughput.

## Candidate selection

| Model | Date | Learned component | Fixed YOLO boxes | Assessment for this project |
|---|---|---|---|---|
| BoT-SORT / BoT-SORT-ReID | 2022 | Optional appearance embedding; Kalman motion and heuristic association | Yes | Important moving-camera baseline; camera compensation does not recover metric ego-motion |
| OC-SORT | 2022 preprint / CVPR 2023 | No learned association or motion; observation-centric Kalman correction | Yes | Fast motion-only control, especially useful when tiny crops lack discriminative appearance |
| Deep OC-SORT | 2023 | Adaptive appearance embeddings | Yes | Strong additional hybrid candidate; pedestrian ReID transfer to aircraft remains unverified |
| GHOST | CVPR 2023 | Appearance network with domain adaptation; simple association/motion | Yes | Useful generalization baseline, but not a learned motion predictor |
| TWiX / C-TWiX | 2024–2025 | Transformer association using only box coordinates | Yes | Particularly relevant when tiny targets have indistinguishable texture; published speed excludes our detector and hardware |
| BUSCA | ECCV 2024 | Transformer reasoning to continue missed detections | Yes, as a tracker add-on | Relevant to detector gaps; continued/predicted boxes must not become measured collision-scale evidence |
| DiffMOT | CVPR 2024 | Diffusion motion predictor conditioned on box history; appearance network | Yes | Direct candidate for nonlinear motion; published detector-inclusive timings use RTX 3090, not 4060 |
| PuTR | 2024 preprint / revised 2025 | Transformer association over appearance patches, box positions, and history | Yes | Cleanest released learned-association candidate among the requested six; no replacement detector required |
| TrackSSM | 2024 | Mamba encoder and cascaded motion decoder | Yes | Relevant motion research baseline; released code needs portability and output-interface repairs, detailed below |
| MASA-R50 | CVPR 2024 | General-purpose instance appearance/association representation | Yes, independent R50 feature extractor | Particularly relevant alternative to pedestrian-specific ReID; 4060 end-to-end performance not established here |
| TrackTrack | CVPR 2025 | FastReID appearance; hand-designed iterative association and Kalman motion | Yes | Strong practical identity-preservation candidate; online core must be separated from optional offline linking/interpolation |
| MM-Tracker | AAAI 2025 | Motion Mamba incorporating local/global motion; detector motion-margin loss | Adaptation needed | UAV-platform results concern mainly ground objects; reported full-system gains include detector training, unavailable under frozen YOLO constraint |
| HyperSSM / HyperMOT | CVPR 2026 | Hypergraph plus SSM collaborative motion prediction | Architecture permits external boxes | Promising recent learned-motion candidate; official repository has code but no documented released checkpoint/download recipe at inspection |
| MASA-OA | 2026 | MASA embeddings plus optimized, occlusion-aware association and Kalman motion | Yes | Relevant newer association candidate; published improvements do not establish small-aircraft performance or local speed |
| MOTIP / MOTIP2 | CVPR 2025 / October 2026 | Learned ID prediction | Not a released plug-in for arbitrary YOLO boxes | Uses detector-produced object embeddings; replacing those with YOLO/crop features requires adaptation and association training |

PuTR, DiffMOT and TrackSSM represent different mechanisms. A higher pedestrian
benchmark score does not identify the best motion predictor or the best tracker
for several visually indistinguishable 5–15 pixel aircraft. Sparse sky texture
also weakens background camera-motion estimation. These are transfer hypotheses
to test, not established advantages of neural methods.

For historical context, official results report MOT17 HOTA/IDF1 of 67.1/83.1 for
TrackTrack, 64.5/79.3 for DiffMOT, 62.1/75.6 for PuTR, and 61.4/74.1 for TrackSSM.
HyperSSM reports 66.9/83.1. These systems use different detectors, training,
protocols and sometimes post-processing; the numbers are not a controlled ranking.
MOTIP2 reports strong DanceTrack results but remains tied to its DEIM detector
features. FC-Track's published 5.7 FPS on MOT17 and 0.6 FPS on MOT20 do not provide
evidence for our 10 FPS requirement, despite its identity-focused design.

## Controlled local evaluation

The six requested methods are TrackTrack, BoT-SORT (interpreting “BoT-STOR” as
BoT-SORT), OC-SORT, DiffMOT, PuTR and TrackSSM. The existing tracker is a seventh
control. No detector weights are changed or trained. The existing TensorRT
YOLO26m engine, 0.03 confidence threshold, full view plus native overlapping
tiles, class-agnostic suppression and 2-pixel internal tile-boundary guard remain
fixed. All online updates consume only current/past frames.

1. Six QUT sequences: M1, M2, M6, M7, S1, S2, last 875 frames each, 1280×720,
   25 Hz. Identical saved actual-YOLO detections isolate association changes.
   Labels are approximate centres with incomplete distractor coverage. Report
   centre-gated coverage, consecutive identity changes, changes including
   reacquisition, and dominant-identity coverage at 5/10/20 pixels. Do not call
   these scores standard IDF1/HOTA or physical collision accuracy.
2. Three real public UAVSwarm sequences, selected before tracker results:
   02 (first test sequence), 06 and 28 (paper-illustrated tracking difficulties).
   All 363 original JPEG frames, 4,616 labelled boxes and persistent IDs are used.
   Inputs are downloaded from a pinned author-repository commit and verified
   against Git blob SHA plus local SHA256. Standard TrackEval HOTA, Identity and
   CLEAR metrics use full box annotations. This is a small diagnostic subset,
   not the full benchmark or a proven detector-training-independent test set.
3. Actual detector-inclusive runs on QUT M2 at 1280×720 and the native 1920×1080
   moving-camera formation video. Count processed updates, disclose frame
   sampling, and include actual decoding, YOLO inference and tracker update.
   Cached detections are never used to claim end-to-end detector FPS.
4. Offline motion-only diagnostic: identical five-observation YOLO histories,
   with identity supplied by labels for sample extraction only, predict the next
   approximate QUT centre. Compare learned predictors with holding position and
   a linear fit. This isolates prediction transfer; it does not measure online
   association and is not a future 3D trajectory/CPA evaluation.

All neural checkpoint inference runs on the RTX 4060. Dependencies are installed
under `external/mot/python_deps`, not into the production environment. The custom
CUDA OpenCV 5 installation is preserved. No AOT/torch.compile or 3D simulator is
used. GPU measurements run sequentially to avoid competing inference processes.

## Explicit implementation differences

- Common high/new-track confidence is 0.03 to match the existing detector's small
  aircraft operating point. Track lifetime is approximately 0.8 seconds. These
  are common adaptation choices, not published benchmark hyperparameters.
- TrackTrack receives identical post-NMS detections in both of its detection
  inputs. Its optional recovery of separately retained suppressed proposals is
  unavailable under this fixed-output contract. CMC files are replaced with
  online masked KLT/RANSAC background similarity; optional AFLink and Gaussian
  interpolation are excluded because they revise past output.
- TrackTrack, BoT-SORT, DiffMOT and TrackSSM share the published DanceTrack
  FastReID SBS-S50 network and RGB crop preprocessing. It is not aircraft-trained.
  Classifier size is read from the checkpoint; inference parameters are checked
  for missing/unexpected keys. PuTR uses its own published DanceTrack weights
  and original resized RGB patch sampling. OC-SORT has no appearance network.
- BoT-SORT's original sparse-flow CMC receives guards for absent sky features
  and failed RANSAC, plus an explicit OpenCV `method=` argument. Other algorithm
  code remains upstream. Legacy NumPy aliases are supplied process-locally.
- Torch's cuDNN is initialized before custom OpenCV adjusts DLL search paths;
  otherwise neural convolution fails with a cuDNN sublibrary-version mismatch.
  This changes no installed library or global environment setting.
- TrackSSM uses the upstream Mamba 1.2.0 reference selective scan and equivalent
  eager PyTorch RMS normalization on Windows. No random/untrained weights are
  substituted. All predictor checkpoint keys load strictly. Its released decoder
  returns absolute boxes shaped `[B,4]`, while its tracker calls `.mean(0)` then
  adds the last box as though the output were delta samples `[S,B,4]`. Our adapter
  returns `(absolute-last)[None,...]`, preserving individual target predictions.
  Results must be labelled **TrackSSM, repaired interface, reference backend**,
  not an exact reproduction of the paper's optimized implementation/FPS.
- PuTR online output includes only currently matched confirmed tracks; its
  submission script's retrospective addition of tentative historical rows is
  excluded. No future-frame linking/interpolation is used for any method.

Tracking predictions remain separate from actual detector measurements. A tracker
can preserve an identity through a detection gap without supplying valid box-size
evidence for collision geometry. Integrating a winner into risk assessment requires
that distinction and a renewed threat-decision regression test.

## Primary sources

- [BoT-SORT authors' code](https://github.com/NirAharon/BoT-SORT)
- [OC-SORT authors' code](https://github.com/noahcao/OC_SORT)
- [Deep OC-SORT authors' code](https://github.com/GerardMaggiolino/Deep-OC-SORT)
- [GHOST authors' code](https://github.com/dvl-tum/GHOST)
- [TWiX authors' code, coordinate-only association and detector-separated timings](https://github.com/Guepardow/TWiX)
- [BUSCA paper](https://arxiv.org/abs/2407.10151)
- [DiffMOT authors' code, checkpoints and detector-inclusive RTX 3090 measurements](https://github.com/ShuCvlab/DiffMOT)
- [PuTR authors' code and checkpoints](https://github.com/chongweiliu/PuTR)
- [PuTR revised paper and efficiency scope](https://arxiv.org/html/2405.14119v2)
- [TrackSSM authors' code and checkpoint links](https://github.com/Xavier-Lin/TrackSSM)
- [TrackTrack paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf)
- [TrackTrack authors' code](https://github.com/kamkyu94/TrackTrack)
- [MASA authors' code and independent R50 variant](https://github.com/siyuanliii/masa)
- [MM-Tracker authors' code](https://github.com/YaoMufeng/MMTracker)
- [MM-Tracker paper](https://ojs.aaai.org/index.php/AAAI/article/view/33019)
- [HyperSSM 2026 paper](https://arxiv.org/html/2604.12665v1)
- [HyperMOT authors' repository](https://github.com/SkyeSong38/HyperMOT)
- [MASA-OA authors' repository](https://github.com/gonzalocarreteroh/MASA-OA)
- [MOTIP paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf)
- [MOTIP2, submitted 7 October 2026](https://arxiv.org/abs/2610.10391)
- [FC-Track paper and reported speed](https://arxiv.org/abs/2603.12758)
- [UAVSwarm authors' dataset](https://github.com/UAVSwarm/UAVSwarm-dataset)
- [UAVSwarm paper](https://www.mdpi.com/2072-4292/14/11/2601)

No universal “most accurate” claim is justified by these sources. Selection must
jointly satisfy local identity performance, detection coverage, and measured
runtime at the stated input resolution and sampling rate.
