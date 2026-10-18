# Delivery: verified offline RGB encounter prototype

Start with the playable [M2 demonstration](demo/M2_final/annotated.mp4),
[M6 failure case](demo/M6_failure/annotated.mp4), and
[Russian video gallery](demo/index.html). The [Russian guide](demo/README.md)
explains what to watch, computed reasons, uncertainty and maneuver abstention.

The implementation runs on an ordinary monocular RGB MP4 using the supplied
five-class detector and TensorRT. It estimates per-target bearing drift and apparent
expansion, accumulates evidence, and distinguishes potential warning, conditional
image-model pass and insufficient evidence. Optional native crops, neural zoom,
DINO sky selection and LoRAT bridging are measured separately. No detector training,
AOT or 3D simulator was used.

All 15 complete QUT sources were processed at the default every-frame cadence:
55,535 timeline frames / 37.02 minutes. Corrected geometry yields 4,220 warnings,
11,877 uncertain and 4,058 untracked approach opportunities out of 20,155, with
zero approach passes. Other targets have 40 warnings, 1,200 conditional passes,
12,294 uncertain and 2,827 untracked out of 16,361. All 15 nominal approaches warn;
two of 13 other events have warning frames (M6: 1.04 s; M7: 0.56 s). The latter
pulse remains visible in the report despite falling below its 0.8 s event criterion.
These are studied encounter categories, not physical collision or CPA truth.

Final history repair excludes invalid clipped measurements from later fits. A
full-source observation replay changes exactly one S1 warning to uncertainty;
all other category counts stay unchanged. Fresh actual neural M2 output exactly
matches current-code replay on all 438 frames. Earlier traces and benchmark
references are archived and retain their provenance.

All eight final H.264/yuv420p videos pass full decode, frame-count, duration and
presentation-time checks: 3,000 frames. Source-to-overlay samples verify export
alignment. All 76 tests pass, including NVENC, VFR/nonzero timestamps and cleanup.
Current M2 is 20.08 processed FPS including fresh process startup, or 23.81 FPS in
the main processing timer, on RTX 4060 at source 1280×720, stride 2, output 1800×1140.
Native 1920×1080 FPV is 10.66 / 12.26 FPS on those timers. These are offline
measurements, not streaming-latency guarantees.

The original CUDA OpenCV 5.0.0 version, path, full build fingerprint and CUDA device
count are unchanged; actual GPU upload/download is exact. Only av 18.1.0 was added.
Use the existing environment directly:

```powershell
.venv/Scripts/python.exe -m airavoid C:/path/input.mp4 --camera auto --output demo/my_video
```

The main limitations remain substantial: no verified real-flight CPA/collision
labels or untouched test set; unknown detector-training overlap; missed targets,
wrong classes and unstable size; unresolved camera translation/rotation and aspect.
The three-state rule does not establish physical clearance. All flight-direction
recommendations are withheld. Birds, balloons and parachutes share the rules but
lack independent real threat validation. Strongest next improvement: synchronized
calibrated RGB and relative-motion truth with a reserved evaluation set.

Recorded mission start: 2026-10-17 22:11 UTC / 2026-10-18 01:11 Europe/Istanbul
(minute precision). Final quota observation: 2026-10-18 02:23:13 UTC, 34% used from
baseline 0%, against a 35 additional percentage-point cap. The meter is integer
and account-wide; it is not an isolated-chat or exact fractional consumption meter.
Final artifact creation timestamps are in reports/mission_final.json.

[Full results](RESULTS.md), [run/environment instructions](README.md),
[research](RESEARCH.md), [geometry and advisory assumptions](GEOMETRY.md),
[experiment ledger](EXPERIMENTS.md), and [artifact hashes](reports/artifact_index.json).
