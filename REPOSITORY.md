# Repository contents

This repository includes the application, research adapters, scripts, tests,
documentation, measured reports, dataset manifests and annotations, and offline
gallery pages with preview images.

Large runtime assets remain local and are excluded by `.gitignore`: `.venv`,
`external`, `artifacts`, model binaries, source footage, dataset image payloads,
and exported videos. A fresh checkout therefore needs the environment and assets
described in `README.md`, `DELIVERY.md`, and `MOT_RESULTS.md` before running the
pipeline or reproducing benchmarks. Gallery videos are generated assets and are
not included in the Git checkout.

`reports/mot_provenance.json` records the MOT upstream repository URLs and exact
commits, the BoT-SORT compatibility patch, isolated dependency versions, and model
weight hashes and download sources. Restore those upstream checkouts under
`external/mot` using the recorded directory names and commits, then apply the
recorded patch. Preserve the custom CUDA OpenCV installation when preparing the
environment. Dataset manifests identify the benchmark inputs.

Artifact indexes describe the original local experiment outputs, including files
excluded from Git. They are provenance records, not a promise that every indexed
binary is distributed in this repository.
