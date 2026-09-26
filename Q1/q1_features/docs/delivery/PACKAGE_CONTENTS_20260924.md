# Final package contents

The final archive contains the Q1 code, fixed downloaded model weights, OpenFace 2.2.0 source/build/model material, all 100-sample run outputs, review evidence, logs, configuration, tests, and this acceptance record.

Included top-level paths:

- `README.md`, `SOURCE_REVISION`, `FINAL_ACCEPTANCE.md`, `PACKAGE_CONTENTS.md`, `pyproject.toml`
- `src/`, `scripts/`, `tests/`, `configs/`
- `models/`
- `runs/q1_full_20260924/`
- `third_party/OpenFace/`, `third_party/dlib-install/`, `third_party/MMSA-FET/`
- build/model/test evidence files under `env/` (but not an executable Python environment)

Excluded intentionally:

- The original dataset `/home/jqy/shumo/E题数据` (never copied, moved, or modified).
- `env/python/` and the external `/home/jqy/openface-runtime` environment. Exact installed-package and version evidence is included instead.
- compiler/download cache directories under `tools/` and `mamba-root/`.
- unrelated projects and archives outside this Q1 project.

The archive is intended for preservation and transfer. Re-execution still requires compatible Python/CUDA and OpenFace shared-library environments described in `FINAL_ACCEPTANCE.md` and `env/installed-python.txt`.
