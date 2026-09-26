# Q1 server execution and acceptance record

Generated: 2026-09-24 (UTC)

## Outcome

- The 100-sample extraction pipeline completed and the final full-run validation passed.
- `reports/validation.json`: `ok=true`, `errors=[]`, `selected_only=false`.
- All seven per-sample stages report `100 ok / 0 failed / 0 not_processed`.
- Final processing state is `49 ok / 51 partial`; the 51 partial rows are intentionally quarantined unresolved review cases, not unprocessed samples.
- Pairing state is `49 no_issue_detected / 51 unverifiable`; every unverifiable row has `paired_use=false`.

## Provenance and runtime

- SOURCE_REVISION: `da6c8e5bd553d0480ca941384ae20f279929872f`
- Source archive SHA256: `d871d92a0fde80515512cf1f7f7287e58021b137cc608a7705cc2fa754dbb674`
- Runtime Python: `${Q1_ENV}/bin/python`, Python 3.10.21
- PyTorch / torchaudio: 2.5.1 / 2.5.1
- CUDA runtime / GPU / driver: 12.1 / NVIDIA GeForce RTX 4090 / 580.178.04
- FFmpeg: 6.1.1
- OpenFace: official `OpenFace_2.2.0`; real-video execution produced 71 data rows and exited 0
- OpenFace C++ dependencies used: OpenCV 4.1.2, OpenBLAS 0.3.34, dlib 19.24.8
- Python OpenCV: 4.8.1
- `pip check`: no broken requirements
- Unit tests: `31 passed in 0.19s`

## Fixed model records

- BERT `google-bert/bert-base-uncased`: revision `86b5e0934494bd15c9632b12f734a8a67f723594`
- CTC `espnet/kamo-naoyuki_librispeech_asr_train_asr_conformer5_raw_bpe5000_schedule-truncated-c8e5f9`: revision `e6a0f274799b5a4c157d1d5bc20de4c569e25f7e`
- Both models passed strict offline loading; BERT returned finite `(1, 7, 768)` float32 output and CTC loaded on `cuda:0`.

## Data and stage accounting

- Manifest / unique sample IDs / unique video IDs: 100 / 100 / 37
- Labels / sample-index rows: 100 / 100
- Input status `ok`: 100; missing source videos: 0
- media: 100 ok
- text: 100 ok
- align: 100 ok
- audio: 100 ok
- vision: 100 ok
- audit: 100 ok
- pool: 100 ok
- zero `not_processed`

## Modality accounting

- Text: extraction succeeded for 100; final time-eligible samples 77; unresolved text/audio alignment samples 23; extraction failures 0.
- Audio: extraction succeeded for 100; native eligible samples 100; natural missing samples 0; extraction failures 0.
- Vision: extraction succeeded for 100; final eligible samples 59; identity-unresolved samples 37; natural zero-face samples 4; extraction failures 0.
- The four zero-face IDs are `-NFrJFQijFE$_$1`, `-NFrJFQijFE$_$2`, `-mJ2ud6oKI8$_$1`, and `-ri04Z7vwnc$_$0`. They are preserved as finite, zero-valued rows with `eligible=false` and reason `no_face`.
- Aggregate observed windows: text 3128/5000, audio 4975/5000, vision 2873/5000.

## Review accounting

- Review rows: 78 across 51 unique samples.
- By issue: 37 `identity_unknown`, 22 `suspected_text_audio_mismatch`, 19 `low_aligned_word_fraction`.
- Confirmed match / confirmed mismatch / unresolved: 0 / 0 / 78.
- Six identity contact sheets (111 inspected frame points) and a per-issue CTC timeline evidence table are retained under `reports/manual_review_evidence/`.
- No target-face segment was released because still-frame evidence could not reliably bind every OpenFace episode to the speaking target. `configs/target_face_segments.csv` therefore remains header-only.
- No unresolved record is represented as human-confirmed. All 51 affected samples remain `unverifiable` with `paired_use=false`.

Unresolved sample IDs (51):

`-9y-fZ3swSY$_$8`, `-HwX2H8Z4hY$_$2`, `-HwX2H8Z4hY$_$5`, `-HwX2H8Z4hY$_$6`, `-HwX2H8Z4hY$_$9`, `-MeTTeMJBNc$_$0`, `-MeTTeMJBNc$_$7`, `-NFrJFQijFE$_$1`, `-NFrJFQijFE$_$2`, `-RfYyzHpjk4$_$2`, `-THoVjtIkeU$_$6`, `-UuX1xuaiiE$_$0`, `-UuX1xuaiiE$_$1`, `-UuX1xuaiiE$_$3`, `-UuX1xuaiiE$_$6`, `-aqamKhZ1Ec$_$0`, `-dxfTGcXJoc$_$1`, `-dxfTGcXJoc$_$6`, `-hnBHBN8p5A$_$6`, `-hnBHBN8p5A$_$7`, `-iRBcNs9oI8$_$3`, `-iRBcNs9oI8$_$6`, `-iRBcNs9oI8$_$7`, `-iRBcNs9oI8$_$8`, `-iRBcNs9oI8$_$9`, `-lzEya4AM_4$_$6`, `-mJ2ud6oKI8$_$1`, `-mJ2ud6oKI8$_$2`, `-mJ2ud6oKI8$_$6`, `-mJ2ud6oKI8$_$8`, `-mJ2ud6oKI8$_$9`, `-ri04Z7vwnc$_$0`, `-ri04Z7vwnc$_$2`, `-ri04Z7vwnc$_$5`, `-s9qJ7ATP7w$_$0`, `-s9qJ7ATP7w$_$1`, `-s9qJ7ATP7w$_$4`, `-s9qJ7ATP7w$_$7`, `-s9qJ7ATP7w$_$8`, `-tANM6ETl_M$_$3`, `-wMB_hJL-3o$_$7`, `-wny0OAz3g8$_$0`, `-wny0OAz3g8$_$1`, `-wny0OAz3g8$_$2`, `-wny0OAz3g8$_$3`, `-wny0OAz3g8$_$5`, `-wny0OAz3g8$_$7`, `-wny0OAz3g8$_$9`, `-yRb-Jum7EQ$_$1`, `-yRb-Jum7EQ$_$5`, `-yRb-Jum7EQ$_$6`.

## Validation details

- Aggregate arrays are `(100,50,768)` text, `(100,50,25)` audio, and `(100,50,22)` vision.
- All arrays have expected dtype and finite values; all masks are binary.
- CSR maps, native source IDs, required timelines, observed windows, and coverage-union recomputation passed.
- Recomputed observed-window count: 10,976.
- `reports/package_size.csv` exists.
- Full run directory byte size before final packaging: 6,474,749,826 bytes.
- Compact-candidate size recorded by the report: 1,544,177,277 bytes.

## Minimal code changes made after real-data diagnosis

1. `src/q1_features/extractors/audio.py`: optionally clips the final openSMILE interval to the authoritative media duration. This fixes two resampling round-up errors of 0.000037 s and 0.000022 s.
2. `src/q1_features/runner.py`: passes media duration to audio extraction; treats a successful OpenFace zero-face result as explicit `no_face` rows; and makes `vision --reuse-raw-csv` verify prior successful raw evidence without invoking OpenFace again.
3. `configs/alignment_review.csv`: 78 evidence-bound unresolved review rows were added. No row was marked confirmed.

No tests were deleted, skipped, or weakened. No original dataset file was deleted, moved, or overwritten.

## Key paths

- Run: `${PROJECT_ROOT}/runs/q1_full_20260924`
- Aggregate features: `${PROJECT_ROOT}/runs/q1_full_20260924/features/q1_compact50.npz`
- Validation: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/validation.json`
- Summary: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/summary.md`
- Quality: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/quality.csv`
- Alignment issues: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/alignment_issues.csv`
- Package sizes: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/package_size.csv`
- Review evidence: `${PROJECT_ROOT}/runs/q1_full_20260924/reports/manual_review_evidence`
- OpenFace build/run evidence: `${PROJECT_ROOT}/env/openface-build.log`, `${PROJECT_ROOT}/env/openface-real-video.log`

