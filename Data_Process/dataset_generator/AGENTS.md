# Agent Guide

This file gives coding agents the minimum project context and operating rules needed to make safe, compatible changes.

## Project purpose

This Python project converts encoded videos or headerless RAW YUV files into deterministic, aligned training samples:

```text
GT frame sequence + one REF frame + matching LQ frame sequence + metadata
```

The most important contract is temporal and spatial alignment:

- `GT/NNN.png` and `LQ/NNN.png` represent the same source moment.
- `REF/ref.png` is exactly one transformed GT frame from the same clip.
- A source video belongs to only one of `train`, `val`, or `test`.
- Source frames are decoded once, in display order, by one worker.

Python 3.8+, FFmpeg, and ffprobe are required. Runtime dependencies are NumPy, OpenCV headless, and PyYAML.

## Start here

Use the existing virtual environment when it is present:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python generate_dataset.py --help
python validate_dataset.py --help
python regenerate_lq.py --help
```

For a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

The package may also be installed in editable mode:

```bash
pip install -e .
livephoto-generate --help
livephoto-validate --help
livephoto-regenerate-lq --help
```

## Repository map

The root scripts are compatibility entry points. Keep them small.

```text
generate_dataset.py          -> dataset.cli.generate_main
regenerate_lq.py             -> dataset.cli.regenerate_main
validate_dataset.py          -> dataset.cli.validate_main

dataset/
  __init__.py                Public Python API
  cli.py                     Argument parsing and terminal output
  config.py                  Defaults, YAML merge, and validation
  generator.py               End-to-end orchestration and workers
  splitting.py               Deterministic source-level splits
  ffmpeg.py                  Discovery, ffprobe, and RAW YUV metadata
  video_reader.py            Streaming FFmpeg decoder
  clip_sampler.py            Frame-based and time-based sampling
  images.py                  Shared spatial transforms and PNG I/O
  reference_selector.py      REF selection
  degradation.py             GT-to-LQ degradation pipeline
  writer.py                  Atomic clip writer
  metadata.py                JSON read/write helpers
  validator.py               Dataset integrity checks
  regenerator.py             Atomic LQ-only regeneration with rollback
  seeding.py                 Stable SHA-256-derived seeds

tests/test_minimal.py         Unit and real FFmpeg end-to-end coverage
examples/meta.json           Example clip metadata
config.yaml                  General default example
config_*_livephoto.yaml      Specialized 4K Live Photo examples
```

## Generation flow

`dataset.generator.generate_dataset()` performs the following steps:

1. Validate configuration and paths.
2. Discover candidate files and probe their video streams.
3. Assign whole source videos to splits with a deterministic seed.
4. Submit one source video to one worker.
5. Decode frames sequentially through FFmpeg.
6. Sample frame-based or fixed-duration clips.
7. Apply one spatial plan to every frame in a clip to create GT.
8. Select REF directly from the transformed GT frames.
9. Degrade GT frames into aligned LQ frames.
10. Write each clip into a staging directory.
11. Rename clips deterministically, independent of worker completion order.
12. Validate the complete dataset and write `dataset_summary.json`.

## Configuration rules

`load_config()` deep-merges YAML values over `DEFAULT_CONFIG`, then validates the result. When adding a setting, update the default, validation, example YAML, README, and tests as appropriate.

Important rules:

- Time sampling is enabled when `clip.sampling_fps` is not null.
- `sampling_fps` and `clip_duration_seconds` must be set together.
- `num_frames` must equal `sampling_fps * clip_duration_seconds`.
- `drop_last: false` creates a final tail-aligned full clip; it never pads or duplicates frames.
- `image.preserve_source_resolution: true` keeps GT and REF at decoded source size.
- `image.lq_width` and `image.lq_height` must be both set or both null.
- Split keys must be exactly `train`, `val`, and `test`, and ratios must sum to 1.
- RAW `.yuv` files require width, height, pixel format, and FPS from global config, a per-file override, or a sidecar JSON file.
- Sidecar values override per-file values, which override global RAW defaults.

Paths in YAML are resolved relative to the process working directory, not the YAML file.

## Invariants that must not be weakened

- Do not decode one source video separately for GT, REF, and LQ.
- Do not assign clips from one source video to multiple splits.
- Do not introduce repeated or non-increasing `frame_indices`.
- Do not select REF from pre-transform source frames; it must match one saved GT image exactly.
- Do not independently crop or resize frames inside one clip.
- Do not create LQ from a second decode path; derive it from transformed GT.
- Keep output deterministic for the same input, config, dependency versions, and seed.
- Use `derive_seed()` instead of Python's randomized `hash()`.
- Keep writes atomic. A failed clip must not appear as a valid completed clip.
- Preserve regeneration rollback: failed LQ regeneration must restore old LQ and metadata.
- Preserve validator coverage whenever the output schema changes.

Two performance behaviors are intentional:

- Time-based sampling retains only frames requested by active target-FPS grids, not an entire 4K source window.
- A source-resolution spatial plan returns a contiguous frame without calling `cv2.resize()` when no transform is needed.

## Output contract

```text
dataset_output/
  train|val|test/
    clip_000000/
      GT/000.png ...
      REF/ref.png
      LQ/000.png ...
      meta.json
  logs/success.log
  logs/failed.log
  dataset_summary.json
```

Clip IDs are consecutive within each split. PNG numbering starts at zero and is consecutive. Metadata resolution uses `[height, width]`, while OpenCV resize arguments use `(width, height)`.

`dataset_output/`, `.venv/`, caches, staging directories, and package build artifacts are generated files. Do not commit them. Do not delete a non-empty dataset output unless the user explicitly asks.

## Coding conventions

- Put a short English module docstring at the top of every Python file.
- Use relative imports inside the `dataset` package.
- Keep reusable behavior in `dataset`; keep root scripts as compatibility wrappers.
- Keep CLI parsing in `dataset/cli.py`, not in core processing modules.
- Maintain Python 3.8 compatibility; avoid syntax or standard-library APIs introduced later.
- Use `pathlib.Path` for filesystem paths.
- Use the byte-based helpers in `dataset.images` for Unicode-safe OpenCV image I/O.
- Keep process-pool worker callables at module scope so they remain picklable.
- Avoid loading a full video or full dataset into memory.
- Do not silently accept changed frame counts, dimensions, or decode ordering.
- Preserve existing public imports from `dataset.__init__` and root compatibility scripts unless a breaking change is explicitly requested.

## Testing expectations

Run the full suite after behavioral or structural changes:

```bash
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The end-to-end tests create temporary MP4 and RAW YUV inputs, including Unicode paths. They are skipped only when FFmpeg or ffprobe is unavailable.

Add focused tests when changing:

- sampling: assert exact source indices, window bounds, overlap, and tail behavior;
- splitting: assert source-level isolation and deterministic counts;
- degradation: assert fixed-seed reproducibility and metadata;
- spatial transforms: assert GT/REF/LQ dimensions and pixel equality for REF;
- RAW YUV: cover metadata precedence and whole-frame size validation;
- output schema: update validator checks and `examples/meta.json` together.

Useful manual checks:

```bash
.venv/bin/python generate_dataset.py --config config.yaml
.venv/bin/python validate_dataset.py --dataset dataset_output --config config.yaml
.venv/bin/python regenerate_lq.py --dataset dataset_output --config config.yaml
```

Use small synthetic videos for development. A few seconds of 4K PNG output can consume hundreds of megabytes.

## Common change recipes

When adding a degradation operation:

1. Add defaults and validation.
2. Sample base parameters in `degradation.py`.
3. Add clip-consistent jitter behavior if applicable.
4. Apply the operation in the documented pipeline order.
5. Record actual per-frame parameters in metadata.
6. Add deterministic tests and update the README pipeline description.

When adding a sampling mode:

1. Validate all new configuration combinations.
2. Keep decoding streaming and frame indices contiguous.
3. Record source window boundaries, selected indices, timestamps, and effective FPS.
4. Test multiple source frame rates, overlapping windows, short inputs, and tail handling.

When changing metadata:

1. Prefer additive fields.
2. Increase `schema_version` for incompatible semantics.
3. Update generation, regeneration, validation, the example JSON, and tests together.

## Definition of done

Before handing work back:

- Confirm the requested command or API still works.
- Run the relevant focused tests and normally the full suite.
- Confirm validation reports zero errors for any generated fixture.
- Check that no staging directory or generated dataset was accidentally added.
- Review `git status` and preserve unrelated user changes.
- Summarize behavior changes, verification performed, and any intentionally retained limitations.
