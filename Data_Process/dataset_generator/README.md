# 视频训练数据集生成器

该工具把不同封装、帧率和分辨率的视频转换为严格对齐的训练样本：

```text
sample = [GT_0 ... GT_T-1, REF, LQ_0 ... LQ_T-1]
GT_t <-> LQ_t
REF ∈ {GT_0 ... GT_T-1}
```

每个 source video 只由一个 worker 解码一次；同一视频的全部 clip 只会进入同一个 train/val/test split。GT、REF 和 LQ 由同一组内存帧派生，不会通过独立解码路径产生时间偏移。

## 环境

- Python 3.8+
- FFmpeg 和 ffprobe（须可从 `PATH` 调用，也可在 YAML 中填写绝对路径）
- Python 依赖见 `requirements.txt`

```bash
cd dataset_generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version
ffprobe -version
```

## 快速开始

1. 把输入视频放到 `./videos`（可包含任意层级的子目录）。
2. 按需编辑 `config.yaml`。
3. 生成并自动校验：

```bash
python generate_dataset.py --config config.yaml
```

也可以直接使用命令行覆盖常用参数：

```bash
python generate_dataset.py \
  --input ./videos \
  --output ./dataset_output \
  --num-frames 7 \
  --frame-interval 1 \
  --clip-stride 7 \
  --num-workers 8
```

裸 `.yuv` 输入还可以直接从 CLI 指定公共参数：

```bash
python generate_dataset.py \
  --input ./videos \
  --output ./dataset_output \
  --yuv-width 1920 \
  --yuv-height 1080 \
  --yuv-pix-fmt yuv420p \
  --yuv-fps 30
```

YAML 会先加载，显式 CLI 参数随后覆盖 YAML。若输出目录已经含有 `train/val/test/logs`，程序默认拒绝混写；确认需要替换时使用 `--overwrite` 或设置 `overwrite: true`。

## 从输入到 clip 的完整示例

假设 `videos/街景/001.mov` 有 300 帧，配置为：

```yaml
clip:
  num_frames: 5
  frame_interval: 2
  clip_stride: 5
  drop_last: true
reference:
  strategy: center
```

第一个 clip 的 source frame indices 为 `[0, 2, 4, 6, 8]`。空间变换只采样一次并应用到五帧；`REF/ref.png` 等于空间变换后的 `GT/002.png`；`LQ/000.png` 到 `LQ/004.png` 分别只由对应 GT 生成。运行：

```bash
python generate_dataset.py \
  --config config.yaml \
  --input ./videos \
  --output ./dataset_output
```

得到：

```text
dataset_output/
├── train/
│   ├── clip_000000/
│   │   ├── GT/
│   │   │   ├── 000.png
│   │   │   ├── 001.png
│   │   │   ├── 002.png
│   │   │   ├── 003.png
│   │   │   └── 004.png
│   │   ├── REF/
│   │   │   └── ref.png
│   │   ├── LQ/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── meta.json
│   └── ...
├── val/
├── test/
├── logs/
│   ├── success.log
│   └── failed.log
└── dataset_summary.json
```

每个 split 内的 clip ID 连续编号；编号不依赖 worker 完成顺序，所以改变 `num_workers` 不会改变样本内容和最终顺序。

## 处理流程与对齐保证

```text
ffprobe 验证真实视频流
  → FFmpeg 单次顺序解码（忽略音频，VFR 不补帧）
  → 按解码显示顺序选择 source frame indices
  → clip 级统一 resize/crop
  → GT
      ├── 选择一张 GT 作为 REF
      └── 逐帧退化 → 可选 clip 级视频压缩 → LQ
  → 原子写入
  → 全数据集校验
```

- 文件扩展名不用于判定文件是否合法。默认 `video.scan_all_files: true`，递归发现的每个普通文件都会交给 ffprobe 验证。若输入目录混有大量非视频文件，可以设为 `false`，此时扩展名只负责候选过滤，候选仍必须通过 ffprobe。
- 裸 `.yuv` 是例外：它没有容器头，文件内部无法携带 width、height、pixel format 或 FPS，因此必须通过 YAML、CLI 或 sidecar 显式提供这些信息；工具随后使用显式 rawvideo 参数交给 FFmpeg 验证和解码。
- VFR 视频使用 passthrough 同步模式（FFmpeg 5+ 为 `-fps_mode passthrough`，4.x 为 `-vsync 0`），不会为了制造恒定 FPS 而复制或丢弃帧。`frame_indices` 表示解码后的显示顺序；`fps` 是 ffprobe 的平均帧率，原始 rate 字段也写入 meta。
- 手机视频的 90/180/270 度旋转元数据会在解码后统一应用。
- FFmpeg 将各种源像素格式转换为 8-bit `bgr24`，OpenCV 中保持这一规范直到无损 PNG 写入。
- 单个视频只由一个进程处理，避免多个 worker 重复 decode。
- 损坏视频不会终止整个任务。若 `allow_partial_decode: true`，解码错误之前已经完整写好的 clip 可保留；否则该视频的 staging 数据全部丢弃。

## RAW YUV 输入

当所有 `.yuv` 文件规格一致时，可在 YAML 中设置公共参数：

```yaml
video:
  raw_yuv:
    enabled: true
    width: 1920
    height: 1080
    pix_fmt: yuv420p
    fps: 30
    rotation: 0
```

若不同文件的规格不同，推荐在每个文件旁放置 `<原文件名>.json`。例如 `街景.yuv` 对应 `街景.yuv.json`：

```json
{
  "width": 1920,
  "height": 1080,
  "pix_fmt": "yuv420p10le",
  "fps": 29.97,
  "rotation": 0
}
```

sidecar 不会被当作视频候选，也不会计入失败文件。它的值覆盖 YAML 公共参数。还可以在 YAML 中按相对 `input_dir` 的路径配置：

```yaml
video:
  raw_yuv:
    files:
      camera_a/clip001.yuv:
        width: 3840
        height: 2160
        pix_fmt: yuv420p10le
        fps: 60
```

参数优先级为：`sidecar > files 中的单文件配置 > raw_yuv 公共配置`。支持 FFmpeg 能解码的 raw pixel format，例如 `yuv420p`、`yuv422p`、`yuv444p`、`nv12`、`p010le` 和高 bit-depth planar YUV。配置错误或文件不足一个完整 raw frame 时，该 source 会写入 `failed.log`，不会影响其他视频。

## Clip 尾部策略

时间跨度为：

```text
span = (num_frames - 1) * frame_interval + 1
```

`drop_last: true` 会丢弃不足一个完整 span 的尾部。`drop_last: false` 不使用重复帧或 padding，而是额外生成一个与视频末帧对齐的完整 clip；这样始终保持 `num_frames` 固定且 source indices 严格递增。短于一个完整 span 的视频无论哪种模式都不会生成 clip，并计入 summary 的 `num_short_videos`。

若 clip 必须对应精确的真实时长，可启用时间采样。例如 45 帧对应 3 秒即目标 15 FPS：

```yaml
clip:
  num_frames: 45
  sampling_fps: 15.0
  clip_duration_seconds: 3.0
  clip_stride_seconds: 3.0
  drop_last: true
```

对于 50 FPS 源视频，每个窗口严格使用 150 个源帧，并以确定性的 3/4 帧交替间隔选出 45 帧。`meta.json` 会分别记录 `source_fps: 50`、`fps/target_fps: 15`、源窗口边界、实际 source indices 和时间戳；视频压缩也按目标 15 FPS 执行。

## 空间变换

- `width/height`（或 `gt_width/gt_height`）定义最终尺寸。
- `preserve_source_resolution: true` 会忽略固定 GT 尺寸，让 GT/REF 保持每个源视频的原生分辨率（例如 1920×1080 或 3840×2160）。
- `lq_width/lq_height` 可单独定义 LQ 尺寸；二者均为 `null` 时 LQ 与 GT 同尺寸。
- `lq_interpolation` 定义 GT 到 LQ 的最终缩小插值，照片/视频退化通常推荐 `area`。
- `crop_size` 可为整数或 `[height, width]`，设置后覆盖上述最终尺寸。
- `keep_aspect_ratio: true` 使用 cover resize，再进行 random/center crop，不会拉伸画面。
- `random_crop: true` 时，每个 clip 使用由 clip seed 确定的一组 crop 坐标。
- GT 的全部帧共享空间参数；REF 直接取变换后的某张 GT；LQ 从变换后的 GT 生成，因此三者空间区域一致。

例如让 2K/4K 源视频的 GT/REF 保持原尺寸、LQ 固定为 1K（16:9 的 960×540）：

```yaml
image:
  preserve_source_resolution: true
  lq_width: 960
  lq_height: 540
  lq_interpolation: area
```

`reference.strategy` 支持：

- `center`：中心 GT（偶数帧时取右侧中心）；
- `first` / `last`：首帧或尾帧；
- `random`：确定性随机选择；
- `sharpest`：Laplacian variance 最大的 GT。

## Degradation pipeline

固定顺序为：

```text
Gaussian blur
→ Motion blur
→ Downsample + upsample
→ GT-to-LQ resize（若配置独立 LQ 尺寸）
→ Gaussian / Poisson noise
→ Saturation / hue
→ Brightness / contrast
→ Gamma
→ Sharpen / oversharpen
→ JPEG encode/decode
→ 可选真实视频 encode/decode
```

每个启用项的数值可以是常数或 `[min, max]`。`frame_independent` 为每一帧独立采样全部逐帧参数；`clip_consistent` 先采样一组 base parameters，再按 `temporal_variation` 对每帧作小幅、范围内扰动。视频压缩天然是 clip 级操作，在两种模式下都只采样一次 CRF，并对完整 LQ 序列编码/解码；解码后的帧数和尺寸若变化，当前 source video 会失败而不是静默接受错位。

`device_style` 可按 clip 采样一种手机风格，依次应用 RGB 分通道 gain/bias、以场景中值亮度为中心的对比度与曝光偏移、以及保亮度的色度缩放。设备风格参数在一个 clip 内保持不变，避免逐帧颜色闪烁，并完整记录到 `degradation.base_parameters`。例如：

```yaml
degradation:
  device_style:
    enabled: true
    profile: random
    profiles:
      phone-a:
        rgb_gain: {r: [0.99, 1.02], g: [0.99, 1.02], b: [0.98, 1.02]}
        rgb_bias: {r: [-2, 2], g: [-2, 2], b: [-2, 2]}
        contrast_multiplier: [0.97, 1.03]
        brightness_offset: [-3, 3]
        chroma_multiplier: [0.95, 1.06]
```

已有数据集只需修改 LQ 时，无须重新解码源视频或改写 GT/REF：

```bash
python regenerate_lq.py \
  --config config.yaml \
  --dataset ./dataset_output
```

该命令先从现有 GT 生成临时 LQ，随后原子替换、更新 `meta.json` 并运行完整校验；失败时会恢复原 LQ。

`meta.json` 同时记录 base parameters、每一帧的实际参数和视频压缩参数。完整示例见 `examples/meta.json`。

## 数据划分与可复现性

划分前先收集通过 ffprobe 的 source video，再使用 `seed` 确定性打乱并按最大余数法分配。划分键是相对输入目录的 source path，不是 clip。自动 validator 还会再次检查三个 split 的 `source_video` 集合无交集。

以下随机过程都由稳定 SHA-256 派生 seed 驱动，不使用 Python 进程随机 hash：

- source video 划分；
- random crop；
- random REF；
- 全部 degradation 参数和噪声。

同一 FFmpeg/OpenCV 版本、同一输入、同一配置和同一 seed 下，输出应可复现。跨平台 codec 实现可能有像素级差异，因此建议记录依赖版本。

## 校验

生成结束后会自动检查：

- `len(GT) == len(LQ) == num_frames`，且 `REF` 只有 `ref.png`；
- GT/LQ 文件名严格对应且从 `000.png` 连续编号；GT、REF、LQ 分别按各自元数据分辨率校验；
- 所有图片可解码，尺寸与配置/meta 一致；
- `frame_indices` 严格递增，start/end 一致；
- REF 像素与 `reference_position` 指向的 GT 完全一致；
- degradation 每帧参数数量正确；
- train/val/test 的 source video 不重叠。

也可独立运行：

```bash
python validate_dataset.py \
  --dataset ./dataset_output \
  --config ./config.yaml
```

成功返回码为 0，失败返回码为 1。生成汇总位于 `dataset_summary.json`，失败原因位于 `logs/failed.log`。

## 测试

最小测试会先用 FFmpeg 合成一个含中文路径的视频，再完成真实的 probe、decode、采样、GT/REF/LQ 写入和 validation：

```bash
python -m unittest discover -s tests -v
```

## 工程结构

```text
dataset_generator/
├── generate_dataset.py
├── validate_dataset.py
├── config.yaml
├── dataset/
│   ├── config.py
│   ├── video_reader.py
│   ├── clip_sampler.py
│   ├── reference_selector.py
│   ├── degradation.py
│   ├── writer.py
│   ├── metadata.py
│   └── validator.py
├── utils/
│   ├── ffmpeg_utils.py
│   ├── image_utils.py
│   └── seed.py
├── tests/test_minimal.py
├── examples/meta.json
└── requirements.txt
```
