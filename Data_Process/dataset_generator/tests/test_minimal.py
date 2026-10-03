"""Exercise deterministic sampling, degradation, generation, and validation."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset.clip_sampler import iter_clips, iter_time_clips
from dataset.config import DEFAULT_CONFIG
from dataset.degradation import degrade_clip
from dataset.metadata import read_json
from dataset.splitting import assign_splits
from dataset.validator import validate_dataset
from dataset.generator import generate_dataset


class ClipSamplerTest(unittest.TestCase):
    def test_interval_and_tail_alignment(self) -> None:
        frames = [(index, np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(10)]
        clips = list(iter_clips(frames, num_frames=5, frame_interval=2, clip_stride=7, drop_last=False))
        self.assertEqual([clip.frame_indices for clip in clips], [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])

    def test_video_level_split_is_deterministic(self) -> None:
        names = ["video_%02d.mp4" % index for index in range(10)]
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        first = assign_splits(names, ratios, 42)
        second = assign_splits(names, ratios, 42)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("train"), 8)
        self.assertEqual(list(first.values()).count("val"), 1)
        self.assertEqual(list(first.values()).count("test"), 1)

    def test_45_frames_span_exactly_three_seconds_at_15_fps(self) -> None:
        frames = [(index, np.zeros((1, 1, 3), dtype=np.uint8)) for index in range(600)]
        clips = list(iter_time_clips(
            frames,
            num_frames=45,
            source_fps=50.0,
            target_fps=15.0,
            clip_duration_seconds=3.0,
            clip_stride_seconds=3.0,
            drop_last=True,
        ))
        self.assertEqual(len(clips), 4)
        self.assertEqual(clips[0].frame_indices[:5], [0, 3, 7, 10, 13])
        self.assertEqual(clips[0].frame_indices[-1], 147)
        self.assertEqual((clips[0].window_start_frame, clips[0].window_end_frame), (0, 149))
        self.assertEqual((clips[-1].window_start_frame, clips[-1].window_end_frame), (450, 599))

    def test_time_sampling_supports_overlapping_windows_and_tail_alignment(self) -> None:
        frames = [(index, np.zeros((1, 1, 3), dtype=np.uint8)) for index in range(18)]
        clips = list(iter_time_clips(
            frames,
            num_frames=5,
            source_fps=10.0,
            target_fps=5.0,
            clip_duration_seconds=1.0,
            clip_stride_seconds=0.5,
            drop_last=False,
        ))
        self.assertEqual(
            [clip.frame_indices for clip in clips],
            [[0, 2, 4, 6, 8], [5, 7, 9, 11, 13], [8, 10, 12, 14, 16]],
        )

    def test_device_style_is_clip_consistent_and_reproducible(self) -> None:
        frame = np.full((48, 64, 3), (80, 120, 180), dtype=np.uint8)
        config = {
            "mode": "clip_consistent",
            "temporal_variation": 0.1,
            "device_style": {
                "enabled": True,
                "profile": "test-phone",
                "profiles": {
                    "test-phone": {
                        "rgb_gain": {"r": [1.02, 1.02], "g": [1.0, 1.0], "b": [0.98, 0.98]},
                        "rgb_bias": {"r": [1.0, 1.0], "g": [0.0, 0.0], "b": [-1.0, -1.0]},
                        "contrast_multiplier": [1.01, 1.01],
                        "brightness_offset": [2.0, 2.0],
                        "chroma_multiplier": [1.03, 1.03],
                    }
                },
            },
        }
        first, first_meta = degrade_clip(
            [frame, frame], config, np.random.default_rng(99), fps=15.0, output_size=(24, 32)
        )
        second, second_meta = degrade_clip(
            [frame, frame], config, np.random.default_rng(99), fps=15.0, output_size=(24, 32)
        )
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(first_meta["base_parameters"]["device_style_profile"], "test-phone")
        self.assertEqual(first_meta["frame_parameters"][0], first_meta["frame_parameters"][1])
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertFalse(np.array_equal(first[0], cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe are required")
class EndToEndTest(unittest.TestCase):
    def test_synthetic_video_to_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dataset-generator-test-") as root_value:
            root = Path(root_value)
            input_dir, output_dir = root / "输入视频", root / "输出数据集"
            input_dir.mkdir()
            video_path = input_dir / "测试视频.mp4"
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=96x64:rate=12",
                "-frames:v", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))

            config = copy.deepcopy(DEFAULT_CONFIG)
            config.update({
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "num_workers": 1,
                "seed": 123,
            })
            config["clip"].update({"num_frames": 5, "frame_interval": 2, "clip_stride": 5, "drop_last": True})
            config["image"].update({
                "width": 64,
                "height": 48,
                "lq_width": 32,
                "lq_height": 24,
                "random_crop": True,
            })
            config["split"] = {"train": 1.0, "val": 0.0, "test": 0.0}
            summary = generate_dataset(config)
            self.assertTrue(summary["validation"]["valid"])
            self.assertEqual(summary["num_train_clips"], 2)
            self.assertEqual(summary["num_failed_videos"], 0)

            first_clip = output_dir / "train" / "clip_000000"
            for hidden_path in (
                first_clip / "GT" / "._000.png",
                first_clip / "LQ" / "._000.png",
                first_clip / "REF" / "._ref.png",
            ):
                hidden_path.write_bytes(b"filesystem metadata")

            report = validate_dataset(
                output_dir,
                expected_size=(48, 64),
                expected_lq_size=(24, 32),
                expected_num_frames=5,
            )
            self.assertTrue(report["valid"], report["errors"])
            first_meta = read_json(first_clip / "meta.json")
            self.assertEqual(first_meta["frame_indices"], [0, 2, 4, 6, 8])
            self.assertEqual(first_meta["reference_frame_index"], 4)
            self.assertEqual(first_meta["gt_resolution"], [48, 64])
            self.assertEqual(first_meta["ref_resolution"], [48, 64])
            self.assertEqual(first_meta["lq_resolution"], [24, 32])

    def test_raw_yuv_with_unicode_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dataset-generator-yuv-test-") as root_value:
            root = Path(root_value)
            input_dir, output_dir = root / "YUV输入", root / "YUV输出"
            input_dir.mkdir()
            yuv_path = input_dir / "裸视频.yuv"
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                "-frames:v", "12", "-pix_fmt", "yuv420p", "-f", "rawvideo", str(yuv_path),
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
            sidecar = {"width": 64, "height": 48, "pix_fmt": "yuv420p", "fps": 10}
            Path(str(yuv_path) + ".json").write_text(json.dumps(sidecar), encoding="utf-8")

            config = copy.deepcopy(DEFAULT_CONFIG)
            config.update({
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "num_workers": 1,
                "seed": 321,
            })
            config["clip"].update({
                "num_frames": 5,
                "frame_interval": 1,
                "clip_stride": 5,
                "drop_last": True,
                "sampling_fps": 5.0,
                "clip_duration_seconds": 1.0,
                "clip_stride_seconds": 1.0,
            })
            config["image"].update({
                "preserve_source_resolution": True,
                "lq_width": 32,
                "lq_height": 24,
                "random_crop": False,
            })
            config["split"] = {"train": 1.0, "val": 0.0, "test": 0.0}
            summary = generate_dataset(config)
            self.assertTrue(summary["validation"]["valid"])
            self.assertEqual(summary["num_source_videos"], 1)
            self.assertEqual(summary["num_train_clips"], 1)
            meta = read_json(output_dir / "train" / "clip_000000" / "meta.json")
            self.assertTrue(meta["source_is_raw_yuv"])
            self.assertEqual(meta["source_pixel_format"], "yuv420p")
            self.assertEqual(meta["source_fps"], 10.0)
            self.assertEqual(meta["fps"], 5.0)
            self.assertEqual(meta["clip_duration_seconds"], 1.0)
            self.assertTrue(meta["gt_preserves_source_resolution"])
            self.assertEqual(meta["gt_resolution"], [48, 64])
            self.assertEqual(meta["lq_resolution"], [24, 32])


if __name__ == "__main__":
    unittest.main()
