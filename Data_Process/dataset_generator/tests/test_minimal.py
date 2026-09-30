from __future__ import annotations

import copy
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.clip_sampler import iter_clips
from dataset.config import DEFAULT_CONFIG
from dataset.metadata import read_json
from dataset.validator import validate_dataset
from generate_dataset import _split_assignments, generate_dataset


class ClipSamplerTest(unittest.TestCase):
    def test_interval_and_tail_alignment(self) -> None:
        frames = [(index, np.full((4, 4, 3), index, dtype=np.uint8)) for index in range(10)]
        clips = list(iter_clips(frames, num_frames=5, frame_interval=2, clip_stride=7, drop_last=False))
        self.assertEqual([clip.frame_indices for clip in clips], [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])

    def test_video_level_split_is_deterministic(self) -> None:
        names = ["video_%02d.mp4" % index for index in range(10)]
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        first = _split_assignments(names, ratios, 42)
        second = _split_assignments(names, ratios, 42)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("train"), 8)
        self.assertEqual(list(first.values()).count("val"), 1)
        self.assertEqual(list(first.values()).count("test"), 1)


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
            config["image"].update({"width": 64, "height": 48, "random_crop": True})
            config["split"] = {"train": 1.0, "val": 0.0, "test": 0.0}
            summary = generate_dataset(config)
            self.assertTrue(summary["validation"]["valid"])
            self.assertEqual(summary["num_train_clips"], 2)
            self.assertEqual(summary["num_failed_videos"], 0)

            report = validate_dataset(output_dir, expected_size=(48, 64), expected_num_frames=5)
            self.assertTrue(report["valid"], report["errors"])
            first_meta = read_json(output_dir / "train" / "clip_000000" / "meta.json")
            self.assertEqual(first_meta["frame_indices"], [0, 2, 4, 6, 8])
            self.assertEqual(first_meta["reference_frame_index"], 4)


if __name__ == "__main__":
    unittest.main()
