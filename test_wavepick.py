"""wavepick 的 unittest 自测：只读 samples/**，临时文件写入系统临时目录。"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wavepick

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "samples" / "audio"
LABEL_DIR = ROOT / "samples" / "labels"
CLI = [sys.executable, str(ROOT / "wavepick.py")]

EXPECTED_DURATION_MS = {
    "soft_passage.wav": 10000,
    "noisy_floor.wav": 8000,
    "dense_transients.wav": 6000,
    "long_silence.wav": 16000,
    "sudden_pop.wav": 12000,
}


def write_wav(path, samples, rate=16000, channels=1, bits=16, extra_chunks=(),
              declared_data_size=None):
    data = array("h", samples).tobytes()
    fmt = struct.pack("<HHIIHH", 1, channels, rate,
                      rate * channels * bits // 8, channels * bits // 8, bits)
    chunks = [(b"fmt ", fmt)] + list(extra_chunks) + [(b"data", data)]
    body = b""
    for cid, payload in chunks:
        size = declared_data_size if cid == b"data" and declared_data_size is not None else len(payload)
        body += cid + struct.pack("<I", size) + payload
        if len(payload) % 2:
            body += b"\x00"
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)


def frames(value, count, rate=16000):
    return [value] * (rate // 100) * count


class SampleLabelsTest(unittest.TestCase):
    def test_events_and_silences_match_labels(self):
        for label_path in sorted(LABEL_DIR.glob("*.json")):
            label = json.loads(label_path.read_text(encoding="utf-8"))
            with self.subTest(file=label["file"]):
                entry = wavepick.analyze(AUDIO_DIR / label["file"])
                self.assertEqual(entry["events"], label["events"])
                self.assertEqual(entry["silences"], label["silences"])

    def test_file_metadata(self):
        for name, duration in EXPECTED_DURATION_MS.items():
            with self.subTest(file=name):
                entry = wavepick.analyze(AUDIO_DIR / name)
                self.assertEqual(entry["file"], name)
                self.assertEqual(entry["sample_rate"], 16000)
                self.assertEqual(entry["channels"], 1)
                self.assertEqual(entry["bits_per_sample"], 16)
                self.assertEqual(entry["duration_ms"], duration)
                self.assertEqual(entry["samples"], duration * 16)

    def test_sample_under_one_second(self):
        for wav in sorted(AUDIO_DIR.glob("*.wav")):
            with self.subTest(file=wav.name):
                start = time.perf_counter()
                wavepick.analyze(wav)
                self.assertLess(time.perf_counter() - start, 1.0)


class DeterminismTest(unittest.TestCase):
    def test_repeated_analysis_identical(self):
        for wav in sorted(AUDIO_DIR.glob("*.wav")):
            first = wavepick.analyze(wav)
            second = wavepick.analyze(wav)
            self.assertEqual(first, second)

    def test_cli_bytes_identical_across_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for seed in ("0", "42"):
                out = Path(tmp) / f"report-{seed}.json"
                env = dict(os.environ, PYTHONHASHSEED=seed)
                proc = subprocess.run(
                    CLI + ["analyze", "--input", str(AUDIO_DIR), "--out", str(out)],
                    capture_output=True, text=True, env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                outputs.append(out.read_bytes())
            self.assertEqual(outputs[0], outputs[1])


class CliTest(unittest.TestCase):
    def run_cli(self, *argv, cwd=None):
        return subprocess.run(CLI + list(argv), capture_output=True, text=True, cwd=cwd)

    def test_directory_input_sorted_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = self.run_cli("analyze", "--input", str(AUDIO_DIR), "--out", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["version"], 1)
            names = [f["file"] for f in report["files"]]
            self.assertEqual(names, sorted(names))
            self.assertEqual(len(names), 5)

    def test_default_out_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_cli("analyze", "--input", str(AUDIO_DIR / "sudden_pop.wav"), cwd=tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((Path(tmp) / "var" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual([f["file"] for f in report["files"]], ["sudden_pop.wav"])

    def test_report_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = self.run_cli("analyze", "--input", str(AUDIO_DIR / "sudden_pop.wav"),
                                "--out", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = out.read_text(encoding="utf-8")
            self.assertTrue(text.startswith('{\n  "version": 1,\n  "files": [\n    {\n'
                                            '      "file": "sudden_pop.wav",'))
            entry = json.loads(text)["files"][0]
            self.assertEqual(list(entry), ["file", "sample_rate", "channels",
                                           "bits_per_sample", "samples", "duration_ms",
                                           "events", "silences"])
            self.assertEqual(list(entry["events"][0]), ["start_ms", "end_ms", "peak"])

    def test_duplicate_input_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            wav = str(AUDIO_DIR / "sudden_pop.wav")
            proc = self.run_cli("analyze", "--input", wav, wav, "--out", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(report["files"]), 1)

    def test_bad_inputs_exit_2_and_skip_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = tmp / "good.wav"
            write_wav(good, frames(200, 100))
            (tmp / "notawav.wav").write_text("hello", encoding="utf-8")
            stereo = tmp / "stereo.wav"
            write_wav(stereo, frames(0, 10), channels=2)
            eightbit = tmp / "eightbit.wav"
            write_wav(eightbit, frames(0, 10), bits=8)
            truncated = tmp / "truncated.wav"
            write_wav(truncated, frames(0, 10), declared_data_size=10 ** 6)
            missing = str(tmp / "missing.wav")
            out = tmp / "report.json"

            proc = self.run_cli("analyze", "--input", str(tmp), missing, "--out", str(out))
            self.assertEqual(proc.returncode, 2)
            lines = [l for l in proc.stderr.splitlines() if l.startswith("error: ")]
            self.assertEqual(len(lines), 5, proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual([f["file"] for f in report["files"]], ["good.wav"])

    def test_bad_arguments_exit_2(self):
        proc = self.run_cli("analyze")
        self.assertEqual(proc.returncode, 2)
        proc = self.run_cli("bogus")
        self.assertEqual(proc.returncode, 2)


class SyntheticSignalTest(unittest.TestCase):
    def analyze_samples(self, samples, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.wav"
            write_wav(path, samples, **kw)
            return wavepick.analyze(path)

    def test_event_merge_and_min_duration(self):
        samples = (frames(200, 10) + frames(20000, 3) + frames(200, 10)
                   + frames(20000, 3) + frames(200, 30))
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["events"],
                         [{"start_ms": 100, "end_ms": 260, "peak": 20000}])

    def test_event_split_after_11_miss_frames(self):
        samples = (frames(200, 10) + frames(20000, 3) + frames(200, 11)
                   + frames(20000, 3) + frames(200, 30))
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["events"],
                         [{"start_ms": 100, "end_ms": 130, "peak": 20000},
                          {"start_ms": 240, "end_ms": 270, "peak": 20000}])

    def test_short_event_dropped(self):
        samples = frames(200, 10) + frames(20000, 2) + frames(200, 30)
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["events"], [])

    def test_relative_threshold_blocks_quiet_tap(self):
        samples = (frames(2621, 100) + frames(9000, 5) + frames(2621, 20)
                   + frames(12000, 5) + frames(2621, 20))
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["events"],
                         [{"start_ms": 1250, "end_ms": 1300, "peak": 12000}])

    def test_silence_glitch_merged_and_peak_kept(self):
        samples = frames(100, 100) + frames(5000, 5) + frames(100, 95)
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["silences"],
                         [{"start_ms": 0, "end_ms": 2000, "peak": 5000}])

    def test_silence_split_after_6_loud_frames(self):
        samples = frames(100, 60) + frames(5000, 6) + frames(100, 60)
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["silences"],
                         [{"start_ms": 0, "end_ms": 600, "peak": 100},
                          {"start_ms": 660, "end_ms": 1260, "peak": 100}])

    def test_short_silence_dropped(self):
        samples = frames(200, 20) + frames(100, 49) + frames(200, 20)
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["silences"], [])
        samples = frames(200, 20) + frames(100, 50) + frames(200, 20)
        entry = self.analyze_samples(samples)
        self.assertEqual(entry["silences"],
                         [{"start_ms": 200, "end_ms": 700, "peak": 100}])

    def test_partial_final_frame_and_duration_rounding(self):
        entry = self.analyze_samples(frames(200, 10) + [200] * 5)
        self.assertEqual(entry["samples"], 1605)
        self.assertEqual(entry["duration_ms"], 101)

    def test_full_scale_negative_peak(self):
        entry = self.analyze_samples(frames(200, 10) + frames(-32768, 5) + frames(200, 30))
        self.assertEqual(entry["events"],
                         [{"start_ms": 100, "end_ms": 150, "peak": 32768}])

    def test_extra_chunks_before_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.wav"
            write_wav(path, frames(200, 100),
                      extra_chunks=[(b"LIST", b"\x1a\x00\x00\x00INFOISFT")])
            entry = wavepick.analyze(path)
            self.assertEqual(entry["samples"], 16000)
            self.assertEqual(entry["events"], [])


if __name__ == "__main__":
    unittest.main()
