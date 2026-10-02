#!/usr/bin/env python3
"""wavepick 的 unittest 自测。只读 samples/**，临时文件写到系统临时目录。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wavepick

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(ROOT, 'samples', 'audio')
LABEL_DIR = os.path.join(ROOT, 'samples', 'labels')


def load_label(name):
    with open(os.path.join(LABEL_DIR, name + '.json'), encoding='utf-8') as fp:
        return json.load(fp)


def make_wav(path, samples, rate=16000, channels=1, sampwidth=2):
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(rate)
        wav.writeframes(samples)


def tone(freq, ms, amp, rate=16000):
    import math
    count = int(rate * ms / 1000)
    return array('h', (int(amp * math.sin(2 * math.pi * freq * i / rate))
                       for i in range(count))).tobytes()


class TestSamplesMatchLabels(unittest.TestCase):
    """逐条核对：每个样例的事件/静音条数、起止毫秒、强度与标注一致。"""

    def test_all_samples(self):
        names = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(AUDIO_DIR) if name.endswith('.wav'))
        self.assertEqual(
            names,
            sorted(os.path.splitext(name)[0]
                   for name in os.listdir(LABEL_DIR) if name.endswith('.json')))
        for name in names:
            with self.subTest(sample=name):
                entry = wavepick.analyze(os.path.join(AUDIO_DIR, name + '.wav'))
                label = load_label(name)
                self.assertEqual(entry['file'], label['file'])
                self.assertEqual(entry['events'], label['events'])
                self.assertEqual(entry['silences'], label['silences'])

    def test_report_meta(self):
        entry = wavepick.analyze(os.path.join(AUDIO_DIR, 'noisy_floor.wav'))
        self.assertEqual(entry['sample_rate'], 16000)
        self.assertEqual(entry['channels'], 1)
        self.assertEqual(entry['bits_per_sample'], 16)
        self.assertEqual(entry['samples'], 128000)
        self.assertEqual(entry['duration_ms'], 8000)


class TestReportShape(unittest.TestCase):
    def test_analyze_many_structure_and_order(self):
        report = wavepick.analyze_many([AUDIO_DIR])
        self.assertEqual(report['version'], 1)
        files = report['files']
        self.assertEqual([e['file'] for e in files],
                         sorted(e['file'] for e in files))
        for entry in files:
            self.assertEqual(
                list(entry.keys()),
                ['file', 'sample_rate', 'channels', 'bits_per_sample',
                 'samples', 'duration_ms', 'events', 'silences'])
            for kind in ('events', 'silences'):
                spans = entry[kind]
                self.assertEqual(spans, sorted(
                    spans, key=lambda s: (s['start_ms'], s['end_ms'])))
                for span in spans:
                    self.assertEqual(list(span.keys()),
                                     ['start_ms', 'end_ms', 'peak'])
                    for value in span.values():
                        self.assertIsInstance(value, int)

    def test_dedup_and_directory_input(self):
        one = os.path.join(AUDIO_DIR, 'sudden_pop.wav')
        report = wavepick.analyze_many([one, one, AUDIO_DIR])
        self.assertEqual(len(report['files']), 5)
        self.assertEqual(
            sum(1 for e in report['files'] if e['file'] == 'sudden_pop.wav'), 1)

    def test_determinism(self):
        first = wavepick.analyze_many([AUDIO_DIR])
        second = wavepick.analyze_many([AUDIO_DIR])
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, indent=2),
            json.dumps(second, ensure_ascii=False, indent=2))


class TestSyntheticRules(unittest.TestCase):
    """用合成波形核对状态机口径（临时目录，不碰 samples）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_short_event_dropped(self):
        # 20 ms 尖峰短于 30 ms 最短事件
        data = b'\x00\x00' * 16000 + tone(440, 20, 20000) + b'\x00\x00' * 16000
        path = self.path('short.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(entry['events'], [])

    def test_event_merge_gap(self):
        # 两次 50 ms 敲击间隔 100 ms（10 帧）应合并为一段
        data = (b'\x00\x00' * 16000 + tone(440, 50, 25000) +
                b'\x00\x00' * 1600 + tone(440, 50, 25000) + b'\x00\x00' * 16000)
        path = self.path('merge.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(len(entry['events']), 1)
        self.assertEqual(entry['events'][0]['start_ms'], 1000)
        self.assertEqual(entry['events'][0]['end_ms'], 1200)

    def test_silence_min_length(self):
        # 300 ms 静音短于 500 ms 不报；600 ms 静音报出
        noise = tone(440, 400, 12000)
        data = (noise + b'\x00\x00' * 4800 + noise +
                b'\x00\x00' * 9600 + noise)
        path = self.path('sil.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(len(entry['silences']), 1)
        self.assertEqual(entry['silences'][0]['start_ms'], 1100)
        self.assertEqual(entry['silences'][0]['end_ms'], 1700)

    def test_silence_glitch_merged(self):
        # 静音中 40 ms（4 帧）毛刺不切断静音段
        data = (b'\x00\x00' * 16000 + tone(440, 40, 8000) +
                b'\x00\x00' * 16000)
        path = self.path('glitch.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(len(entry['silences']), 1)
        self.assertEqual(entry['silences'][0]['start_ms'], 0)
        self.assertEqual(entry['silences'][0]['end_ms'], 2040)

    def test_peak_negative_full_scale(self):
        # x = -32768 记作峰值 32768
        data = array('h', [0] * 16000 + [-32768] * 480 + [0] * 16000).tobytes()
        path = self.path('neg.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(entry['events'][0]['peak'], 32768)

    def test_short_final_frame(self):
        # 采样数不是帧长整数倍：末帧短于 10 ms 也要处理
        data = b'\x00\x00' * 16000 + tone(440, 55, 25000)
        path = self.path('odd.wav')
        make_wav(path, data)
        entry = wavepick.analyze(path)
        self.assertEqual(entry['duration_ms'], 1055)
        self.assertEqual(len(entry['events']), 1)


class TestWavParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_list_chunk_skipped(self):
        # soft_passage.wav 头部夹 LIST/INFO 块，能按块解析即通过
        entry = wavepick.analyze(os.path.join(AUDIO_DIR, 'soft_passage.wav'))
        self.assertEqual(entry['events'], [])
        self.assertEqual(entry['silences'], [])

    def test_truncated_data_is_corrupt(self):
        src = os.path.join(AUDIO_DIR, 'sudden_pop.wav')
        with open(src, 'rb') as handle:
            raw = handle.read()
        path = self.path('trunc.wav')
        with open(path, 'wb') as fp:
            fp.write(raw[:-1000])
        with self.assertRaises(wavepick.WavCorruptError):
            wavepick.analyze(path)

    def test_unsupported_formats(self):
        stereo = self.path('stereo.wav')
        make_wav(stereo, b'\x00\x00' * 3200, channels=2)
        with self.assertRaises(wavepick.WavFormatError):
            wavepick.analyze(stereo)

        eight_bit = self.path('8bit.wav')
        make_wav(eight_bit, b'\x80' * 1600, sampwidth=1)
        with self.assertRaises(wavepick.WavFormatError):
            wavepick.analyze(eight_bit)

        not_wav = self.path('not.wav')
        with open(not_wav, 'wb') as fp:
            fp.write(b'NOPE' * 100)
        with self.assertRaises(wavepick.WavFormatError):
            wavepick.analyze(not_wav)

    def test_missing_file(self):
        with self.assertRaises(OSError):
            wavepick.analyze(self.path('missing.wav'))


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, 'wavepick.py'), *args],
            capture_output=True, text=True)

    def test_analyze_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'report.json')
            proc = self.run_cli('analyze', '--input', AUDIO_DIR, '--out', out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, '')
            report = json.load(open(out, encoding='utf-8'))
            self.assertEqual(report['version'], 1)
            self.assertEqual(len(report['files']), 5)
            misses, false_alarms = 0, 0
            for entry in report['files']:
                label = load_label(os.path.splitext(entry['file'])[0])
                for kind in ('events', 'silences'):
                    self.assertEqual(entry[kind], label[kind])
            self.assertEqual((misses, false_alarms), (0, 0))

    def test_cli_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = os.path.join(tmp, 'r1.json')
            out2 = os.path.join(tmp, 'r2.json')
            env1 = dict(os.environ, PYTHONHASHSEED='1')
            env2 = dict(os.environ, PYTHONHASHSEED='42')
            for out, env in ((out1, env1), (out2, env2)):
                proc = subprocess.run(
                    [sys.executable, os.path.join(ROOT, 'wavepick.py'),
                     'analyze', '--input', AUDIO_DIR, '--out', out],
                    capture_output=True, text=True, env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(open(out1, 'rb').read(), open(out2, 'rb').read())

    def test_bad_file_exit_code_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, 'bad.wav')
            with open(bad, 'wb') as fp:
                fp.write(b'not a wav')
            out = os.path.join(tmp, 'report.json')
            proc = self.run_cli('analyze', '--input', bad,
                                '--input', AUDIO_DIR, '--out', out)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('error: bad.wav:', proc.stderr)
            report = json.load(open(out, encoding='utf-8'))
            self.assertEqual(len(report['files']), 5)

    def test_bad_args_exit_code_2(self):
        proc = self.run_cli('analyze')
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(proc.stderr.startswith('error: '))
        proc = self.run_cli('analyze', '--input')
        self.assertEqual(proc.returncode, 2)


if __name__ == '__main__':
    unittest.main()
