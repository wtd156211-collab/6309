"""wavepick: 波形峰值事件与静音检测。

库接口 analyze / analyze_many 与命令行共用同一套检测口径：
固定 10 ms 帧长，帧峰值触发事件（绝对门限 + 前 1 s 峰值下四分位
构成的相对门限），帧均方根判定静音，并按固定的抖动抑制与最短时长
规则输出半开区间 [start_ms, end_ms) 与区间峰值。

只用标准库。长音频按块流式扫描，常驻内存不随音频时长增长。
"""

import argparse
import json
import math
import os
import struct
import sys
from array import array
from collections import deque

__all__ = ["analyze", "analyze_many", "WavepickError", "main"]

ABSOLUTE_THRESHOLD = 0.20
CONTINUE_THRESHOLD = 0.10
BASELINE_WINDOW_FRAMES = 100
BASELINE_FACTOR = 4.0
EVENT_MIN_FRAMES = 3
EVENT_HANGOVER_FRAMES = 11
SILENCE_RMS = 0.004
SILENCE_GLITCH_FRAMES = 6
SILENCE_MIN_FRAMES = 50
FRAME_MS = 10
FULL_SCALE = 32768.0

START_MIN_PEAK = 6554
CONTINUE_MIN_PEAK = 3277

REPORT_VERSION = 1
DEFAULT_REPORT_PATH = os.path.join("var", "report.json")
READ_CHUNK_BYTES = 1 << 20


class WavepickError(Exception):
    """输入文件无法解析或格式不受支持。"""


def _parse_wav_header(fh):
    """按 RIFF 块定位 fmt 与 data，返回 (sample_rate, data_pos, data_size)。"""
    file_size = os.fstat(fh.fileno()).st_size
    header = fh.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise WavepickError("not a RIFF/WAVE file")

    sample_rate = None
    data_pos = data_size = None
    while True:
        chunk_header = fh.read(8)
        if len(chunk_header) == 0:
            break
        if len(chunk_header) < 8:
            raise WavepickError("corrupt: truncated chunk header")
        chunk_id = chunk_header[:4]
        (chunk_size,) = struct.unpack("<I", chunk_header[4:])
        payload_pos = fh.tell()
        if payload_pos + chunk_size > file_size:
            raise WavepickError("corrupt: chunk extends past end of file")

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise WavepickError("corrupt: fmt chunk too small")
            fields = struct.unpack("<HHIIHH", fh.read(16))
            audio_format, channels, rate, _byte_rate, _block_align, bits = fields
            if audio_format != 1:
                raise WavepickError("unsupported: not PCM (format %d)" % audio_format)
            if channels != 1:
                raise WavepickError("unsupported: %d channels (mono only)" % channels)
            if bits != 16:
                raise WavepickError("unsupported: %d-bit samples (16-bit only)" % bits)
            if rate < 100:
                raise WavepickError("unsupported: sample rate %d" % rate)
            sample_rate = rate
        elif chunk_id == b"data":
            if chunk_size % 2:
                raise WavepickError("corrupt: odd data size for 16-bit samples")
            data_pos = payload_pos
            data_size = chunk_size

        fh.seek(payload_pos + chunk_size + (chunk_size & 1))

    if sample_rate is None:
        raise WavepickError("corrupt: missing fmt chunk")
    if data_pos is None:
        raise WavepickError("corrupt: missing data chunk")
    return sample_rate, data_pos, data_size


def _open_wav(path):
    try:
        fh = open(path, "rb")
    except OSError as exc:
        raise WavepickError("cannot open: %s" % (exc.strerror or exc)) from exc
    try:
        sample_rate, data_pos, data_size = _parse_wav_header(fh)
    except Exception:
        fh.close()
        raise
    fh.seek(data_pos)
    return fh, sample_rate, data_size


def _iter_frames(fh, data_size, sample_rate):
    """按 README 的帧边界公式逐帧产出 array('h')，常量内存。"""
    total_samples = data_size // 2
    byteswap = sys.byteorder == "big"
    remaining = data_size
    pending = array("h")
    pos = 0
    frame_index = 0

    while remaining > 0:
        size = min(READ_CHUNK_BYTES, remaining)
        buf = fh.read(size)
        if len(buf) < size:
            raise WavepickError("corrupt: data chunk truncated")
        remaining -= len(buf)

        chunk = array("h")
        chunk.frombytes(buf)
        if byteswap:
            chunk.byteswap()

        i = 0
        count = len(chunk)
        while i < count:
            boundary = min((frame_index + 1) * sample_rate // 100, total_samples)
            need = boundary - pos
            if not pending:
                if need <= count - i:
                    yield chunk[i:i + need]
                    i += need
                    pos = boundary
                    frame_index += 1
                else:
                    pending.extend(chunk[i:])
                    i = count
            else:
                take = min(need - len(pending), count - i)
                pending.extend(chunk[i:i + take])
                i += take
                if len(pending) == need:
                    yield pending
                    pending = array("h")
                    pos = boundary
                    frame_index += 1

    if pos != total_samples:
        raise WavepickError("corrupt: data chunk truncated")


class _Analyzer:
    def __init__(self):
        self.baseline_window = deque(maxlen=BASELINE_WINDOW_FRAMES)
        self.frame_index = 0

        self.event_start = None
        self.event_active = None
        self.event_miss = 0
        self.event_peak = 0
        self.events = []

        self.silence_start = None
        self.silence_last = None
        self.silence_peak = 0
        self.silence_gap = 0
        self.silence_gap_peak = 0
        self.silences = []

    def feed(self, frame):
        n = self.frame_index
        top = max(frame)
        bottom = min(frame)
        peak = top if top >= -bottom else -bottom
        normalized_peak = peak / FULL_SCALE

        rms = math.sqrt(sum(map(int.__mul__, frame, frame)) / len(frame)) / FULL_SCALE

        if self.event_start is None:
            if peak >= START_MIN_PEAK:
                m = len(self.baseline_window)
                baseline = (
                    sorted(self.baseline_window)[(m - 1) // 4] if m else 0.0
                )
                threshold = max(ABSOLUTE_THRESHOLD, BASELINE_FACTOR * baseline)
                if normalized_peak >= threshold:
                    self.event_start = n
                    self.event_active = n
                    self.event_miss = 0
                    self.event_peak = peak
        else:
            if peak >= CONTINUE_MIN_PEAK:
                self.event_active = n
                self.event_miss = 0
                if peak > self.event_peak:
                    self.event_peak = peak
            else:
                self.event_miss += 1
                if self.event_miss >= EVENT_HANGOVER_FRAMES:
                    self._close_event()

        if rms <= SILENCE_RMS:
            if self.silence_start is None:
                self.silence_start = n
                self.silence_peak = 0
            if self.silence_gap:
                if self.silence_gap_peak > self.silence_peak:
                    self.silence_peak = self.silence_gap_peak
                self.silence_gap = 0
                self.silence_gap_peak = 0
            if peak > self.silence_peak:
                self.silence_peak = peak
            self.silence_last = n
        elif self.silence_start is not None:
            self.silence_gap += 1
            if peak > self.silence_gap_peak:
                self.silence_gap_peak = peak
            if self.silence_gap >= SILENCE_GLITCH_FRAMES:
                self._close_silence()

        self.baseline_window.append(normalized_peak)
        self.frame_index += 1

    def _close_event(self):
        start = self.event_start
        end = self.event_active + 1
        if end - start >= EVENT_MIN_FRAMES:
            self.events.append((start, end, self.event_peak))
        self.event_start = None

    def _close_silence(self):
        start = self.silence_start
        end = self.silence_last + 1
        if end - start >= SILENCE_MIN_FRAMES:
            self.silences.append((start, end, self.silence_peak))
        self.silence_start = None
        self.silence_gap = 0
        self.silence_gap_peak = 0

    def finish(self):
        if self.event_start is not None:
            self._close_event()
        if self.silence_start is not None:
            self._close_silence()

    def result(self):
        events = [
            {"start_ms": s * FRAME_MS, "end_ms": e * FRAME_MS, "peak": p}
            for s, e, p in self.events
        ]
        silences = [
            {"start_ms": s * FRAME_MS, "end_ms": e * FRAME_MS, "peak": p}
            for s, e, p in self.silences
        ]
        events.sort(key=lambda iv: (iv["start_ms"], iv["end_ms"]))
        silences.sort(key=lambda iv: (iv["start_ms"], iv["end_ms"]))
        return events, silences


def analyze(path):
    """分析单个 WAV 文件，返回报告中的单文件条目 dict。"""
    fh, sample_rate, data_size = _open_wav(os.fspath(path))
    try:
        total_samples = data_size // 2
        analyzer = _Analyzer()
        for frame in _iter_frames(fh, data_size, sample_rate):
            analyzer.feed(frame)
        analyzer.finish()
        events, silences = analyzer.result()
    finally:
        fh.close()

    return {
        "file": os.path.basename(os.fspath(path)),
        "sample_rate": sample_rate,
        "channels": 1,
        "bits_per_sample": 16,
        "samples": total_samples,
        "duration_ms": (1000 * total_samples + sample_rate - 1) // sample_rate,
        "events": events,
        "silences": silences,
    }


def _expand_inputs(inputs, on_error=None):
    """展开目录（只取其中的 *.wav，不递归）并按绝对路径去重。"""
    paths = []
    seen = set()

    def add(path):
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)

    for item in inputs:
        item = os.fspath(item)
        if os.path.isdir(item):
            try:
                names = sorted(os.listdir(item))
            except OSError as exc:
                if on_error is None:
                    raise WavepickError("cannot read directory: %s" % exc) from exc
                on_error(item, WavepickError("cannot read directory: %s" % exc))
                continue
            for name in names:
                if name.endswith(".wav"):
                    add(os.path.join(item, name))
        elif os.path.isfile(item):
            add(item)
        else:
            error = WavepickError("no such file or directory")
            if on_error is None:
                raise error
            on_error(item, error)
    return paths


def analyze_many(paths, on_error=None):
    """分析多个文件或目录，返回完整报告 dict；失败项交给 on_error 回调。"""
    entries = []
    for path in _expand_inputs(paths, on_error=on_error):
        try:
            entries.append(analyze(path))
        except (WavepickError, OSError) as exc:
            if on_error is None:
                raise
            on_error(path, exc if isinstance(exc, WavepickError) else WavepickError(str(exc)))
    entries.sort(key=lambda entry: entry["file"])
    return {"version": REPORT_VERSION, "files": entries}


def _write_report(report, out_path):
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _cmd_analyze(args):
    errors = 0

    def on_error(path, exc):
        nonlocal errors
        errors += 1
        sys.stderr.write("error: %s: %s\n" % (path, exc))

    report = analyze_many(args.input, on_error=on_error)
    try:
        _write_report(report, args.out)
    except OSError as exc:
        sys.stderr.write("error: %s: %s\n" % (args.out, exc.strerror or exc))
        return 2
    return 2 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wavepick.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="analyze WAV files")
    analyze_parser.add_argument(
        "--input", nargs="+", required=True, metavar="PATH",
        help="WAV 文件或目录（目录只取其中的 *.wav，不递归）",
    )
    analyze_parser.add_argument(
        "--out", default=DEFAULT_REPORT_PATH, metavar="REPORT_JSON",
        help="报告输出路径（默认 var/report.json）",
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
