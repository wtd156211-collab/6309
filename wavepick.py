#!/usr/bin/env python3
"""波形峰值事件与静音检测（仅标准库）。

口径见 README.md：10 ms 帧、绝对门限 + 相对基线门限、事件/静音抖动合并、
最短时长过滤。WAV 按 RIFF 块定位 fmt/data，音频数据分块流式扫描，
常驻内存不随音频时长增长。
"""

import bisect
import json
import math
import os
import struct
import sys
from array import array
from collections import deque

VERSION = 1

FRAME_MS = 10                  # 帧长 10 ms
FRAMES_PER_SEC = 100

ABSOLUTE_GATE = 0.20           # 事件绝对门限 A
CONTINUE_GATE = 0.10           # 事件延续门限 A/2
BASELINE_FRAMES = 100          # 基线窗口：当前帧之前 100 帧（1 s）
BASELINE_FACTOR = 4            # 相对阈值 = 4 × B[n]
SILENCE_RMS = 0.004            # 静音帧均方根门限
EVENT_MISS_FRAMES = 11         # 连续 11 帧不满足延续条件则事件结束
EVENT_MIN_FRAMES = 3           # 事件最短 3 帧（30 ms）
SILENCE_MISS_FRAMES = 6        # 连续 6 帧非静音则静音段结束
SILENCE_MIN_FRAMES = 50        # 静音段最短 50 帧（500 ms）

NORMALIZE = 32768.0
READ_CHUNK_BYTES = 1 << 18     # 256 KiB / 次


class WavFormatError(Exception):
    """RIFF/WAVE 结构存在但格式不支持。"""


class WavCorruptError(Exception):
    """文件损坏（声明长度与实际不符等）。"""


def _parse_chunks(fp):
    """按块扫描 RIFF/WAVE，返回 (采样率, data 位置, data 字节数)。"""
    head = fp.read(12)
    if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        raise WavFormatError('不是 RIFF/WAVE 文件')
    fmt_body = None
    data_pos = -1
    data_size = 0
    while True:
        chunk_head = fp.read(8)
        if not chunk_head:
            break
        if len(chunk_head) < 8:
            raise WavCorruptError('块头不完整')
        chunk_id = chunk_head[:4]
        (chunk_size,) = struct.unpack('<I', chunk_head[4:8])
        body_pos = fp.tell()
        if chunk_id == b'fmt ':
            if chunk_size < 16:
                raise WavFormatError('fmt 块长度不足 16 字节')
            fmt_body = fp.read(16)
            if len(fmt_body) < 16:
                raise WavCorruptError('fmt 块不完整')
            fp.seek(chunk_size - 16, 1)
        elif chunk_id == b'data':
            if data_pos >= 0:
                raise WavCorruptError('出现多个 data 块')
            data_pos = body_pos
            data_size = chunk_size
            fp.seek(chunk_size, 1)
        else:
            fp.seek(chunk_size, 1)
        if chunk_size & 1:
            fp.seek(1, 1)
    if fmt_body is None:
        raise WavFormatError('缺少 fmt 块')
    if data_pos < 0:
        raise WavFormatError('缺少 data 块')
    (audio_format, channels, sample_rate, _byte_rate,
     _block_align, bits_per_sample) = struct.unpack('<HHIIHH', fmt_body)
    if audio_format != 1:
        raise WavFormatError('不支持非 PCM 格式（format %d）' % audio_format)
    if channels != 1:
        raise WavFormatError('不支持 %d 声道（仅支持单声道）' % channels)
    if bits_per_sample != 16:
        raise WavFormatError('不支持 %d 位采样（仅支持 16 位）' % bits_per_sample)
    if sample_rate == 0:
        raise WavCorruptError('采样率为 0')
    file_size = fp.seek(0, 2)
    if data_pos + data_size > file_size:
        raise WavCorruptError('data 块声明长度超出文件实际大小')
    if data_size & 1:
        raise WavCorruptError('data 块长度为奇数，16 位采样不完整')
    return sample_rate, data_pos, data_size


def _frame_span(frame, rate):
    return (frame * rate // FRAMES_PER_SEC,
            (frame + 1) * rate // FRAMES_PER_SEC)


def _frame_stats(seg):
    """返回一帧的 (采样绝对值峰值(int), 归一化峰值, 归一化均方根)。"""
    vmax = max(seg)
    neg_max = -min(seg)            # -32768 -> 32768
    int_peak = neg_max if neg_max > vmax else vmax
    sum_sq = 0
    for value in seg:
        sum_sq += value * value
    return int_peak, int_peak / NORMALIZE, math.sqrt(sum_sq / len(seg)) / NORMALIZE


def _scan(path):
    """流式扫描一个 WAV，返回 (元信息 dict, 事件列表, 静音列表)。"""
    with open(path, 'rb') as fp:
        rate, data_pos, data_size = _parse_chunks(fp)
        total_samples = data_size // 2
        frame_count = math.ceil(FRAMES_PER_SEC * total_samples / rate)

        events = []
        silences = []

        # 事件状态机
        in_event = False
        ev_start = 0
        ev_active = 0
        ev_peak = 0
        ev_miss = 0

        # 静音状态机：sil_peak 只统计到最后一个静音帧；
        # 未确认并入的非静音毛刺单独记账，静音恢复时才并入。
        in_silence = False
        sil_start = 0
        sil_last = 0
        sil_peak = 0
        sil_glitch = 0
        sil_glitch_peak = 0

        # 基线窗口：当前帧之前最多 100 帧的峰值
        recent = deque()
        recent_sorted = []

        def close_event():
            if ev_active + 1 - ev_start >= EVENT_MIN_FRAMES:
                events.append({
                    'start_ms': ev_start * FRAME_MS,
                    'end_ms': (ev_active + 1) * FRAME_MS,
                    'peak': ev_peak,
                })

        def close_silence():
            if sil_last + 1 - sil_start >= SILENCE_MIN_FRAMES:
                silences.append({
                    'start_ms': sil_start * FRAME_MS,
                    'end_ms': (sil_last + 1) * FRAME_MS,
                    'peak': sil_peak,
                })

        def feed(frame, int_peak, peak, rms):
            nonlocal in_event, ev_start, ev_active, ev_peak, ev_miss
            nonlocal in_silence, sil_start, sil_last, sil_peak
            nonlocal sil_glitch, sil_glitch_peak

            # 基线只看当前帧之前的窗口
            m = len(recent_sorted)
            baseline = recent_sorted[(m - 1) // 4] if m else 0.0

            # 事件状态机
            if not in_event:
                if peak >= max(ABSOLUTE_GATE, BASELINE_FACTOR * baseline):
                    in_event = True
                    ev_start = frame
                    ev_active = frame
                    ev_peak = int_peak
                    ev_miss = 0
            elif peak >= CONTINUE_GATE:
                ev_active = frame
                if int_peak > ev_peak:
                    ev_peak = int_peak
                ev_miss = 0
            else:
                ev_miss += 1
                if ev_miss >= EVENT_MISS_FRAMES:
                    close_event()
                    in_event = False

            # 静音状态机
            if rms <= SILENCE_RMS:
                if in_silence:
                    if sil_glitch:
                        if sil_glitch_peak > sil_peak:
                            sil_peak = sil_glitch_peak
                        sil_glitch = 0
                        sil_glitch_peak = 0
                    sil_last = frame
                    if int_peak > sil_peak:
                        sil_peak = int_peak
                else:
                    in_silence = True
                    sil_start = frame
                    sil_last = frame
                    sil_peak = int_peak
                    sil_glitch = 0
                    sil_glitch_peak = 0
            elif in_silence:
                sil_glitch += 1
                if int_peak > sil_glitch_peak:
                    sil_glitch_peak = int_peak
                if sil_glitch >= SILENCE_MISS_FRAMES:
                    close_silence()
                    in_silence = False

            # 当前帧处理完后才进入基线窗口
            if len(recent) == BASELINE_FRAMES:
                old = recent.popleft()
                del recent_sorted[bisect.bisect_left(recent_sorted, old)]
            recent.append(peak)
            bisect.insort(recent_sorted, peak)

        fp.seek(data_pos)
        remain = data_size
        carry = b''          # 上一块未凑成整帧的尾巴（< 一帧多一点）
        base_sample = 0      # carry 起点的全局采样下标
        frame = 0

        while True:
            if remain > 0:
                want = min(READ_CHUNK_BYTES, remain)
                raw = fp.read(want)
                if not raw:
                    raise WavCorruptError('data 块在文件结束前截断')
                remain -= len(raw)
                buf = array('h')
                buf.frombytes(carry + raw)
            elif carry:
                buf = array('h')
                buf.frombytes(carry)
            else:
                break

            avail = base_sample + len(buf)
            lo, hi = _frame_span(frame, rate)
            while hi <= avail and frame < frame_count:
                feed(frame, *_frame_stats(buf[lo - base_sample:hi - base_sample]))
                frame += 1
                lo, hi = _frame_span(frame, rate)

            if remain == 0:
                # 末帧允许短于 10 ms
                end_sample = min(hi, total_samples)
                if frame < frame_count and lo < end_sample <= avail:
                    feed(frame, *_frame_stats(buf[lo - base_sample:end_sample - base_sample]))
                    frame += 1
                carry = b''
                break

            consumed = lo - base_sample
            tail = buf[consumed:]
            carry = bytes(tail)
            base_sample += consumed

        if in_event:
            close_event()
        if in_silence:
            close_silence()

        meta = {
            'sample_rate': rate,
            'channels': 1,
            'bits_per_sample': 16,
            'samples': total_samples,
            'duration_ms': math.ceil(1000 * total_samples / rate),
        }
        return meta, events, silences


def analyze(path):
    """分析单个 WAV，返回一个文件条目 dict。失败抛 OSError / Wav*Error。"""
    meta, events, silences = _scan(path)
    return {
        'file': os.path.basename(os.fspath(path)),
        'sample_rate': meta['sample_rate'],
        'channels': meta['channels'],
        'bits_per_sample': meta['bits_per_sample'],
        'samples': meta['samples'],
        'duration_ms': meta['duration_ms'],
        'events': events,
        'silences': silences,
    }


def _expand_inputs(paths):
    """展开文件/目录（目录不递归、只取 *.wav），按绝对路径去重，保持顺序。"""
    result = []
    seen = set()

    def add(p):
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            result.append(p)

    for path in paths:
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.lower().endswith('.wav'):
                    add(os.path.join(path, name))
        else:
            add(path)
    return result


def analyze_many(paths, on_error=None):
    """分析多个路径，返回完整报告 dict。

    on_error(path, message) 在某个文件失败时被调用；失败文件不进报告。
    """
    entries = []
    for path in _expand_inputs(paths):
        display = os.path.basename(os.fspath(path).rstrip(os.sep)) or os.fspath(path)
        try:
            entries.append(analyze(path))
        except (OSError, WavFormatError, WavCorruptError) as exc:
            if on_error is not None:
                on_error(display, str(exc))
    entries.sort(key=lambda entry: entry['file'])
    return {'version': VERSION, 'files': entries}


def _parse_args(argv):
    """命令行：analyze --input <路径>… [--input ...] --out <报告 JSON>"""
    out = os.path.join('var', 'report.json')
    inputs = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == 'analyze':
            i += 1
        elif arg == '--input':
            if i + 1 >= len(argv):
                return None, '缺少 --input 的参数'
            inputs.append(argv[i + 1])
            i += 2
        elif arg.startswith('--input='):
            inputs.append(arg[len('--input='):])
            i += 1
        elif arg == '--out':
            if i + 1 >= len(argv):
                return None, '缺少 --out 的参数'
            out = argv[i + 1]
            i += 2
        elif arg.startswith('--out='):
            out = arg[len('--out='):]
            i += 1
        else:
            return None, '无法识别的参数: %s' % arg
    if not inputs:
        return None, '缺少 --input 参数'
    return (inputs, out), None


def main(argv=None):
    parsed, error = _parse_args(sys.argv[1:] if argv is None else argv)
    if error:
        print('error: 命令行: %s' % error, file=sys.stderr)
        return 2

    inputs, out = parsed
    failures = []

    def on_error(name, message):
        failures.append((name, message))
        print('error: %s: %s' % (name, message), file=sys.stderr)

    report = analyze_many(inputs, on_error=on_error)

    out_dir = os.path.dirname(out)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            print('error: %s: %s' % (out, exc), file=sys.stderr)
            return 2
    try:
        with open(out, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
            fp.write('\n')
    except OSError as exc:
        print('error: %s: %s' % (out, exc), file=sys.stderr)
        return 2
    return 2 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
