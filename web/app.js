// 波形复核页：区间与统计只来自引擎报告与人工标注，页面不重算检测。

const WIDTH = 1000;
const HEIGHT = 120;
const MID = HEIGHT / 2;
const AMP = HEIGHT / 2 - 4;
const KINDS = ['events', 'silences'];

const $ = (id) => document.getElementById(id);

function svgEl(tag, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

// 毫秒 -> SVG 横坐标，线性取整
function xOf(ms, durationMs) {
  return Math.round(ms * WIDTH / durationMs);
}

function widthOf(startMs, endMs, durationMs) {
  return Math.max(1, xOf(endMs, durationMs) - xOf(startMs, durationMs));
}

function overlap(startA, endA, startB, endB) {
  return Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
}

// 配对：重叠长度 >= 两者较短时长的 50%。贪心取最大重叠，返回 [漏检数, 误报数]。
function compareSpans(labels, outputs) {
  const usedOutputs = new Set();
  let misses = 0;
  for (const label of labels) {
    let best = -1;
    let bestOverlap = 0;
    for (let i = 0; i < outputs.length; i += 1) {
      if (usedOutputs.has(i)) continue;
      const out = outputs[i];
      const ov = overlap(label.start_ms, label.end_ms, out.start_ms, out.end_ms);
      const need = 0.5 * Math.min(label.end_ms - label.start_ms,
                                  out.end_ms - out.start_ms);
      if (ov >= need && ov > bestOverlap) {
        best = i;
        bestOverlap = ov;
      }
    }
    if (best >= 0) {
      usedOutputs.add(best);
    } else {
      misses += 1;
    }
  }
  return [misses, outputs.length - usedOutputs.size];
}

// 读取 16 位单声道 PCM，返回 Int16Array（按块定位，跳过 LIST 等杂块）
async function loadWav(url) {
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  const dv = new DataView(buf);
  const sig = (off) => String.fromCharCode(...new Uint8Array(buf, off, 4));
  if (sig(0) !== 'RIFF' || sig(8) !== 'WAVE') {
    throw new Error('不是 RIFF/WAVE 文件');
  }
  let off = 12;
  let dataBytes = null;
  while (off + 8 <= buf.byteLength) {
    const id = sig(off);
    const size = dv.getUint32(off + 4, true);
    if (id === 'data') {
      dataBytes = new DataView(buf, off + 8, size);
      break;
    }
    off += 8 + size + (size & 1);
  }
  if (!dataBytes) throw new Error('缺少 data 块');
  const samples = new Int16Array(dataBytes.byteLength / 2);
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = dataBytes.getInt16(i * 2, true);
  }
  return samples;
}

// 1000 个桶取峰谷包络，返回闭合 path
function envelopePath(samples) {
  const buckets = new Array(WIDTH);
  for (let i = 0; i < WIDTH; i += 1) buckets[i] = [32768, -32768];
  for (let i = 0; i < samples.length; i += 1) {
    const b = Math.min(WIDTH - 1, Math.floor(i * WIDTH / samples.length));
    const v = samples[i];
    if (v < buckets[b][0]) buckets[b][0] = v;
    if (v > buckets[b][1]) buckets[b][1] = v;
  }
  let d = '';
  for (let x = 0; x < WIDTH; x += 1) {
    d += `${x === 0 ? 'M' : 'L'}${x},${(MID - buckets[x][1] / 32768 * AMP).toFixed(2)} `;
  }
  for (let x = WIDTH - 1; x >= 0; x -= 1) {
    d += `L${x},${(MID - buckets[x][0] / 32768 * AMP).toFixed(2)} `;
  }
  return `${d}Z`;
}

function addRect(svg, cls, span, durationMs) {
  svg.appendChild(svgEl('rect', {
    class: cls,
    x: xOf(span.start_ms, durationMs),
    width: widthOf(span.start_ms, span.end_ms, durationMs),
    y: 0,
    height: HEIGHT,
    'data-start-ms': span.start_ms,
    'data-end-ms': span.end_ms,
  }));
}

async function renderFile(entry, container) {
  const block = document.createElement('div');
  block.className = 'file-block';

  const title = document.createElement('div');
  title.className = 'file-title';
  title.textContent = entry.file;
  block.appendChild(title);

  const svg = svgEl('svg', {
    class: 'scope',
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    preserveAspectRatio: 'none',
    'data-file': entry.file,
  });

  for (const span of entry.silences) addRect(svg, 'silence', span, entry.duration_ms);
  for (const span of entry.events) addRect(svg, 'event', span, entry.duration_ms);

  // 波形数据来自原 WAV，仅用于绘制；检测区间全部来自报告
  try {
    const samples = await loadWav(`../samples/audio/${entry.file}`);
    svg.appendChild(svgEl('path', { class: 'wave', d: envelopePath(samples) }));
  } catch (err) {
    const note = document.createElement('span');
    note.className = 'note';
    note.textContent = `波形加载失败：${err.message}`;
    title.appendChild(note);
  }

  let misses = 0;
  let falseAlarms = 0;
  try {
    const label = await fetch(`../samples/labels/${entry.file.replace(/\.wav$/i, '.json')}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
    for (const span of label.silences) addRect(svg, 'mark-silence', span, entry.duration_ms);
    for (const span of label.events) addRect(svg, 'mark-event', span, entry.duration_ms);
    for (const kind of KINDS) {
      const [m, fa] = compareSpans(label[kind], entry[kind]);
      misses += m;
      falseAlarms += fa;
    }
  } catch (err) {
    const note = document.createElement('span');
    note.className = 'note';
    note.textContent = `标注加载失败：${err.message}`;
    title.appendChild(note);
  }

  block.appendChild(svg);
  container.appendChild(block);
  return [misses, falseAlarms];
}

async function loadReport() {
  const path = $('report-path').value.trim();
  const container = $('files');
  const verdict = $('verdict');
  const status = $('status');
  container.innerHTML = '';
  verdict.textContent = '';
  verdict.className = '';
  status.textContent = '加载中…';

  let report;
  try {
    report = await fetch(`${path}?t=${Date.now()}`).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
  } catch (err) {
    status.textContent = `报告加载失败：${err.message}`;
    return;
  }

  let totalMisses = 0;
  let totalFalseAlarms = 0;
  for (const entry of report.files) {
    const [m, fa] = await renderFile(entry, container);
    totalMisses += m;
    totalFalseAlarms += fa;
  }
  status.textContent = `共 ${report.files.length} 个文件`;
  verdict.textContent = `漏检 ${totalMisses} 个　误报 ${totalFalseAlarms} 个`;
  verdict.className = (totalMisses === 0 && totalFalseAlarms <= 1) ? 'ok' : 'bad';
}

$('load').addEventListener('click', loadReport);
$('report-path').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') loadReport();
});
loadReport();
