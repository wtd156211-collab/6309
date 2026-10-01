const AUDIO_DIR = '../samples/audio/';
const LABEL_DIR = '../samples/labels/';

const WIDTH = 1000;
const HEIGHT = 150;
const MID = HEIGHT / 2;
const AMP = HEIGHT / 2 - 4;
const SVG_NS = 'http://www.w3.org/2000/svg';

const reportInput = document.getElementById('report-path');
const loadButton = document.getElementById('load');
const verdict = document.getElementById('verdict');
const filesContainer = document.getElementById('files');

function fourcc(view, offset) {
  return String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1),
                             view.getUint8(offset + 2), view.getUint8(offset + 3));
}

function parseWav(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 12 || fourcc(view, 0) !== 'RIFF' || fourcc(view, 8) !== 'WAVE') {
    throw new Error('not a RIFF/WAVE file');
  }
  let offset = 12;
  let dataOffset = 0;
  let dataLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = fourcc(view, offset);
    const size = view.getUint32(offset + 4, true);
    const payload = offset + 8;
    if (payload + size > view.byteLength) break;
    if (id === 'data') {
      dataOffset = payload;
      dataLength = size;
    }
    offset = payload + size + (size & 1);
  }
  if (!dataLength) throw new Error('missing data chunk');
  const count = Math.floor(dataLength / 2);
  return new Int16Array(buffer, dataOffset, count);
}

async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(String(response.status));
  return response.arrayBuffer();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(String(response.status));
  return response.json();
}

function envelopePath(samples) {
  const total = samples.length;
  const tops = [];
  const bottoms = [];
  for (let x = 0; x < WIDTH; x++) {
    const start = Math.floor(x * total / WIDTH);
    const end = Math.max(start + 1, Math.floor((x + 1) * total / WIDTH));
    let low = Infinity;
    let high = -Infinity;
    for (let i = start; i < end && i < total; i++) {
      const value = samples[i];
      if (value < low) low = value;
      if (value > high) high = value;
    }
    if (low === Infinity) {
      low = 0;
      high = 0;
    }
    tops.push(x + ',' + (MID - (high / 32768) * AMP).toFixed(2));
    bottoms.push(x + ',' + (MID - (low / 32768) * AMP).toFixed(2));
  }
  return 'M' + tops.join(' L ') + ' L ' + bottoms.reverse().join(' L ') + ' Z';
}

function addRect(svg, className, interval, durationMs) {
  const x1 = Math.round(interval.start_ms * WIDTH / durationMs);
  const x2 = Math.round(interval.end_ms * WIDTH / durationMs);
  const rect = document.createElementNS(SVG_NS, 'rect');
  rect.setAttribute('class', className);
  rect.setAttribute('x', x1);
  rect.setAttribute('y', 0);
  rect.setAttribute('width', Math.max(0, x2 - x1));
  rect.setAttribute('height', HEIGHT);
  rect.setAttribute('data-start-ms', interval.start_ms);
  rect.setAttribute('data-end-ms', interval.end_ms);
  svg.appendChild(rect);
}

function overlap(a, b) {
  return Math.max(0, Math.min(a.end_ms, b.end_ms) - Math.max(a.start_ms, b.start_ms));
}

function pairUp(labels, outputs) {
  const used = new Array(outputs.length).fill(false);
  let misses = 0;
  for (const label of labels) {
    let best = -1;
    let bestOverlap = 0;
    const labelLength = label.end_ms - label.start_ms;
    for (let i = 0; i < outputs.length; i++) {
      if (used[i]) continue;
      const output = outputs[i];
      const shared = overlap(label, output);
      const required = Math.min(labelLength, output.end_ms - output.start_ms) / 2;
      if (shared >= required && shared > bestOverlap) {
        best = i;
        bestOverlap = shared;
      }
    }
    if (best >= 0) used[best] = true;
    else misses++;
  }
  return { misses, falseAlarms: used.filter((u) => !u).length };
}

function renderFile(entry, labels, samples) {
  const block = document.createElement('div');
  block.className = 'file-block';

  const head = document.createElement('div');
  head.className = 'file-head';
  head.innerHTML = '<b></b> <span class="meta"></span>';
  head.querySelector('b').textContent = entry.file;
  head.querySelector('.meta').textContent =
    '时长 ' + entry.duration_ms + ' ms · 事件 ' + entry.events.length +
    ' 段 · 静音 ' + entry.silences.length + ' 段';
  block.appendChild(head);

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'scope');
  svg.setAttribute('viewBox', '0 0 ' + WIDTH + ' ' + HEIGHT);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('data-file', entry.file);

  const durationMs = Math.max(1, entry.duration_ms);
  for (const silence of entry.silences) addRect(svg, 'silence', silence, durationMs);
  for (const event of entry.events) addRect(svg, 'event', event, durationMs);

  if (samples && samples.length) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('class', 'wave');
    path.setAttribute('d', envelopePath(samples));
    svg.appendChild(path);
  }

  if (labels) {
    for (const silence of labels.silences) addRect(svg, 'mark-silence', silence, durationMs);
    for (const event of labels.events) addRect(svg, 'mark-event', event, durationMs);
  }

  block.appendChild(svg);
  filesContainer.appendChild(block);
}

async function loadReport() {
  const reportPath = reportInput.value.trim();
  filesContainer.textContent = '';
  verdict.classList.remove('zero', 'bad');
  verdict.textContent = '加载中…';

  let report;
  try {
    report = await fetchJson(reportPath);
  } catch (err) {
    verdict.classList.add('bad');
    verdict.textContent = '无法加载报告：' + reportPath + '（' + err.message + '）';
    return;
  }

  let totalMisses = 0;
  let totalFalseAlarms = 0;

  for (const entry of report.files || []) {
    let labels = null;
    let samples = null;
    const stem = entry.file.replace(/\.wav$/, '');
    try {
      labels = await fetchJson(LABEL_DIR + stem + '.json');
    } catch (err) {
      labels = null;
    }
    try {
      samples = parseWav(await fetchBuffer(AUDIO_DIR + entry.file));
    } catch (err) {
      samples = null;
    }

    renderFile(entry, labels, samples);

    if (labels) {
      const eventStats = pairUp(labels.events || [], entry.events || []);
      const silenceStats = pairUp(labels.silences || [], entry.silences || []);
      totalMisses += eventStats.misses + silenceStats.misses;
      totalFalseAlarms += eventStats.falseAlarms + silenceStats.falseAlarms;
    }
  }

  verdict.textContent = '漏检 ' + totalMisses + ' 个，误报 ' + totalFalseAlarms + ' 个';
  verdict.classList.add(totalMisses === 0 && totalFalseAlarms <= 1 ? 'zero' : 'bad');
}

loadButton.addEventListener('click', loadReport);
reportInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') loadReport();
});
window.addEventListener('DOMContentLoaded', loadReport);
