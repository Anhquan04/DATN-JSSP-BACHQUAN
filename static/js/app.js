/**
 * app.js — Frontend logic cho Job-Shop Scheduling Visualization
 * =============================================================
 * Giao tiếp với Flask backend qua fetch() API calls.
 * Render Gantt chart bằng HTML5 Canvas.
 *
 * ĐATN 2026 | Bạch Công Quân
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const JOB_COLORS = [
  '#4f8ef7','#34d399','#fb923c','#f472b6',
  '#a78bfa','#fbbf24','#38bdf8','#f87171','#86efac','#c084fc'
];
const ALGO_COLORS = {
  a2c:'#4f8ef7', fifo:'#fb923c', spt:'#34d399', lpt:'#a78bfa', edd:'#f87171'
};
const ALGO_NAMES = {
  a2c:'A2C (Actor-Critic)', fifo:'FIFO', spt:'SPT', lpt:'LPT', edd:'EDD'
};

// ── State ─────────────────────────────────────────────────────────────────────
let currentInstance  = '3x3';
let currentAlgo      = 'a2c';
let compareMode      = false;
let scheduleCache    = {};      // { 'ft06/a2c': {...}, ... }
let compareDataCache = {};      // { 'ft06': {...}, ... }

let animTime  = 0;
let playing   = false;
let speedMult = 5;
let animFrame = null;
let lastTs    = null;
let currentData = null;         // schedule data đang hiển thị

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await loadInstances();
  await loadSchedule();           // Load 3x3 / a2c mặc định
});

// ── Instance Loading ──────────────────────────────────────────────────────────
async function loadInstances() {
  const res  = await fetch('/api/instances');
  const data = await res.json();
  const list = document.getElementById('instanceList');
  list.innerHTML = '';

  const instOrder = ['3x3','4x4','5x5','ft06','ft10'];
  instOrder.forEach(key => {
    if (!data[key]) return;
    const info = data[key];
    const btn  = document.createElement('button');
    btn.className = 'instance-btn' + (key === currentInstance ? ' active' : '');
    btn.setAttribute('data-inst', key);
    btn.onclick = () => selectInstance(key);
    btn.innerHTML = `
      <span>
        ${info.trained ? '<span class="trained-dot" title="Model đã train"></span>' : ''}
        ${info.label}
      </span>
      <span class="inst-meta">${info.n_jobs}J&times;${info.n_machines}M</span>
    `;
    list.appendChild(btn);
  });
}

// ── Selectors ─────────────────────────────────────────────────────────────────
async function selectInstance(name) {
  currentInstance = name;
  document.querySelectorAll('.instance-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-inst') === name);
  });
  resetAnim();
  await loadSchedule();
}

async function selectAlgo(algo) {
  currentAlgo = algo;
  document.querySelectorAll('.algo-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-algo') === algo);
  });
  document.getElementById('algoLabel').textContent = ALGO_NAMES[algo] || algo.toUpperCase();
  if (compareMode) return;
  resetAnim();
  await loadSchedule();
}

async function toggleCompare() {
  compareMode = !compareMode;
  document.getElementById('compareBtn').classList.toggle('active', compareMode);
  document.getElementById('ganttWrapper').style.display = compareMode ? 'none' : '';
  document.getElementById('compareGrid').style.display  = compareMode ? 'grid' : 'none';
  resetAnim();

  if (compareMode) {
    showToast('🔀 Đang tải dữ liệu so sánh...');
    await loadCompare();
  } else {
    await loadSchedule();
  }
}

// ── Data Loading ──────────────────────────────────────────────────────────────
async function loadSchedule() {
  const cacheKey = `${currentInstance}/${currentAlgo}`;
  showLoading(true);

  try {
    if (!scheduleCache[cacheKey]) {
      const res = await fetch(`/api/schedule/${currentInstance}/${currentAlgo}`);
      if (!res.ok) throw new Error(await res.text());
      scheduleCache[cacheKey] = await res.json();
    }
    currentData = scheduleCache[cacheKey];
    updateLegend(currentData.instance.n_jobs);
    updateGanttTitle();
    updateMetrics(0);
    drawFrame(0);
    showToast(`✅ ${ALGO_NAMES[currentAlgo]} — ${currentInstance.toUpperCase()}`);
  } catch (e) {
    showToast('❌ Lỗi tải dữ liệu: ' + e.message);
    console.error(e);
  } finally {
    showLoading(false);
  }
}

async function loadCompare() {
  const cacheKey = currentInstance;
  showLoading(true);

  try {
    if (!compareDataCache[cacheKey]) {
      const res = await fetch(`/api/compare/${currentInstance}`);
      if (!res.ok) throw new Error(await res.text());
      compareDataCache[cacheKey] = await res.json();
    }
    const data = compareDataCache[cacheKey];
    updateLegend(data.instance.n_jobs);
    buildCompareGrid(data);
    drawCompare(0, data);
    showToast(`✅ So sánh ${currentInstance.toUpperCase()} — ${Object.keys(data.results).length} algorithms`);
  } catch (e) {
    showToast('❌ Lỗi: ' + e.message);
    console.error(e);
  } finally {
    showLoading(false);
  }
}

// ── Animation ─────────────────────────────────────────────────────────────────
function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('playBtn');
  btn.textContent  = playing ? '⏸ Pause' : '▶ Play';
  btn.className    = 'ctrl-btn ' + (playing ? 'pause' : 'play');
  if (playing) { lastTs = null; animFrame = requestAnimationFrame(animate); }
  else         { cancelAnimationFrame(animFrame); }
}

function resetAnim() {
  playing = false;
  cancelAnimationFrame(animFrame);
  lastTs   = null;
  animTime = 0;
  const btn = document.getElementById('playBtn');
  btn.textContent = '▶ Play'; btn.className = 'ctrl-btn play';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('timeLabel').textContent    = 't = 0';
  drawFrame(0);
  updateMetrics(0);
}

function animate(ts) {
  if (!lastTs) lastTs = ts;
  const dt = (ts - lastTs) / 1000;
  lastTs   = ts;
  animTime += dt * speedMult * 10;

  const maxT = getMaxTime();
  if (animTime >= maxT) {
    animTime = maxT;
    playing  = false;
    const btn = document.getElementById('playBtn');
    btn.textContent = '▶ Play'; btn.className = 'ctrl-btn play';
    showToast('✅ Hoàn thành lịch sản xuất!');
  }

  drawFrame(animTime);
  updateMetrics(animTime);
  const pct = maxT > 0 ? (animTime / maxT * 100).toFixed(1) : 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('timeLabel').textContent    = 't = ' + Math.floor(animTime);

  if (playing) animFrame = requestAnimationFrame(animate);
  else lastTs = null;
}

function getMaxTime() {
  if (compareMode && compareDataCache[currentInstance]) {
    const results = compareDataCache[currentInstance].results;
    return Math.max(...Object.values(results).map(r => r.makespan));
  }
  return currentData ? currentData.makespan : 1;
}

// ── Drawing ───────────────────────────────────────────────────────────────────
const ROW_H = 44, PAD_L = 74, PAD_R = 18, PAD_T = 10, PAD_B = 28;

function drawFrame(t) {
  if (compareMode) {
    const data = compareDataCache[currentInstance];
    if (data) drawCompare(t, data);
    return;
  }
  if (!currentData) return;
  const canvas = document.getElementById('mainCanvas');
  drawGantt(canvas, currentData.schedule, currentData.instance.n_machines,
            currentData.makespan, t);
}

function drawGantt(canvas, schedule, nMachines, makespan, t) {
  const W = canvas.parentElement.clientWidth || 800;
  const H = PAD_T + nMachines * ROW_H + PAD_B;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const sx = v => PAD_L + (v / makespan) * (W - PAD_L - PAD_R);

  // Background rows
  for (let m = 0; m < nMachines; m++) {
    const y = PAD_T + m * ROW_H;
    ctx.fillStyle = m % 2 === 0 ? '#141727' : '#181b2c';
    ctx.fillRect(PAD_L, y, W - PAD_L - PAD_R, ROW_H);
  }

  // Grid lines & time labels
  const nTicks = Math.min(makespan, 10);
  for (let i = 0; i <= nTicks; i++) {
    const tv = Math.round(i * makespan / nTicks);
    const x  = sx(tv);
    ctx.strokeStyle = '#2d3748'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, H - PAD_B); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText(tv, x, H - PAD_B + 14);
  }

  // Machine labels
  for (let m = 0; m < nMachines; m++) {
    const y = PAD_T + m * ROW_H + ROW_H / 2;
    ctx.fillStyle = '#94a3b8'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText('M' + m, PAD_L - 8, y + 4);
  }

  // Job blocks
  schedule.forEach(({ job, machine, start, end }) => {
    const visEnd = Math.min(end, t);
    if (visEnd <= start) return;

    const x = sx(start), w = sx(visEnd) - sx(start);
    const y = PAD_T + machine * ROW_H + 5, h = ROW_H - 10;
    const col = JOB_COLORS[job % JOB_COLORS.length];

    // Glow effect
    ctx.shadowColor = col; ctx.shadowBlur = 8;
    ctx.fillStyle   = col;
    rrect(ctx, x, y, w, h, 5);
    ctx.shadowBlur  = 0;

    // Border
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
    rrectStroke(ctx, x, y, w, h, 5);

    // Label
    if (w > 22) {
      ctx.fillStyle = '#000'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('J' + job, x + w / 2, y + h / 2 + 4);
    }

    // Shimmer at leading edge (active block)
    if (end > t && t > start) {
      const g = ctx.createLinearGradient(sx(visEnd) - 14, 0, sx(visEnd), 0);
      g.addColorStop(0, 'transparent');
      g.addColorStop(1, 'rgba(255,255,255,.45)');
      ctx.fillStyle = g;
      rrect(ctx, x, y, w, h, 5);
    }
  });

  // Current time line
  if (t > 0 && t < makespan) {
    const tx = sx(t);
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
    ctx.beginPath(); ctx.moveTo(tx, PAD_T); ctx.lineTo(tx, H - PAD_B); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f87171'; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'left';
    ctx.fillText('t=' + Math.floor(t), tx + 4, PAD_T + 11);
  }

  // Makespan marker
  if (t >= makespan) {
    const mx = sx(makespan);
    ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(mx, PAD_T); ctx.lineTo(mx, H - PAD_B); ctx.stroke();
    ctx.fillStyle = '#34d399'; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'right';
    ctx.fillText('Makespan = ' + makespan, mx - 5, PAD_T + 13);
  }
}

// Compare mode
function buildCompareGrid(data) {
  const grid = document.getElementById('compareGrid');
  grid.innerHTML = '';

  // ✅ Fixed order: A2C trước, rồi 4 baselines — đảm bảo render NHẤT QUÁN
  const ORDER = ['a2c', 'fifo', 'spt', 'lpt', 'edd'];
  const algos = ORDER.filter(a => data.results[a]);

  // ✅ Tìm best & worst để highlight
  const allMs = algos.map(a => data.results[a].makespan);
  const minMs = Math.min(...allMs);
  const maxMs = Math.max(...allMs);

  algos.forEach((algo, idx) => {
    const r  = data.results[algo];
    const ms = r.makespan;
    const isBest  = ms === minMs;
    const isWorst = ms === maxMs && ms !== minMs;
    const badge   = isBest  ? '<span style="background:rgba(52,211,153,.2);color:#34d399;padding:2px 8px;border-radius:99px;font-size:10px;font-weight:700">🏆 Tốt nhất</span>'
                  : isWorst ? '<span style="background:rgba(248,113,113,.2);color:#f87171;padding:2px 8px;border-radius:99px;font-size:10px;font-weight:700">Kém nhất</span>'
                  : '';

    const card = document.createElement('div');
    card.className = 'compare-card';
    card.innerHTML = `
      <div class="compare-card-header">
        <div class="algo-dot" style="background:${ALGO_COLORS[algo]||'#aaa'};width:12px;height:12px;border-radius:50%"></div>
        <div class="compare-card-name">
          ${idx + 1}. ${ALGO_NAMES[algo] || algo.toUpperCase()}
        </div>
        ${badge}
        <div class="compare-card-ms" id="cmp_ms_${algo}" style="color:${ALGO_COLORS[algo]||'#aaa'}">
          MS = ${ms}
        </div>
      </div>
      <canvas id="cmp_canvas_${algo}"></canvas>
    `;
    grid.appendChild(card);
  });

  console.log(`[Compare] Rendered ${algos.length} algorithms:`, algos);
}

function drawCompare(t, data) {
  const algos = Object.keys(data.results);
  const nM    = data.instance.n_machines;
  algos.forEach(algo => {
    const cv = document.getElementById('cmp_canvas_' + algo);
    if (!cv) return;
    const res = data.results[algo];
    const W   = cv.parentElement.clientWidth - 24 || 400;
    cv.width  = W;
    drawGantt(cv, res.schedule, nM, res.makespan, t);
  });
}

// ── Metrics ───────────────────────────────────────────────────────────────────
function updateMetrics(t) {
  const d = compareMode
    ? (compareDataCache[currentInstance]?.results?.a2c || null)
    : currentData;
  if (!d) return;

  const ms  = d.makespan;
  const pct = Math.min(t / ms, 1);

  document.getElementById('metMakespan').textContent = Math.min(Math.floor(t), ms);
  document.getElementById('metUtil').textContent =
    pct < 1 ? (pct * d.utilization * 100).toFixed(1) + '%'
             : (d.utilization * 100).toFixed(1) + '%';
  document.getElementById('metIdle').textContent =
    pct >= 1 ? d.idle_time.toFixed(0) : '…';

  // vs FIFO — compare against cached FIFO schedule
  const fifoKey = `${currentInstance}/fifo`;
  if (scheduleCache[fifoKey] || compareDataCache[currentInstance]) {
    const fifoMs = compareDataCache[currentInstance]?.results?.fifo?.makespan
                || scheduleCache[fifoKey]?.makespan;
    if (fifoMs) {
      const chg  = ((fifoMs - ms) / fifoMs * 100).toFixed(1);
      const el   = document.getElementById('metVsFifo');
      const sub  = document.getElementById('metVsFifoSub');
      el.textContent  = parseFloat(chg) > 0 ? '+' + chg + '%' : chg + '%';
      el.style.color  = parseFloat(chg) > 0 ? '#34d399' : '#f87171';
      sub.textContent = `FIFO = ${fifoMs} | ${ALGO_NAMES[currentAlgo]?.split(' ')[0]} = ${ms}`;
    }
  }
}

// ── Gantt Title ───────────────────────────────────────────────────────────────
function updateGanttTitle() {
  const inst = currentInstance.toUpperCase();
  const algo = ALGO_NAMES[currentAlgo] || currentAlgo.toUpperCase();
  document.getElementById('ganttTitle').textContent = compareMode
    ? `So sánh tất cả Algorithms — ${inst}`
    : `${algo} — Instance ${inst}`;
  document.getElementById('ganttBadge').textContent = currentData
    ? `Makespan = ${currentData.makespan}`
    : '';
  document.getElementById('algoLabel').textContent = compareMode ? 'Compare' : (ALGO_NAMES[currentAlgo] || currentAlgo);
}

// ── Legend ────────────────────────────────────────────────────────────────────
function updateLegend(nJobs) {
  const cont = document.getElementById('legendContainer');
  cont.innerHTML = '';
  for (let j = 0; j < nJobs; j++) {
    const div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = `<div class="legend-box" style="background:${JOB_COLORS[j % JOB_COLORS.length]}"></div> Job ${j}`;
    cont.appendChild(div);
  }
}

// ── Loading overlay ───────────────────────────────────────────────────────────
function showLoading(on) {
  document.getElementById('loadingOverlay').classList.toggle('hidden', !on);
}

// ── Train Modal ───────────────────────────────────────────────────────────────
function openTrainModal() {
  document.getElementById('trainModal').classList.add('open');
  document.getElementById('trainProgress').style.display = 'none';
  document.getElementById('trainStartBtn').disabled = false;
  document.getElementById('trainStartBtn').textContent = '🚀 Bắt đầu Train';
}

function closeTrainModal(e) {
  if (e && e.target !== e.currentTarget) return;
  document.getElementById('trainModal').classList.remove('open');
}

let trainPoller = null;
async function startTraining() {
  const instance  = document.getElementById('trainInstance').value;
  const episodes  = parseInt(document.getElementById('trainEpisodes').value);
  const startBtn  = document.getElementById('trainStartBtn');

  startBtn.disabled    = true;
  startBtn.textContent = '⏳ Đang train...';
  document.getElementById('trainProgress').style.display = 'block';
  document.getElementById('trainLog').textContent = '';

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ instance, episodes }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast('❌ ' + (err.error || 'Lỗi training'));
      startBtn.disabled = false; startBtn.textContent = '🚀 Bắt đầu Train';
      return;
    }
    showToast(`🚀 Training ${instance} bắt đầu!`);
    trainPoller = setInterval(pollTrainStatus, 1000);
  } catch (e) {
    showToast('❌ Không thể kết nối server');
    startBtn.disabled = false; startBtn.textContent = '🚀 Bắt đầu Train';
  }
}

async function pollTrainStatus() {
  try {
    const res    = await fetch('/api/train/status');
    const status = await res.json();

    const pct  = status.total > 0 ? (status.episode / status.total * 100).toFixed(0) : 0;
    document.getElementById('trainProgressFill').style.width = pct + '%';
    document.getElementById('trainProgressPct').textContent  = pct + '%';
    document.getElementById('trainProgressText').textContent =
      status.running ? `Training ${status.instance} — Ep ${status.episode}/${status.total}`
                     : `Hoàn thành! Best makespan = ${status.best_ms}`;

    // Log
    const logEl = document.getElementById('trainLog');
    logEl.textContent = status.log.join('\n');
    logEl.scrollTop   = logEl.scrollHeight;

    if (!status.running) {
      clearInterval(trainPoller);
      document.getElementById('trainStartBtn').disabled = false;
      document.getElementById('trainStartBtn').textContent = '🚀 Bắt đầu Train';
      showToast(`✅ Training xong! Best MS = ${status.best_ms}`);
      // Reload instance list để cập nhật trained badge
      await loadInstances();
      // Xoá cache để reload data mới
      scheduleCache   = {};
      compareDataCache = {};
    }
  } catch (e) {
    clearInterval(trainPoller);
  }
}

// ── Canvas helpers ────────────────────────────────────────────────────────────
function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath(); ctx.fill();
}
function rrectStroke(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath(); ctx.stroke();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg; el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

// Resize redraw
window.addEventListener('resize', () => {
  if (!playing) drawFrame(animTime);
});

// ══════════════════════════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ══════════════════════════════════════════════════════════════════════════════
function switchTab(tab) {
  ['vizApp','evalPanel','realPanel'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  if (tab === 'viz') {
    document.getElementById('vizApp').style.display = 'flex';
    document.getElementById('tabViz')?.classList.add('active');
  } else if (tab === 'eval') {
    document.getElementById('evalPanel').style.display = 'flex';
    document.getElementById('tabEval')?.classList.add('active');
    if (!evalLoaded) loadEvaluation(false);
  } else if (tab === 'real') {
    document.getElementById('realPanel').style.display = 'flex';
    document.getElementById('tabReal')?.classList.add('active');
    const sl = document.getElementById('rtClassList');
    if (sl && sl.children.length === 0) renderRTConfig();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  EVALUATION TAB
// ══════════════════════════════════════════════════════════════════════════════
const ALGO_META_EVAL = {a2c:{label:'A2C (RL Agent)',color:'#4f8ef7'},fifo:{label:'FIFO',color:'#fb923c'},spt:{label:'SPT',color:'#34d399'},lpt:{label:'LPT',color:'#a78bfa'},edd:{label:'EDD',color:'#f87171'}};
const TIER_ICONS = {basic:'🟢',medium:'🟡',complex:'🔴'};
let evalLoaded = false;

async function loadEvaluation(forceRefresh=false) {
  const loading=document.getElementById('evalLoading'), result=document.getElementById('evalResult');
  loading.style.display='flex'; result.style.display='none';
  try {
    const res=await fetch('/api/evaluation/full'+(forceRefresh?'?refresh=true':''));
    if(!res.ok) throw new Error('Server error: '+res.status);
    renderEvaluation(await res.json()); evalLoaded=true; showToast('✅ Đánh giá hoàn thành!');
  } catch(e) {
    result.innerHTML=`<div style="color:var(--red);padding:40px;text-align:center">❌ ${e.message}</div>`;
    result.style.display='block'; showToast('❌ '+e.message);
  } finally { loading.style.display='none'; result.style.display='block'; }
}

function renderEvaluation(data) {
  const container=document.getElementById('evalResult'), tiers=data.tiers;
  let totalI=0,a2cW=0,sumI=0,cnt=0;
  Object.values(tiers).forEach(tier=>Object.values(tier.instances).forEach(inst=>{
    if(inst.error) return; totalI++; if(inst.best_algo==='a2c') a2cW++;
    sumI+=inst.improvement_vs_fifo; cnt++;
  }));
  const avgI=cnt>0?(sumI/cnt).toFixed(1):0;
  let html=`<div class="eval-summary">
    <div class="summary-card blue"><div class="summary-card-label">Instances đánh giá</div><div class="summary-card-value">${totalI}</div><div class="summary-card-sub">3 mức độ</div></div>
    <div class="summary-card green"><div class="summary-card-label">A2C đứng đầu</div><div class="summary-card-value">${a2cW}/${totalI}</div><div class="summary-card-sub">Best makespan</div></div>
    <div class="summary-card purple"><div class="summary-card-label">Cải thiện TB vs FIFO</div><div class="summary-card-value">${avgI>0?'+':''}${avgI}%</div><div class="summary-card-sub">TB tất cả</div></div>
  </div>`;
  const tDesc={basic:'3×3, 4×4 — verify thuật toán',medium:'5×5, FT06 — benchmark chuẩn',complex:'FT10 — bài toán thực tế'};
  const tTitle={basic:'Trường hợp cơ bản',medium:'Trường hợp trung bình',complex:'Trường hợp phức tạp'};
  ['basic','medium','complex'].forEach(tk=>{
    const tier=tiers[tk]; if(!tier) return;
    html+=`<div class="tier-section"><div class="tier-header">
      <span class="tier-badge" style="background:${tier.color}22;color:${tier.color};border:1px solid ${tier.color}55">${TIER_ICONS[tk]} ${tier.label}</span>
      <span class="tier-title">${tTitle[tk]}</span><span class="tier-desc">${tDesc[tk]}</span></div>
      <div class="instance-cards">${Object.entries(tier.instances).map(([n,d])=>renderInstanceCard(n,d)).join('')}</div></div>`;
  });
  container.innerHTML=html;
}

function renderInstanceCard(instName,instData) {
  if(instData.error) return `<div class="inst-card"><div class="inst-card-header"><span class="inst-card-name">${instName.toUpperCase()}</span></div><div style="color:var(--red);font-size:12px">❌ ${instData.error}</div></div>`;
  const {info,lower_bound,results,best_algo,improvement_vs_fifo}=instData;
  const algos=['a2c','fifo','spt','lpt','edd'];
  const mss=algos.map(a=>results[a]?.makespan||9999), minMS=Math.min(...mss), maxMS=Math.max(...mss);
  let html=`<div class="inst-card"><div class="inst-card-header">
    <div><div class="inst-card-name">${instName.toUpperCase()}</div><div class="inst-card-meta">${info.n_jobs}J × ${info.n_machines}M</div></div>
    <span class="inst-card-lb">LB=${lower_bound}</span>
    <span class="inst-best-badge">🏆 ${(ALGO_META_EVAL[best_algo]?.label||best_algo).split(' ')[0]}</span></div>
    <div style="margin-bottom:10px;padding:7px 10px;background:var(--surface2);border-radius:8px;font-size:12px;display:flex;justify-content:space-between">
      <span style="color:var(--muted)">A2C vs FIFO:</span>
      <span class="${improvement_vs_fifo>0?'improve-pos':'improve-neg'}">${improvement_vs_fifo>0?'▼':'▲'}${Math.abs(improvement_vs_fifo)}%</span>
      <span style="color:var(--muted)">Gap to LB:</span>
      <span style="color:var(--accent);font-weight:700">+${results.a2c?.gap_to_lb??'—'}%</span></div>
    <table class="cmp-table"><thead><tr><th>Algorithm</th><th>Makespan</th><th>Util</th><th>Idle</th><th>vs FIFO</th></tr></thead><tbody>`;
  algos.forEach(algo=>{
    const r=results[algo]; if(!r) return;
    const isBest=r.makespan===minMS, isWorst=r.makespan===maxMS;
    const msC=isBest?'cell-best':isWorst?'cell-worst':'cell-neutral';
    const fifoMs=results['fifo']?.makespan||1, vF=((fifoMs-r.makespan)/fifoMs*100).toFixed(1);
    const vFStr=algo==='fifo'?'—':`<span class="${parseFloat(vF)>0?'improve-pos':'improve-neg'}">${parseFloat(vF)>0?'▼':'▲'}${Math.abs(vF)}%</span>`;
    const bw=Math.round((1-(r.makespan-minMS)/(maxMS-minMS+1))*100);
    const meta=ALGO_META_EVAL[algo]||{label:algo,color:'#aaa'};
    const tCls=algo==='a2c'?(r.trained?'tag-trained':'tag-ai'):'tag-rule';
    html+=`<tr><td><div class="algo-name-cell"><div class="algo-color-dot" style="background:${meta.color}"></div>${meta.label.split(' ')[0]}<span class="cell-tag ${tCls}">${algo==='a2c'?(r.trained?'✓':'AI'):'Rule'}</span></div></td>
    <td><div class="bar-row"><div class="bar-wrap"><div class="bar-fill" style="width:${bw}%;background:${meta.color}"></div></div><span class="bar-val ${msC}">${r.makespan}</span></div></td>
    <td style="color:${r.utilization>60?'var(--green)':r.utilization>40?'var(--yellow)':'var(--red)'};font-weight:600;font-size:12px">${r.utilization}%</td>
    <td style="font-family:monospace;font-size:12px;color:var(--muted)">${r.idle_time}</td><td>${vFStr}</td></tr>`;
  });
  html+=`</tbody></table></div>`;
  return html;
}

// ══════════════════════════════════════════════════════════════════════════════
//  REAL-WORLD TAB — XẾP LỊCH DẠY (với Priority Triage System)
// ══════════════════════════════════════════════════════════════════════════════
const DAYS_RT    = ['T2','T3','T4','T5','T6','T7'];
const PERIODS_RT = ['Sáng','Chiều'];
const N_DAYS_RT  = 6;
const N_SLOTS_RT = 12;
const COLORS_RT  = ['#4f8ef7','#f472b6','#34d399','#fb923c','#a78bfa','#fbbf24','#38bdf8','#f87171'];
const slotIdx    = (d, p) => d * 2 + p;
const ALL_SLOTS  = Array.from({length: N_SLOTS_RT}, (_,i) => i);

// Priority metadata
const PRIORITY_META = {
  critical: {label:'Cấp bách',  icon:'🔴', color:'#f87171',
             desc:'Phải xử lý ngay — ảnh hưởng kết quả'},
  warning : {label:'Trung bình', icon:'🟡', color:'#fbbf24',
             desc:'Cần xử lý nhưng không khẩn cấp'},
  info    : {label:'Cảnh báo',  icon:'🟢', color:'#34d399',
             desc:'Chưa có vấn đề — đề phòng phát sinh'},
};

let rtClasses = [
  {name:'CNTT1601', color:'#4f8ef7', sessions_per_week:3, duration_weeks:3,
   auto_mode:false, allowed_slots:[slotIdx(0,0), slotIdx(2,0), slotIdx(4,0)]},
  {name:'CNTT1602', color:'#f472b6', sessions_per_week:2, duration_weeks:3,
   auto_mode:false, allowed_slots:[slotIdx(1,0), slotIdx(3,0)]},
  {name:'CNTT1603', color:'#34d399', sessions_per_week:2, duration_weeks:2,
   auto_mode:true,  allowed_slots:[...ALL_SLOTS]},
  {name:'CNTT1604', color:'#fb923c', sessions_per_week:2, duration_weeks:3,
   auto_mode:true,  allowed_slots:[...ALL_SLOTS]},
];
let rtResult_t = null, rtAlgo_t = 'a2c';

// ── Validation (chỉ cấu hình form, KHÔNG phải triage) ────────────────────────
function validateClasses() {
  const issues = [];
  rtClasses.forEach(cls => {
    const nSlots = cls.auto_mode ? N_SLOTS_RT : cls.allowed_slots.length;
    if (cls.sessions_per_week > nSlots) {
      issues.push({type:'error', cls:cls.name,
        msg:`${cls.sessions_per_week} buổi/tuần nhưng chỉ có ${nSlots} slot khả dụng`});
    }
    if (!cls.auto_mode && cls.allowed_slots.length === 0) {
      issues.push({type:'error', cls:cls.name, msg:'Chưa chọn slot nào'});
    }
  });
  return issues;
}

// ── Render config ────────────────────────────────────────────────────────────
function renderRTConfig() {
  const list = document.getElementById('rtClassList');
  if (!list) return;
  const issues = validateClasses();
  const errors = issues.filter(i => i.type === 'error');

  let issuesHTML = '';
  if (errors.length > 0) {
    issuesHTML = `<div class="issue-banner error">
      ⛔ ${errors.length} lỗi cấu hình:
      <ul>${errors.map(e => `<li><b>${e.cls||''}</b>: ${e.msg}</li>`).join('')}</ul>
    </div>`;
  }

  const totalSessions = rtClasses.reduce((s,c) => s + c.sessions_per_week*c.duration_weeks, 0);
  const autoCount = rtClasses.filter(c => c.auto_mode).length;
  const statsHTML = `<div class="rt-stats">
    📊 ${rtClasses.length} lớp · ${totalSessions} buổi tổng · ${autoCount}/${rtClasses.length} dùng AI Auto
  </div>`;

  list.innerHTML = issuesHTML + statsHTML + rtClasses.map((cls,i) => {
    const totalSess = cls.sessions_per_week * cls.duration_weeks;
    const slotsHTML = cls.auto_mode ? `
      <div class="auto-mode-badge">🤖 AI tự chọn slot phù hợp (tránh trùng giờ)</div>`
      : PERIODS_RT.map((per, pIdx) => {
        const cells = DAYS_RT.map((dayLbl, dIdx) => {
          const sIdx = slotIdx(dIdx, pIdx);
          const active = cls.allowed_slots.includes(sIdx);
          return `<button class="slot-btn ${active?'active':''}"
                          onclick="toggleSlot(${i},${dIdx},${pIdx})">${dayLbl}</button>`;
        }).join('');
        return `<div class="slot-row">
          <span class="slot-period-lbl">${per}:</span>
          <div class="slot-day-row">${cells}</div></div>`;
      }).join('');

    const slotInfo = cls.auto_mode
      ? `<span style="color:var(--accent)">🤖 Auto</span>`
      : `${cls.allowed_slots.length} slot ưu tiên`;

    return `<div class="class-item">
      <div class="class-item-header">
        <div class="subj-color-swatch" style="background:${cls.color}" onclick="cycleCls(${i})"></div>
        <input class="class-name-input" value="${cls.name}" onchange="updateClsName(${i}, this.value)" />
        <button class="subj-remove" onclick="removeCls(${i})">×</button>
      </div>
      <div class="class-item-body">
        <div class="class-row">
          <span class="class-lbl">Số tuần:</span>
          <div class="num-ctrl">
            <button onclick="adjustCls(${i},'duration_weeks',-1)">−</button>
            <span>${cls.duration_weeks}</span>
            <button onclick="adjustCls(${i},'duration_weeks',1)">+</button>
          </div>
          <span class="class-lbl">Buổi/tuần:</span>
          <div class="num-ctrl">
            <button onclick="adjustCls(${i},'sessions_per_week',-1)">−</button>
            <span>${cls.sessions_per_week}</span>
            <button onclick="adjustCls(${i},'sessions_per_week',1)">+</button>
          </div>
        </div>
        <label class="auto-toggle">
          <input type="checkbox" ${cls.auto_mode?'checked':''} onchange="toggleAutoMode(${i})" />
          <span>🤖 <b>AI Auto</b> — Để AI tự chọn slot tối ưu</span>
        </label>
        <div class="slot-grid">${slotsHTML}</div>
        <div class="class-meta">
          ${slotInfo} · Cần xếp <b style="color:var(--accent)">${totalSess} buổi</b>
        </div>
      </div>
    </div>`;
  }).join('');
}

function toggleAutoMode(i) {
  rtClasses[i].auto_mode = !rtClasses[i].auto_mode;
  if (rtClasses[i].auto_mode) {
    rtClasses[i]._saved_slots = [...rtClasses[i].allowed_slots];
    rtClasses[i].allowed_slots = [...ALL_SLOTS];
  } else {
    rtClasses[i].allowed_slots = (rtClasses[i]._saved_slots && rtClasses[i]._saved_slots.length > 0)
      ? rtClasses[i]._saved_slots : [slotIdx(0,0), slotIdx(2,0)];
  }
  renderRTConfig();
}

function toggleSlot(i, dayIdx, periodIdx) {
  if (rtClasses[i].auto_mode) {
    showToast('🤖 Tắt AI Auto trước khi chọn thủ công'); return;
  }
  const sIdx = slotIdx(dayIdx, periodIdx);
  const arr = rtClasses[i].allowed_slots;
  const pos = arr.indexOf(sIdx);
  if (pos >= 0) {
    if (arr.length > 1) arr.splice(pos, 1);
    else showToast('⚠️ Cần ít nhất 1 slot');
  } else {
    arr.push(sIdx); arr.sort((a,b)=>a-b);
  }
  renderRTConfig();
}

function cycleCls(i) {
  const cur = COLORS_RT.indexOf(rtClasses[i].color);
  rtClasses[i].color = COLORS_RT[(cur+1) % COLORS_RT.length];
  renderRTConfig();
}
function updateClsName(i, name) { rtClasses[i].name = name.trim() || `Lớp ${i+1}`; }
function removeCls(i) {
  if (rtClasses.length <= 2) { showToast('⚠️ Cần ít nhất 2 lớp'); return; }
  rtClasses.splice(i, 1); renderRTConfig();
}
function adjustCls(i, key, delta) {
  const max = key === 'sessions_per_week' ? 6 : 8;
  rtClasses[i][key] = Math.max(1, Math.min(max, rtClasses[i][key] + delta));
  renderRTConfig();
}
function addCls() {
  const name = `CNTT${1600 + rtClasses.length + 1}`;
  rtClasses.push({name, color:COLORS_RT[rtClasses.length % COLORS_RT.length],
    sessions_per_week:2, duration_weeks:3, auto_mode:true, allowed_slots:[...ALL_SLOTS]});
  renderRTConfig();
  showToast(`✅ Thêm "${name}" (AI Auto)`);
}
function resetRT() {
  rtClasses = [
    {name:'CNTT1601', color:'#4f8ef7', sessions_per_week:3, duration_weeks:3,
     auto_mode:false, allowed_slots:[slotIdx(0,0), slotIdx(2,0), slotIdx(4,0)]},
    {name:'CNTT1602', color:'#f472b6', sessions_per_week:2, duration_weeks:3,
     auto_mode:false, allowed_slots:[slotIdx(1,0), slotIdx(3,0)]},
    {name:'CNTT1603', color:'#34d399', sessions_per_week:2, duration_weeks:2,
     auto_mode:true,  allowed_slots:[...ALL_SLOTS]},
    {name:'CNTT1604', color:'#fb923c', sessions_per_week:2, duration_weeks:3,
     auto_mode:true,  allowed_slots:[...ALL_SLOTS]},
  ];
  rtResult_t = null;
  renderRTConfig();
  document.getElementById('rtResultArea').innerHTML = rtPlaceholderHTML();
  showToast('↺ Reset xong');
}

async function runTeacherSchedule() {
  const issues = validateClasses();
  const errors = issues.filter(i => i.type === 'error');
  if (errors.length > 0) { showToast(`❌ ${errors[0].msg}`); return; }

  const btn = document.getElementById('rtRunBtn');
  btn.disabled = true; btn.innerHTML = '⏳ Đang xếp lịch...';
  document.getElementById('rtResultArea').innerHTML = `
    <div class="real-placeholder">
      <div class="spinner"></div>
      <div class="ph-title" style="margin-top:8px">Đang xếp lịch thông minh...</div>
      <div class="ph-sub">Greedy Best-Fit Allocator + Priority Triage</div>
    </div>`;

  const payload = {
    classes: rtClasses.map(c => ({
      name: c.name, color: c.color,
      sessions_per_week: c.sessions_per_week,
      duration_weeks: c.duration_weeks,
      allowed_slots: c.auto_mode ? [...ALL_SLOTS] : c.allowed_slots,
    }))
  };

  try {
    const res = await fetch('/api/realworld/teacher', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    rtResult_t = await res.json();
    rtAlgo_t = 'a2c';
    renderTeacherResult();
    showToast('✅ Xếp lịch hoàn thành!');
  } catch (e) {
    document.getElementById('rtResultArea').innerHTML =
      `<div style="color:var(--red);padding:40px;text-align:center">❌ ${e.message}</div>`;
    showToast('❌ ' + e.message);
  } finally {
    btn.disabled = false; btn.innerHTML = '🚀 Xếp lịch với A2C';
  }
}

// ── PRIORITY TRIAGE PANEL (mới) ──────────────────────────────────────────────
function renderTriagePanel(triage) {
  if (!triage || triage.total === 0) {
    return `<div class="triage-panel ok">
      <div class="triage-header">
        <span class="triage-icon">✅</span>
        <div>
          <div class="triage-title">Lịch hoàn hảo</div>
          <div class="triage-subtitle">Không phát hiện vấn đề nào — cấu hình tối ưu</div>
        </div>
      </div>
    </div>`;
  }

  const c = triage.counts;
  // Header với 3 badges đếm
  let html = `<div class="triage-panel">
    <div class="triage-header">
      <span class="triage-icon">📋</span>
      <div class="triage-titlebox">
        <div class="triage-title">Phân loại vấn đề (Priority Triage)</div>
        <div class="triage-subtitle">Tổng ${triage.total} vấn đề được phát hiện — sắp theo mức độ ưu tiên</div>
      </div>
      <div class="triage-counts">
        ${c.critical > 0 ? `<span class="triage-count-badge critical">🔴 ${c.critical}</span>` : ''}
        ${c.warning > 0 ? `<span class="triage-count-badge warning">🟡 ${c.warning}</span>` : ''}
        ${c.info > 0 ? `<span class="triage-count-badge info">🟢 ${c.info}</span>` : ''}
      </div>
    </div>
    <div class="triage-body">`;

  // Group by priority
  const grouped = {critical:[], warning:[], info:[]};
  triage.issues.forEach(issue => {
    if (grouped[issue.priority]) grouped[issue.priority].push(issue);
  });

  ['critical', 'warning', 'info'].forEach(pri => {
    if (grouped[pri].length === 0) return;
    const meta = PRIORITY_META[pri];
    html += `<div class="triage-section ${pri}">
      <div class="triage-section-header">
        <span class="triage-section-icon">${meta.icon}</span>
        <span class="triage-section-label" style="color:${meta.color}">${meta.label.toUpperCase()}</span>
        <span class="triage-section-count">${grouped[pri].length} vấn đề</span>
        <span class="triage-section-desc">${meta.desc}</span>
      </div>
      <div class="triage-issues">`;

    grouped[pri].forEach(issue => {
      const tagClass = issue.affected_class
        ? `<span class="triage-tag" style="background:rgba(255,255,255,.06)">@ ${issue.affected_class}</span>`
        : '';
      html += `<div class="triage-issue ${pri}">
        <div class="triage-issue-head">
          <span class="triage-issue-title">${issue.title}</span>
          ${tagClass}
        </div>
        <div class="triage-issue-desc">${issue.description}</div>
        <div class="triage-issue-suggest">
          <span class="suggest-icon">💡</span>
          <span class="suggest-text">${issue.suggestion}</span>
        </div>
      </div>`;
    });

    html += `</div></div>`;
  });

  html += `</div></div>`;
  return html;
}

function renderTeacherResult() {
  if (!rtResult_t) return;
  const r = rtResult_t, exp = r.explanation, triage = r.triage;
  const algos = {
    a2c :{label:'A2C (RL)', color:'#4f8ef7'},
    edd :{label:'EDD',      color:'#f87171'},
    spt :{label:'SPT',      color:'#34d399'},
    fifo:{label:'FIFO',     color:'#fb923c'},
  };
  const cur = r[rtAlgo_t];
  const maxMS = Math.max(...Object.keys(algos).map(a => r[a]?.makespan || 0));
  const a2cMS = r.a2c?.makespan || 1, fifoMS = r.fifo?.makespan || 1;
  const improve = ((fifoMS - a2cMS) / fifoMS * 100).toFixed(1);

  // ── Triage Panel (mới — luôn hiện đầu tiên) ──────────────────────────────
  const triageHTML = renderTriagePanel(triage);

  // Báo cáo xếp lịch
  const counts = {};
  cur.schedule.forEach(e => { counts[e.class_name] = (counts[e.class_name]||0) + 1; });
  let summaryHTML = `<div class="schedule-summary">
    <div class="schedule-summary-title">📋 Báo cáo xếp lịch</div>
    <div class="schedule-summary-grid">`;
  r.classes.forEach(cls => {
    const placed = counts[cls.name] || 0;
    const ok = placed === cls.total_sessions;
    summaryHTML += `<div class="schedule-summary-item ${ok?'ok':'fail'}">
      <span class="ssi-color" style="background:${cls.color}"></span>
      <span class="ssi-name">${cls.name}</span>
      <span class="ssi-count ${ok?'ok':'fail'}">${ok?'✓':'⚠️'} ${placed}/${cls.total_sessions} buổi</span>
    </div>`;
  });
  summaryHTML += `</div></div>`;

  // Insight
  const insightHTML = `
    <div class="insight-box">
      <div class="insight-title">💡 So sánh Tuần tự vs Xen kẽ</div>
      <div class="insight-grid">
        <div class="insight-card bad">
          <div class="ic-label">❌ Tuần tự (FIFO)</div>
          <div class="ic-desc">Xếp tuần tự lớp này xong → mới đến lớp kia. Không tận dụng slot trống xen kẽ.</div>
          <div class="ic-value">${fifoMS} tuần</div>
        </div>
        <div class="insight-card good">
          <div class="ic-label">✅ Xen kẽ (A2C)</div>
          <div class="ic-desc">A2C ưu tiên lớp có ràng buộc nhất + cân tải các slot, xen kẽ vào slot trống.</div>
          <div class="ic-value">${a2cMS} tuần</div>
          <div class="ic-sub">${exp.saved > 0 ? `Tiết kiệm ${exp.saved} tuần (${exp.saved_pct}%)` : 'Tương đương FIFO trong case này'}</div>
        </div>
      </div>
    </div>`;

  // Metrics
  const metricsHTML = `<div class="real-metrics" style="margin-top:14px">
    <div class="real-metric best"><div class="real-metric-label">📅 Tổng tuần (A2C)</div>
      <div class="real-metric-val">${a2cMS}</div><div class="real-metric-sub">tuần dạy</div></div>
    <div class="real-metric info"><div class="real-metric-label">✓ Đã xếp</div>
      <div class="real-metric-val">${cur.schedule.length}</div><div class="real-metric-sub">buổi học</div></div>
    <div class="real-metric warn"><div class="real-metric-label">📈 Hiệu suất</div>
      <div class="real-metric-val">${cur.util}%</div><div class="real-metric-sub">slot có lịch</div></div>
    <div class="real-metric great"><div class="real-metric-label">🎯 vs Tuần tự</div>
      <div class="real-metric-val" style="color:${parseFloat(improve)>0?'var(--green)':'var(--muted)'}">
        ${improve > 0 ? '▼' : improve < 0 ? '▲' : '='}${Math.abs(improve)}%</div>
      <div class="real-metric-sub">so với FIFO</div></div>
  </div>`;

  // Algo tabs
  const tabsHTML = `<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;">
    <div class="algo-tabs">${Object.entries(algos).map(([k,v]) => `
      <button class="algo-tab-btn ${rtAlgo_t===k?'active':''}"
              style="${rtAlgo_t===k?'background:'+v.color:''}"
              onclick="switchTeacherAlgo('${k}')">${v.label}</button>`).join('')}</div>
    <span style="font-size:11px;color:var(--muted)">
      ${cur.makespan} tuần · ${cur.schedule.length} buổi · ${cur.util}% util
    </span>
  </div>`;

  // Calendar
  const sched = cur?.schedule || [];
  const nWeeks = Math.max(cur.makespan || 1, 1);
  const gridMap = {};
  sched.forEach(e => {
    if (!gridMap[e.week]) gridMap[e.week] = {};
    gridMap[e.week][e.machine] = e;
  });

  let calHTML = `<div class="weekly-cal-wrap">
    <div class="weekly-cal-header">
      <div class="wcal-corner">Tuần / Buổi</div>
      ${DAYS_RT.map(d => `<div class="wcal-day-hd">${d}</div>`).join('')}
    </div>`;

  for (let w = 0; w < nWeeks; w++) {
    for (let p = 0; p < 2; p++) {
      const periodLbl = PERIODS_RT[p];
      const isFirst = (p === 0);
      calHTML += `<div class="wcal-row ${p===0?'period-sang':'period-chieu'}">
        <div class="wcal-week-label">
          ${isFirst ? `<div class="wcal-week-num">Tuần ${w+1}</div>` : ''}
          <div class="wcal-period-lbl">${periodLbl}</div>
        </div>`;
      for (let d = 0; d < N_DAYS_RT; d++) {
        const slot = slotIdx(d, p);
        const entry = gridMap[w]?.[slot];
        if (entry) {
          calHTML += `<div class="wcal-cell has-class"
                            style="background:${entry.class_color}33; border-color:${entry.class_color}66;">
            <div class="wcal-class-name" style="color:${entry.class_color}">${entry.class_name}</div>
          </div>`;
        } else {
          calHTML += `<div class="wcal-cell empty"><span style="color:var(--muted);font-size:10px;opacity:.4">—</span></div>`;
        }
      }
      calHTML += `</div>`;
    }
  }
  calHTML += `</div>`;

  // Comparison bars
  const cmpHTML = `<div class="comparison-section" style="margin-top:14px">
    <div class="comparison-title">So sánh Makespan giữa các thuật toán</div>
    ${Object.entries(algos).map(([k,v]) => {
      const ms = r[k]?.makespan || 0;
      const pct = Math.round(ms / maxMS * 100);
      const isBest = ms === Math.min(...Object.keys(algos).map(a => r[a]?.makespan || 9999));
      const unsched = r[k]?.unscheduled?.length || 0;
      return `<div class="cmp-algo-row">
        <div class="cmp-algo-name" style="color:${v.color}">${v.label}</div>
        <div class="cmp-bar-wrap"><div class="cmp-bar-fill" style="width:${pct}%;background:${v.color}">${ms} tuần</div></div>
        <div class="cmp-makespan">${ms}</div>
        ${isBest ? '<span class="cmp-badge" style="background:rgba(52,211,153,.2);color:#34d399">🏆</span>' : ''}
        ${unsched > 0 ? `<span class="cmp-badge" style="background:rgba(248,113,113,.2);color:#f87171">⚠️ ${unsched}</span>` : ''}
      </div>`;
    }).join('')}
  </div>`;

  // ─── Compose: Triage trước, sau đó tới các phần khác ────────────────────
  document.getElementById('rtResultArea').innerHTML =
    triageHTML + summaryHTML + insightHTML + metricsHTML + tabsHTML + calHTML + cmpHTML;
}

function switchTeacherAlgo(algo) {
  rtAlgo_t = algo;
  renderTeacherResult();
}

function rtPlaceholderHTML() {
  return `<div class="real-placeholder">
    <div class="ph-icon">📅</div>
    <div class="ph-title">Chưa có lịch dạy</div>
    <div class="ph-sub">
      <strong style="color:var(--green)">🤖 AI Auto Mode</strong>: A2C tự chọn slot tối ưu<br>
      <strong style="color:var(--accent)">Strict Mode</strong>: Bạn tự chọn slot ưu tiên<br><br>
      Sau khi xếp lịch, hệ thống <strong style="color:var(--accent2)">Priority Triage</strong> sẽ phân loại<br>
      các vấn đề thành 3 mức: 🔴 Cấp bách / 🟡 Trung bình / 🟢 Cảnh báo
    </div>
  </div>`;
}

document.addEventListener('DOMContentLoaded', () => { setTimeout(renderRTConfig, 100); });