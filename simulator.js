/**
 * SIMD Vector Processor Simulator — simulator.js
 * 128-bit SSE-like simulation engine with full pipeline visualization
 */

'use strict';

// ─── Constants ─────────────────────────────────────────────────────────────
const NUM_REGS   = 8;   // V0 – V7
const LANE_COUNT = 4;   // 4 × int32 = 128 bits
const MEM_SIZE   = 256; // words (each = int32)

// Pipeline stage names in order
const STAGES = ['fetch', 'decode', 'execute', 'writeback'];

// Demo program
const DEMO_CODE = `; =============================================
; SIMD Vektör İşlemci — Tüm Komutları Demo
; 19 komutun tamamını sırayla deniyor
; =============================================

; --- 1) BELLEK İŞLEMLERİ ---
; Registerlara değer ata
VSET V0, 10, 20, 30, 40
VSET V1, 2, 4, 6, 8
; V0'ı belleğe yaz, sonra geri oku
VSTORE V0, [100]
VLOAD V2, [100]

; --- 2) ARİTMETİK İŞLEMLER ---
; Toplama: [10+2, 20+4, 30+6, 40+8] → [12, 24, 36, 48]
VADD V3, V0, V1
; Çıkarma: [10-2, 20-4, 30-6, 40-8] → [8, 16, 24, 32]
VSUB V4, V0, V1
; Çarpma: [10*2, 20*4, 30*6, 40*8] → [20, 80, 180, 320]
VMUL V5, V0, V1
; Bölme: [10/2, 20/4, 30/6, 40/8] → [5, 5, 5, 5]
VDIV V6, V0, V1
; Dot product: 10*2 + 20*4 + 30*6 + 40*8 = 600
VDOT V7, V0, V1
; Mutlak değer: V2 zaten [10,20,30,40]
VSET V2, -5, 3, -8, 1
VABS V3, V2
; Maksimum: max(V0,V2) → [10, 20, 30, 40]
VMAX V4, V0, V2
; Minimum: min(V0,V2) → [-5, 3, -8, 1]
VMIN V5, V0, V2

; --- 3) MANTIK ve BİT İŞLEMLERİ ---
VSET V0, 15, 255, 7, 128
VSET V1, 12, 170, 3, 64
; AND: bit bazında VE → [12, 170, 3, 0]
VAND V2, V0, V1
; OR: bit bazında VEYA → [15, 255, 7, 192]
VOR V3, V0, V1
; XOR: bit bazında ÖZEL VEYA
VXOR V4, V0, V1
; NOT: bit bazında DEĞİL
VNOT V5, V0
; Sola kaydır: her eleman 2 bit sola (×4)
VSHL V6, V0, 2
; Sağa kaydır: her eleman 1 bit sağa (÷2)
VSHR V7, V0, 1

; --- 4) KARŞILAŞTIRMA İŞLEMLERİ ---
VSET V0, 5, 10, 15, 20
VSET V1, 5, 8, 15, 25
; Eşitlik: [5==5, 10==8, 15==15, 20==25] → [1, 0, 1, 0]
VCMPEQ V2, V0, V1
; Büyüktür: [5>5, 10>8, 15>15, 20>25] → [0, 1, 0, 0]
VCMPGT V3, V0, V1
; Küçüktür: [5<5, 10<8, 15<15, 20<25] → [0, 0, 0, 1]
VCMPLT V4, V0, V1

; --- Sonuçları belleğe kaydet ---
VSTORE V2, [200]
VSTORE V3, [204]`;

// ─── State ─────────────────────────────────────────────────────────────────
const state = {
  registers: Array.from({ length: NUM_REGS }, () => new Array(LANE_COUNT).fill(0)),
  memory:    new Array(MEM_SIZE).fill(0),
  pc:        0,
  cycles:    0,
  instrCount: 0,
  elemCount:  0,
  scalarOps:  0,
  flags: { ZF: false, OF: false, NF: false, CF: false },
  lastMask: null,
  running:  false,
  stepMode: false,
  instructions: [],
  stepIdx: 0,
  speed: 700,
  viewMode: 'dec',
  modifiedRegs: new Set(),
  modifiedMem:  new Set(),
  runTimer: null,
  startTime: Date.now(),
};

// ─── DOM Refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  editor:       $('code-editor'),
  lineNumbers:  $('line-numbers'),
  cursorPos:    $('cursor-pos'),
  btnRun:       $('btn-run'),
  btnStep:      $('btn-step'),
  btnStop:      $('btn-stop'),
  btnReset:     $('btn-reset'),
  btnClearAll:  $('btn-clear-all'),
  btnLoadDemo:  $('btn-load-demo'),
  btnClearLog:  $('btn-clear-log'),
  btnCopy:      $('btn-copy'),
  speedSlider:  $('speed-slider'),
  speedVal:     $('speed-val'),
  statusBadge:  $('status-badge'),
  statusText:   $('status-text'),
  pipelinePC:   $('pipeline-pc'),
  fetchInstr:   $('fetch-instr'),
  decodeInstr:  $('decode-instr'),
  executeInstr: $('execute-instr'),
  wbInstr:      $('writeback-instr'),
  cibInstr:     $('cib-instr'),
  cibCycle:     $('cib-cycle'),
  regGrid:      $('registers-grid'),
  logBody:      $('log-body'),
  memBody:      $('memory-body'),
  memJumpAddr:  $('mem-jump-addr'),
  btnMemJump:   $('btn-mem-jump'),
  statCycles:   $('stat-cycles'),
  statInstr:    $('stat-instr'),
  statElems:    $('stat-elems'),
  statSpeedup:  $('stat-speedup'),
  scalarBar:    $('scalar-bar'),
  simdBar:      $('simd-bar'),
  scalarOps:    $('scalar-ops'),
  simdOps:      $('simd-ops'),
  perfNote:     $('perf-note'),
  isaToggle:    $('isa-toggle'),
  isaBody:      $('isa-body'),
  maskDisplay:  $('mask-display'),
  maskValue:    $('mask-value'),
};

// ─── Utilities ──────────────────────────────────────────────────────────────
function toInt32(n) {
  return n | 0;
}

function formatValue(v, mode) {
  const u = v >>> 0;
  if (mode === 'hex') return '0x' + u.toString(16).toUpperCase().padStart(8, '0');
  if (mode === 'bin') return u.toString(2).padStart(32, '0').replace(/(.{8})/g, '$1 ').trim();
  return String(v);
}

function nowStr() {
  const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function pcHex(pc) {
  return '0x' + (pc * 4).toString(16).toUpperCase().padStart(4, '0');
}

// ─── Parser ──────────────────────────────────────────────────────────────────
function parseCode(raw) {
  const lines = raw.split('\n');
  const instructions = [];
  const errors = [];

  lines.forEach((line, lineIdx) => {
    const stripped = line.trim().replace(/;.*$/, '').trim();
    if (!stripped) return;

    const tokens = stripped.split(/[\s,]+/).filter(Boolean);
    if (!tokens.length) return;

    const op = tokens[0].toUpperCase();
    const args = tokens.slice(1);

    // Opcode validation with expected argument counts
    const argCounts = {
      'VLOAD': 2, 'VSTORE': 2, 'VSET': 5,
      'VADD': 3, 'VSUB': 3, 'VMUL': 3, 'VDIV': 3, 'VDOT': 3,
      'VABS': 2, 'VMAX': 3, 'VMIN': 3,
      'VAND': 3, 'VOR': 3, 'VXOR': 3, 'VNOT': 2,
      'VSHL': 3, 'VSHR': 3,
      'VCMPEQ': 3, 'VCMPGT': 3, 'VCMPLT': 3,
    };

    if (!(op in argCounts)) {
      errors.push({ line: lineIdx + 1, msg: `Bilinmeyen komut: ${op}` });
      return;
    }

    const expected = argCounts[op];
    if (args.length !== expected) {
      errors.push({ line: lineIdx + 1, msg: `${op}: ${expected} argüman bekleniyor, ${args.length} verildi` });
      return;
    }

    const instr = { op, args, raw: stripped, line: lineIdx + 1 };
    instructions.push(instr);
  });

  return { instructions, errors };
}

function parseReg(s) {
  const m = s.match(/^V([0-7])$/i);
  if (!m) return null;
  return parseInt(m[1]);
}

function parseAddr(s) {
  const m = s.match(/^\[(\d+)\]$/);
  if (!m) return null;
  return parseInt(m[1]);
}

// ─── Execution Engine ─────────────────────────────────────────────────────
function executeInstruction(instr) {
  const { op, args } = instr;
  let result = null;
  let destReg = null;
  let destAddr = null;
  let cfSet = false;

  state.flags.OF = false;

  const getReg = i => {
    const idx = parseReg(args[i]);
    if (idx === null) throw new Error(`Geçersiz register: ${args[i]}`);
    return state.registers[idx].slice();
  };
  const setReg = (i, val) => {
    const idx = parseReg(args[i]);
    if (idx === null) throw new Error(`Geçersiz register: ${args[i]}`);
    state.registers[idx] = val.map(toInt32);
    state.modifiedRegs.add(idx);
    return idx;
  };

  switch (op) {
    // ── Memory ───────────────────────────────
    case 'VLOAD': {
      const dest = parseReg(args[0]);
      const addr = parseAddr(args[1]);
      if (dest === null) throw new Error(`Geçersiz register: ${args[0]}`);
      if (addr === null) throw new Error(`Geçersiz adres: ${args[1]}`);
      if (addr + LANE_COUNT > MEM_SIZE) throw new Error(`Adres aralık dışı: [${addr}] (max: [${MEM_SIZE - LANE_COUNT}])`);
      const data = [];
      for (let i = 0; i < LANE_COUNT; i++) {
        data.push(state.memory[addr + i]);
      }
      state.registers[dest] = data.map(toInt32);
      state.modifiedRegs.add(dest);
      destReg = dest;
      state.elemCount += LANE_COUNT;
      state.scalarOps += LANE_COUNT;
      log(`VLOAD: V${dest} ← Bellek[${addr}..${addr+3}] = [${data.join(', ')}]`, 'exec');
      break;
    }
    case 'VSTORE': {
      const src = parseReg(args[0]);
      const addr = parseAddr(args[1]);
      if (src === null) throw new Error(`Geçersiz register: ${args[0]}`);
      if (addr === null) throw new Error(`Geçersiz adres: ${args[1]}`);
      if (addr + LANE_COUNT > MEM_SIZE) throw new Error(`Adres aralık dışı: [${addr}] (max: [${MEM_SIZE - LANE_COUNT}])`);
      for (let i = 0; i < LANE_COUNT; i++) {
        const a = addr + i;
        state.memory[a] = state.registers[src][i];
        state.modifiedMem.add(a);
      }
      state.elemCount += LANE_COUNT;
      state.scalarOps += LANE_COUNT;
      log(`VSTORE: Bellek[${addr}..${addr+3}] ← V${src} = [${state.registers[src].join(', ')}]`, 'exec');
      break;
    }
    case 'VSET': {
      const dest = parseReg(args[0]);
      if (dest === null) throw new Error(`Geçersiz register: ${args[0]}`);
      const vals = [];
      for (let i = 1; i <= 4; i++) {
        vals.push(toInt32(parseInt(args[i] || '0')));
      }
      state.registers[dest] = vals;
      state.modifiedRegs.add(dest);
      destReg = dest;
      state.elemCount += LANE_COUNT;
      state.scalarOps += LANE_COUNT;
      log(`VSET: V${dest} ← [${vals.join(', ')}]`, 'exec');
      break;
    }

    // ── Arithmetic ────────────────────────────
    case 'VADD': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(x + v2[i]));
      // Unsigned carry: herhangi bir lane'de (a >>> 0) + (b >>> 0) > 0xFFFFFFFF ise CF = 1
      state.flags.CF = v1.some((x, i) => ((x >>> 0) + (v2[i] >>> 0)) > 0xFFFFFFFF);
      cfSet = true;
      // Signed overflow: aynı işaretli operandlar farklı işaretli sonuç üretiyorsa OF = 1
      state.flags.OF = v1.some((x, i) => ((x ^ result[i]) & (v2[i] ^ result[i])) < 0);
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VADD: V${d} = V${parseReg(args[1])} + V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VSUB': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(x - v2[i]));
      // Unsigned borrow: herhangi bir lane'de b > a (unsigned) ise CF = 1
      state.flags.CF = v1.some((x, i) => (v2[i] >>> 0) > (x >>> 0));
      cfSet = true;
      // Signed overflow: farklı işaretli operandlar ve sonuç a'dan farklı işaretli ise OF = 1
      state.flags.OF = v1.some((x, i) => ((x ^ v2[i]) & (x ^ result[i])) < 0);
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VSUB: V${d} = V${parseReg(args[1])} - V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VMUL': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(Math.imul(x, v2[i])));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT * 2;
      log(`VMUL: V${d} = V${parseReg(args[1])} × V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VDIV': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => v2[i] === 0 ? 0 : toInt32(Math.trunc(x / v2[i])));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT * 3;
      log(`VDIV: V${d} = V${parseReg(args[1])} / V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      if (v2.some(v => v === 0)) { state.flags.OF = true; log('Uyarı: Sıfıra bölme — sonuç 0 olarak ayarlandı', 'error'); }
      break;
    }
    case 'VDOT': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      const dot = toInt32(v1.reduce((acc, x, i) => acc + Math.imul(x, v2[i]), 0));
      result = [dot, 0, 0, 0];
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT * 3;
      log(`VDOT: V${d}[0] = dot(V${parseReg(args[1])}, V${parseReg(args[2])}) = ${dot}`, 'exec');
      break;
    }
    case 'VABS': {
      const d = parseReg(args[0]), v1 = getReg(1);
      result = v1.map(x => toInt32(Math.abs(x)));
      // INT32_MIN (-2147483648) durumunda abs() geri negatife sarılır → OF = 1
      if (v1.some(x => x === -2147483648)) {
        state.flags.OF = true;
        log('Uyarı: |INT32_MIN| = INT32_MIN (32-bit taşma — mutlak değer negatif kaldı)', 'error');
      }
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VABS: V${d} = |V${parseReg(args[1])}| → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VMAX': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => Math.max(x, v2[i]));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VMAX: V${d} = max(V${parseReg(args[1])}, V${parseReg(args[2])}) → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VMIN': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => Math.min(x, v2[i]));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VMIN: V${d} = min(V${parseReg(args[1])}, V${parseReg(args[2])}) → [${result.join(', ')}]`, 'exec');
      break;
    }

    // ── Logic / Bitwise ───────────────────────
    case 'VAND': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(x & v2[i]));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VAND: V${d} = V${parseReg(args[1])} & V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VOR': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(x | v2[i]));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VOR: V${d} = V${parseReg(args[1])} | V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VXOR': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => toInt32(x ^ v2[i]));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VXOR: V${d} = V${parseReg(args[1])} ^ V${parseReg(args[2])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VNOT': {
      const d = parseReg(args[0]), v1 = getReg(1);
      result = v1.map(x => toInt32(~x));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VNOT: V${d} = ~V${parseReg(args[1])} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VSHL': {
      const d = parseReg(args[0]), v1 = getReg(1);
      const n = parseInt(args[2]) & 31;
      result = v1.map(x => toInt32(x << n));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VSHL: V${d} = V${parseReg(args[1])} << ${n} → [${result.join(', ')}]`, 'exec');
      break;
    }
    case 'VSHR': {
      const d = parseReg(args[0]), v1 = getReg(1);
      const n = parseInt(args[2]) & 31;
      result = v1.map(x => toInt32(x >> n));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      log(`VSHR: V${d} = V${parseReg(args[1])} >> ${n} → [${result.join(', ')}]`, 'exec');
      break;
    }

    // ── Compare / Mask ────────────────────────
    case 'VCMPEQ': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => (x === v2[i] ? -1 : 0));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      state.lastMask = result.map(x => x ? 1 : 0);
      log(`VCMPEQ: V${d} (maske) = V${parseReg(args[1])} == V${parseReg(args[2])} → [${state.lastMask.join(', ')}]`, 'exec');
      break;
    }
    case 'VCMPGT': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => (x > v2[i] ? -1 : 0));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      state.lastMask = result.map(x => x ? 1 : 0);
      log(`VCMPGT: V${d} (maske) = V${parseReg(args[1])} > V${parseReg(args[2])} → [${state.lastMask.join(', ')}]`, 'exec');
      break;
    }
    case 'VCMPLT': {
      const d = parseReg(args[0]), v1 = getReg(1), v2 = getReg(2);
      result = v1.map((x, i) => (x < v2[i] ? -1 : 0));
      state.registers[d] = result;
      state.modifiedRegs.add(d); destReg = d;
      state.elemCount += LANE_COUNT; state.scalarOps += LANE_COUNT;
      state.lastMask = result.map(x => x ? 1 : 0);
      log(`VCMPLT: V${d} (maske) = V${parseReg(args[1])} < V${parseReg(args[2])} → [${state.lastMask.join(', ')}]`, 'exec');
      break;
    }

    default:
      throw new Error(`Desteklenmeyen komut: ${op}`);
  }

  // ── Update flags based on result ──────────
  if (result) {
    state.flags.ZF = result.every(x => x === 0);
    state.flags.NF = result.some(x => x < 0);
    // CF: taşma kontrolü — komut kendi CF hesabını yapmadıysa varsayılan kontrolü uygula
    if (!cfSet) {
      state.flags.CF = result.some(x => x === -2147483648);
    }
  }

  state.instrCount++;
  state.cycles += 4; // Each instruction: 4 pipeline stages
  state.pc++;
}

// ─── Pipeline Animation ───────────────────────────────────────────────────
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function animatePipeline(instr) {
  const speed = state.speed / 4; // divide across 4 stages
  const stageNames = ['fetch', 'decode', 'execute', 'writeback'];
  const stageIds   = ['stage-fetch', 'stage-decode', 'stage-execute', 'stage-writeback'];
  const instrIds   = [dom.fetchInstr, dom.decodeInstr, dom.executeInstr, dom.wbInstr];

  for (let s = 0; s < 4; s++) {
    if (!state.running && !state.stepMode) break;
    // Activate current stage
    stageIds.forEach(id => document.getElementById(id).classList.remove('active'));
    document.getElementById(stageIds[s]).classList.add('active');
    instrIds[s].textContent = instr.op;

    // Update current instruction bar
    dom.cibInstr.textContent = instr.raw;
    dom.cibCycle.textContent = state.cycles + s + 1;
    dom.pipelinePC.textContent = pcHex(state.pc);

    await delay(speed);
  }
}

// ─── Run / Step logic ────────────────────────────────────────────────────
function parseAndLoad() {
  const { instructions, errors } = parseCode(dom.editor.value);
  if (errors.length) {
    errors.forEach(e => log(`Satır ${e.line}: ${e.msg}`, 'error'));
    showToast(`${errors.length} sözdizim hatası bulundu`, 'error');
    return false;
  }
  state.instructions = instructions;
  state.stepIdx = 0;
  return true;
}

async function runAll() {
  if (!parseAndLoad()) return;
  if (state.instructions.length === 0) { showToast('Çalıştırılacak komut yok', 'info'); return; }

  setStatus('running', 'Çalışıyor...');
  state.running = true;
  state.stepMode = false;
  dom.btnStop.disabled = false;
  dom.btnRun.disabled  = true;
  dom.btnStep.disabled = true;

  log(`--- Yürütme başladı: ${state.instructions.length} komut ---`, 'info');

  let errorOccurred = false;
  for (let i = state.stepIdx; i < state.instructions.length; i++) {
    if (!state.running) break;
    state.stepIdx = i;
    const instr = state.instructions[i];
    highlightLine(instr.line);

    state.modifiedRegs.clear();
    state.modifiedMem.clear();

    await animatePipeline(instr);
    if (!state.running) break;

    try {
      executeInstruction(instr);
    } catch (err) {
      log(`Hata (Satır ${instr.line}): ${err.message}`, 'error');
      showToast(`Hata: ${err.message}`, 'error');
      errorOccurred = true;
      break;
    }

    renderRegisters();
    renderMemory();
    renderFlags();
    renderPerf();
    if (state.lastMask) renderMask();

    await delay(state.speed / 8);
  }

  if (!errorOccurred && state.running) {
    finishExecution();
  } else if (errorOccurred) {
    state.running = false;
    state.instructions = [];
    state.stepIdx = 0;
    dom.btnStop.disabled = true;
    dom.btnRun.disabled  = false;
    dom.btnStep.disabled = false;
    setStatus('error', 'Hata');
    resetPipelineDisplay();
  }
}

async function stepOnce() {
  // Only parse when no instructions are loaded
  if (state.instructions.length === 0) {
    if (!parseAndLoad()) return;
    if (state.instructions.length === 0) { showToast('Çalıştırılacak komut yok', 'info'); return; }
    log(`--- Adım adım yürütme başladı ---`, 'info');
    setStatus('running', 'Adım Modunda');
  }

  if (state.stepIdx >= state.instructions.length) {
    log('--- Tüm komutlar yürütüldü ---', 'success');
    showToast('Program tamamlandı!', 'success');
    setStatus('idle', 'Tamamlandı');
    resetPipelineDisplay();
    state.instructions = [];
    state.stepIdx = 0;
    return;
  }

  state.stepMode = true;
  state.running = true;

  const instr = state.instructions[state.stepIdx];
  highlightLine(instr.line);
  state.modifiedRegs.clear();
  state.modifiedMem.clear();

  await animatePipeline(instr);

  try {
    executeInstruction(instr);
  } catch (err) {
    log(`Hata (Satır ${instr.line}): ${err.message}`, 'error');
    showToast(`Hata: ${err.message}`, 'error');
    state.running = false;
    state.stepMode = false;
    state.instructions = [];
    state.stepIdx = 0;
    setStatus('error', 'Hata');
    return;
  }

  state.stepIdx++;
  renderRegisters();
  renderMemory();
  renderFlags();
  renderPerf();
  if (state.lastMask) renderMask();

  state.running = false;
  state.stepMode = false;

  if (state.stepIdx >= state.instructions.length) {
    log('--- Tüm komutlar yürütüldü ---', 'success');
    showToast('Program tamamlandı!', 'success');
    setStatus('idle', 'Tamamlandı');
    resetPipelineDisplay();
    state.instructions = [];
    state.stepIdx = 0;
  }
}

function stopExecution() {
  state.running = false;
  state.instructions = [];
  state.stepIdx = 0;
  dom.btnStop.disabled = true;
  dom.btnRun.disabled  = false;
  dom.btnStep.disabled = false;
  setStatus('idle', 'Durduruldu');
  log('--- Yürütme kullanıcı tarafından durduruldu ---', 'info');
  resetPipelineDisplay();
}

function finishExecution() {
  state.running = false;
  state.instructions = [];
  state.stepIdx = 0;
  dom.btnStop.disabled = true;
  dom.btnRun.disabled  = false;
  dom.btnStep.disabled = false;
  setStatus('idle', 'Tamamlandı');
  log(`--- Yürütme tamamlandı: ${state.instrCount} komut, ${state.cycles} döngü ---`, 'success');
  showToast('Program başarıyla tamamlandı!', 'success');
  resetPipelineDisplay();
}

function resetAll() {
  state.registers  = Array.from({ length: NUM_REGS }, () => new Array(LANE_COUNT).fill(0));
  state.memory     = new Array(MEM_SIZE).fill(0);
  state.pc         = 0;
  state.cycles     = 0;
  state.instrCount = 0;
  state.elemCount  = 0;
  state.scalarOps  = 0;
  state.flags      = { ZF: false, OF: false, NF: false, CF: false };
  state.lastMask   = null;
  state.running    = false;
  state.stepMode   = false;
  state.instructions = [];
  state.stepIdx    = 0;
  state.modifiedRegs.clear();
  state.modifiedMem.clear();

  dom.btnStop.disabled = true;
  dom.btnRun.disabled  = false;
  dom.btnStep.disabled = false;

  dom.maskDisplay.style.display = 'none';
  dom.pipelinePC.textContent = '0x0000';
  dom.cibInstr.textContent = '—';
  dom.cibCycle.textContent = '0';

  setStatus('idle', 'Hazır');
  resetPipelineDisplay();
  renderRegisters();
  renderMemory();
  renderFlags();
  renderPerf();
  log('--- Simülatör sıfırlandı ---', 'info');
}

function resetPipelineDisplay() {
  ['stage-fetch','stage-decode','stage-execute','stage-writeback'].forEach(id => {
    document.getElementById(id).classList.remove('active');
  });
  [dom.fetchInstr, dom.decodeInstr, dom.executeInstr, dom.wbInstr].forEach(el => {
    el.textContent = '—';
  });
}

// ─── Rendering ───────────────────────────────────────────────────────────
function renderRegisters() {
  dom.regGrid.innerHTML = '';
  for (let r = 0; r < NUM_REGS; r++) {
    const item = document.createElement('div');
    item.className = 'register-item' + (state.modifiedRegs.has(r) ? ' modified' : '');
    item.id = `reg-${r}`;

    const nameDiv = document.createElement('div');
    nameDiv.className = 'reg-name';
    nameDiv.textContent = `V${r}`;
    item.appendChild(nameDiv);

    const lanesDiv = document.createElement('div');
    lanesDiv.className = 'reg-lanes';

    for (let l = 0; l < LANE_COUNT; l++) {
      const lane = document.createElement('div');
      lane.className = `reg-lane lane-${l}`;
      const v = state.registers[r][l];
      if (state.viewMode === 'bin') {
        lane.style.fontSize = '7.5px';
        lane.textContent = formatValue(v, 'bin').replace(/ /g, '');
      } else {
        lane.style.fontSize = '';
        lane.textContent = formatValue(v, state.viewMode);
      }
      lane.title = `V${r}[${l}] = ${v} (0x${(v>>>0).toString(16).toUpperCase()})`;
      lanesDiv.appendChild(lane);
    }
    item.appendChild(lanesDiv);
    dom.regGrid.appendChild(item);
  }
}

function renderMemory(scrollToModified = true) {
  const rows = 20;
  dom.memBody.innerHTML = '';

  let firstModified = -1;
  if (state.modifiedMem.size > 0) {
    firstModified = Math.min(...state.modifiedMem);
  }

  const startRow = Math.max(0, Math.floor((firstModified >= 0 ? firstModified : 0) / 4) - 2);

  for (let row = startRow; row < startRow + rows && row * 4 < MEM_SIZE; row++) {
    const addr = row * 4;
    const tr = document.createElement('tr');
    tr.id = `mem-row-${row}`;

    const addrTd = document.createElement('td');
    addrTd.textContent = '0x' + addr.toString(16).toUpperCase().padStart(3, '0');
    tr.appendChild(addrTd);

    for (let c = 0; c < 4; c++) {
      const a = addr + c;
      const td = document.createElement('td');
      const v = state.memory[a] || 0;
      td.className = 'mem-cell' + (v === 0 ? ' zero' : '') + (state.modifiedMem.has(a) ? ' active' : '');
      td.textContent = v;
      td.title = `Adres ${a}: ${v} (0x${(v>>>0).toString(16).toUpperCase()})`;
      tr.appendChild(td);
    }
    dom.memBody.appendChild(tr);
  }
}

function renderFlags() {
  const flagMap = {
    'flag-zero':     state.flags.ZF,
    'flag-overflow': state.flags.OF,
    'flag-negative': state.flags.NF,
    'flag-carry':    state.flags.CF,
  };
  for (const [id, val] of Object.entries(flagMap)) {
    const el = document.getElementById(id);
    if (val) el.classList.add('active');
    else el.classList.remove('active');
  }
}

function renderMask() {
  dom.maskDisplay.style.display = 'block';
  dom.maskValue.textContent = `[${state.lastMask.join(', ')}]`;
}

function renderPerf() {
  dom.statCycles.textContent = state.cycles;
  dom.statInstr.textContent  = state.instrCount;
  dom.statElems.textContent  = state.elemCount;

  if (state.instrCount > 0) {
    const simdInstr  = state.instrCount;
    const scalarInstr = state.scalarOps;
    const speedup = scalarInstr > 0 ? (scalarInstr / simdInstr).toFixed(2) : '—';
    dom.statSpeedup.textContent = speedup !== '—' ? `${speedup}×` : '—';

    // bar chart: scalar = scalarOps, simd = instrCount
    const maxOps = Math.max(scalarInstr, 1);
    const simdPct = Math.max(10, Math.round((simdInstr / maxOps) * 100));
    dom.scalarBar.style.width = '100%';
    dom.simdBar.style.width   = simdPct + '%';
    dom.scalarOps.textContent = scalarInstr + ' op';
    dom.simdOps.textContent   = simdInstr + ' op';
    dom.perfNote.textContent  = `SIMD, ${speedup}x daha az komutla aynı sonucu üretiyor!`;
  }
}

// ─── Logging ──────────────────────────────────────────────────────────────
function log(msg, type = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;

  const timeSpan = document.createElement('span');
  timeSpan.className = 'log-time';
  timeSpan.textContent = nowStr();

  const msgSpan = document.createElement('span');
  msgSpan.className = 'log-msg';
  msgSpan.textContent = msg;

  entry.appendChild(timeSpan);
  entry.appendChild(msgSpan);
  dom.logBody.appendChild(entry);
  dom.logBody.scrollTop = dom.logBody.scrollHeight;
}

// ─── Toast ────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = '0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ─── Status ───────────────────────────────────────────────────────────────
function setStatus(mode, text) {
  dom.statusBadge.className = 'status-badge';
  if (mode === 'running') dom.statusBadge.classList.add('running');
  else if (mode === 'error') dom.statusBadge.classList.add('error');
  dom.statusText.textContent = text;
}

// ─── Editor helpers ───────────────────────────────────────────────────────
function updateLineNumbers() {
  const lines = dom.editor.value.split('\n').length;
  dom.lineNumbers.textContent = Array.from({ length: lines }, (_, i) => i + 1).join('\n');
}

function highlightLine(lineNum) {
  // Scroll editor to make line visible
  const lines = dom.editor.value.split('\n');
  const lineHeight = 19.8; // approx px per line
  dom.editor.scrollTop = (lineNum - 3) * lineHeight;
}

// ─── ISA collapsible ─────────────────────────────────────────────────────
function setupISA() {
  dom.isaToggle.addEventListener('click', () => {
    const open = dom.isaBody.style.display !== 'none';
    dom.isaBody.style.display = open ? 'none' : '';
    dom.isaToggle.querySelector('.chevron').classList.toggle('open', !open);
  });

  // Click on isa item to insert into editor
  dom.isaBody.querySelectorAll('.isa-item').forEach(item => {
    item.addEventListener('click', () => {
      const code = item.querySelector('code').textContent;
      const op = code.split(' ')[0];
      const template = generateTemplate(op);
      const pos = dom.editor.selectionStart;
      const val = dom.editor.value;
      dom.editor.value = val.slice(0, pos) + '\n' + template + val.slice(pos);
      updateLineNumbers();
      dom.editor.focus();
    });
  });
}

function generateTemplate(op) {
  const tpl = {
    'VLOAD': 'VLOAD V0, [100]',
    'VSTORE': 'VSTORE V0, [100]',
    'VSET': 'VSET V0, 1, 2, 3, 4',
    'VADD': 'VADD V2, V0, V1',
    'VSUB': 'VSUB V2, V0, V1',
    'VMUL': 'VMUL V2, V0, V1',
    'VDIV': 'VDIV V2, V0, V1',
    'VDOT': 'VDOT V2, V0, V1',
    'VABS': 'VABS V1, V0',
    'VMAX': 'VMAX V2, V0, V1',
    'VMIN': 'VMIN V2, V0, V1',
    'VAND': 'VAND V2, V0, V1',
    'VOR':  'VOR V2, V0, V1',
    'VXOR': 'VXOR V2, V0, V1',
    'VNOT': 'VNOT V1, V0',
    'VSHL': 'VSHL V1, V0, 2',
    'VSHR': 'VSHR V1, V0, 2',
    'VCMPEQ': 'VCMPEQ V2, V0, V1',
    'VCMPGT': 'VCMPGT V2, V0, V1',
    'VCMPLT': 'VCMPLT V2, V0, V1',
  };
  return tpl[op] || op;
}

// ─── Memory jump ─────────────────────────────────────────────────────────
function jumpToMemAddr() {
  const addr = parseInt(dom.memJumpAddr.value);
  if (isNaN(addr) || addr < 0 || addr >= MEM_SIZE) {
    showToast('Geçersiz adres (0-255 arası)', 'error');
    return;
  }
  // Render memory from addr
  state.modifiedMem.clear();
  state.modifiedMem.add(addr);
  renderMemory(true);
}

// ─── Export Results to CSV ────────────────────────────────────────────────
function exportToCSV() {
  let csv = [];
  csv.push(['=== SIMD SIMULATOR RAPORU ===']);
  csv.push(['Tarih/Saat', new Date().toLocaleString('tr-TR')]);
  csv.push(['']);

  // Performans Metrikleri
  csv.push(['--- PERFORMANS METRIKLERI ---']);
  csv.push(['Metrik', 'Değer']);
  csv.push(['Toplam Döngü (Cycles)', state.cycles]);
  csv.push(['Çalıştırılan Komut Sayısı', state.instrCount]);
  csv.push(['İşlenen Eleman Sayısı', state.elemCount]);
  const speedup = state.instrCount > 0 && state.scalarOps > 0 ? (state.scalarOps / state.instrCount).toFixed(2) : '1.0';
  csv.push(['Hızlanma Oranı (Scalar vs SIMD)', speedup + 'x']);
  csv.push(['']);

  // Vektör Registerları
  csv.push(['--- VEKTÖR REGİSTERLARI (128-bit) ---']);
  csv.push(['Register', 'Lane 0 (int32)', 'Lane 1 (int32)', 'Lane 2 (int32)', 'Lane 3 (int32)']);
  for (let r = 0; r < NUM_REGS; r++) {
    csv.push([`V${r}`, state.registers[r][0], state.registers[r][1], state.registers[r][2], state.registers[r][3]]);
  }
  csv.push(['']);

  // Durum Bayrakları
  csv.push(['--- DURUM BAYRAKLARI (FLAGS) ---']);
  csv.push(['Zero Flag (ZF)', state.flags.ZF ? '1 (Aktif)' : '0']);
  csv.push(['Overflow Flag (OF)', state.flags.OF ? '1 (Aktif)' : '0']);
  csv.push(['Negative Flag (NF)', state.flags.NF ? '1 (Aktif)' : '0']);
  csv.push(['Carry Flag (CF)', state.flags.CF ? '1 (Aktif)' : '0']);
  csv.push(['']);

  // Bellek Alanı (Sadece 0 olmayan hücreler)
  csv.push(['--- BELLEK DURUMU (Dolu Adresler) ---']);
  csv.push(['Adres (Dec)', 'Adres (Hex)', 'Değer (int32)']);
  let hasMem = false;
  for (let a = 0; a < MEM_SIZE; a++) {
    if (state.memory[a] !== 0) {
      hasMem = true;
      csv.push([a, '0x' + a.toString(16).toUpperCase().padStart(3, '0'), state.memory[a]]);
    }
  }
  if (!hasMem) csv.push(['Tüm bellek hücreleri sıfır (0)']);

  // CSV dosyasını oluştur ve indir
  const csvContent = 'data:text/csv;charset=utf-8,\uFEFF' + csv.map(e => e.join(';')).join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `SIMD_Simulasyon_Raporu_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showToast('Simülasyon raporu Excel/CSV olarak indirildi!', 'success');
}

// ─── Event Listeners ─────────────────────────────────────────────────────
function setupEvents() {
  dom.btnRun.addEventListener('click', runAll);
  dom.btnStep.addEventListener('click', stepOnce);
  dom.btnStop.addEventListener('click', stopExecution);
  dom.btnReset.addEventListener('click', resetAll);
  const btnExport = $('btn-export-csv');
  if (btnExport) btnExport.addEventListener('click', exportToCSV);
  dom.btnClearAll.addEventListener('click', () => {
    resetAll();
    dom.editor.value = '';
    updateLineNumbers();
  });
  dom.btnLoadDemo.addEventListener('click', () => {
    resetAll();
    dom.editor.value = DEMO_CODE;
    updateLineNumbers();
    showToast('Demo programı yüklendi!', 'info');
  });
  dom.btnClearLog.addEventListener('click', () => {
    dom.logBody.innerHTML = '';
  });
  dom.btnCopy.addEventListener('click', () => {
    navigator.clipboard.writeText(dom.editor.value)
      .then(() => showToast('Kod kopyalandı!', 'success'))
      .catch(() => showToast('Kopyalama başarısız', 'error'));
  });

  dom.editor.addEventListener('input', updateLineNumbers);
  dom.editor.addEventListener('scroll', () => {
    dom.lineNumbers.scrollTop = dom.editor.scrollTop;
  });
  dom.editor.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); runAll(); }
    if (e.key === 'F8') { e.preventDefault(); stepOnce(); }
    if (e.key === 'Tab') {
      e.preventDefault();
      const s = dom.editor.selectionStart;
      const val = dom.editor.value;
      dom.editor.value = val.slice(0, s) + '  ' + val.slice(s);
      dom.editor.selectionStart = dom.editor.selectionEnd = s + 2;
    }
  });
  dom.editor.addEventListener('keyup', updateCursorPos);
  dom.editor.addEventListener('click', updateCursorPos);

  dom.speedSlider.addEventListener('input', () => {
    state.speed = parseInt(dom.speedSlider.value);
    dom.speedVal.textContent = state.speed + 'ms';
  });

  // Register view toggle
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.viewMode = btn.dataset.view;
      renderRegisters();
    });
  });

  dom.btnMemJump.addEventListener('click', jumpToMemAddr);
  dom.memJumpAddr.addEventListener('keydown', e => { if (e.key === 'Enter') jumpToMemAddr(); });

  setupISA();
}

function updateCursorPos() {
  const val = dom.editor.value.slice(0, dom.editor.selectionStart);
  const lines = val.split('\n');
  const line = lines.length;
  const col  = lines[lines.length - 1].length + 1;
  dom.cursorPos.textContent = `Satır ${line}, Sütun ${col}`;
}

// ─── Init ────────────────────────────────────────────────────────────────
function init() {
  updateLineNumbers();
  renderRegisters();
  renderMemory();
  renderFlags();
  renderPerf();
  setupEvents();

  // Load demo on start
  dom.editor.value = DEMO_CODE;
  updateLineNumbers();

  log('SIMD Vektör İşlemci Simülatörü v1.0 hazır.', 'info');
  log('128-bit vektörler (4×int32) · 8 register (V0–V7) · 1KB bellek', 'info');
  log('"Demo Yükle" ile örnek kodu çalıştırabilirsiniz.', 'info');
}

document.addEventListener('DOMContentLoaded', init);
