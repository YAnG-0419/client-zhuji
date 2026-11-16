#!/usr/bin/env python3
"""Build a self-contained HTML view of one XFlow live run.

Reads the audit records written by ``live_client`` (``audit_plans/*.npz`` and
``audit_executions/*.npz``) and writes a single HTML file with no external
dependencies, so it can be copied off zhuji and opened anywhere:

* the 320x320 frame the model actually saw, with the predicted 21 keypoints of
  the selected step drawn on it next to the keypoints that were sent in;
* predicted wrist paths, commanded vs measured SDK EEF, and hand joints;
* a per-cycle overview (round trip, planning time, chunk length, tracking).

Pure NumPy + stdlib: PNG is written here so no image library is needed.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np


SIDES = ("left", "right")
# MediaPipe-21 bones, used for the skeleton overlay.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
MAX_SERIES_POINTS = 240


def png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder for a uint8 HxWx3 array."""
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def data_url(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgb)).decode("ascii")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Camera-optical [.,3] metres -> pixel [.,2]; points behind the camera get NaN."""
    points = np.asarray(points, dtype=float)
    z = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
        v = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    bad = ~np.isfinite(u) | ~np.isfinite(v) | (z <= 1e-6)
    return np.stack([np.where(bad, np.nan, u), np.where(bad, np.nan, v)], axis=-1)


def thin(values: np.ndarray, limit: int = MAX_SERIES_POINTS) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit).round().astype(int)]


def rounded(values, digits=4):
    """Round to JSON-safe values; non-finite entries become null for the viewer."""
    array = np.round(np.asarray(values, dtype=float), digits)
    return np.where(np.isfinite(array), array, None).tolist()


def virtual_intrinsics(client_dir: Path, result: dict) -> tuple[np.ndarray, str]:
    """The pinhole the stored 320x320 frame is expressed in."""
    config = json.loads((client_dir / "deployment_config.json").read_text())["image_preprocess"]
    planner = result.get("planner") or {}
    mode = planner.get("image_mode", "virtual_cam0")
    preset = planner.get("virtual_preset", "cam0_newk")
    if mode != "virtual_cam0":
        # stretch_undistort: the live camera's own rectified K scaled to 320x320.
        live = config["zhuji_live_reference"]
        new_k = np.asarray(config["zhuji_stretch"]["new_K_alpha0"], dtype=float)
        scale = np.array([320.0 / live["width"], 320.0 / live["height"]])
        matrix = new_k.copy()
        matrix[0, 0] *= scale[0]; matrix[0, 2] = (matrix[0, 2] + 0.5) * scale[0] - 0.5
        matrix[1, 1] *= scale[1]; matrix[1, 2] = (matrix[1, 2] + 0.5) * scale[1] - 0.5
        return matrix, "%s (live camera K scaled)" % mode
    if preset == "cam0_newk":
        return np.asarray(config["cam0"]["virtual_K_320"], dtype=float), "virtual_cam0/cam0_newk"
    return (np.asarray(config["virtual_targets"][preset]["K_320"], dtype=float),
            "virtual_cam0/%s" % preset)


def load_plan(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"meta": meta, "arrays": arrays}


def load_execution(path: Path) -> dict:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"meta": meta, "arrays": arrays}


def build_cycle(plan: dict | None, execution: dict | None, intrinsics: np.ndarray) -> dict:
    out: dict = {}
    if plan is not None:
        meta, arrays = plan["meta"], plan["arrays"]
        inference = meta.get("inference") or {}
        report = meta.get("planner_report") or {}
        timing = report.get("timing") or {}
        out.update(plan_id=int(meta["plan_id"]), status=meta.get("status"),
                   task=inference.get("task"), model=inference.get("model"),
                   reason=meta.get("reason"), history_sent=meta.get("history_sent"),
                   rtt_ms=inference.get("response_round_trip_ms"),
                   server_ms=(inference.get("response_timings_ms") or {}).get("total"),
                   denoise_ms=(inference.get("response_timings_ms") or {}).get("denoise"),
                   plan_ms=(report.get("stage_ms") or {}).get("total_ms"),
                   stretch=timing.get("stretch"), binding=timing.get("binding"),
                   planned_duration_s=timing.get("duration_s"),
                   splice=report.get("splice"), shaping_slowdown=timing.get("shaping_slowdown"))
        anchor = report.get("anchor") or {}
        out["anchor_mm"] = [
            (None if (anchor.get(side) or {}).get("local_offset_translation_norm_m") is None
             else round(1000 * anchor[side]["local_offset_translation_norm_m"], 2))
            for side in SIDES]
        if "model_head_rgb" in arrays:
            out["image"] = data_url(arrays["model_head_rgb"])
        if "model_response_keypoints" in arrays:
            predicted = arrays["model_response_keypoints"]          # [32,2,21,3]
            out["predicted_px"] = [rounded(project(predicted[:, side], intrinsics), 1)
                                   for side in range(2)]
            out["predicted_depth_m"] = [rounded(predicted[:, side, :, 2].mean(axis=1), 3)
                                        for side in range(2)]
            wrist = predicted[:, :, 0, :]                            # [32,2,3]
            out["predicted_wrist_m"] = [rounded(wrist[:, side], 4) for side in range(2)]
        if "model_request_keypoints" in arrays:
            current = arrays["model_request_keypoints"]
            out["current_px"] = [rounded(project(current[side], intrinsics), 1)
                                 for side in range(2)]
        if "model_response_keypoint_time_s" in arrays:
            out["model_time_s"] = rounded(arrays["model_response_keypoint_time_s"], 4)
        if "plan_times_s" in arrays:
            out["plan_times_s"] = rounded(arrays["plan_times_s"], 4)
            out["plan_eef_m"] = [rounded(arrays["plan_eef_xyzw"][:, side, :3], 4)
                                 for side in range(2)]
    if execution is not None:
        meta, arrays = execution["meta"], execution["arrays"]
        out.setdefault("plan_id", int(meta["plan_id"]))
        out.update(executed=bool(meta.get("execute")),
                   completed=bool(meta.get("completed_chunk")),
                   exec_status=meta.get("status"),
                   duration_s=meta.get("duration_recorded_s"),
                   eef_error=meta.get("eef_position_error_m"),
                   rotation_error=meta.get("eef_rotation_error_rad"),
                   q20_error=meta.get("q20_abs_error_rad"))
        if "feedback_elapsed_s" in arrays:
            keep = np.linspace(0, len(arrays["feedback_elapsed_s"]) - 1,
                               min(len(arrays["feedback_elapsed_s"]), MAX_SERIES_POINTS))
            keep = np.unique(keep.round().astype(int))
            out["feedback"] = {
                "t": rounded(arrays["feedback_elapsed_s"][keep], 3),
                "target_m": [rounded(arrays["feedback_eef_target_xyzw"][keep, side, :3], 4)
                             for side in range(2)],
                "measured_m": [rounded(arrays["feedback_eef_measured_xyzw"][keep, side, :3], 4)
                               for side in range(2)],
                "error_mm": [rounded(arrays["eef_position_error_m"][keep, side] * 1000, 2)
                             for side in range(2)],
                "hand_error_rad": [rounded(np.max(np.abs(
                    arrays["q20_abs_error_rad"][keep, side]), axis=-1)
                    if arrays["q20_abs_error_rad"].ndim == 3
                    else arrays["q20_abs_error_rad"][keep], 4) for side in range(2)],
            }
    return out


def collect(run_dir: Path, client_dir: Path) -> dict:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    intrinsics, intrinsics_label = virtual_intrinsics(client_dir, result)
    plans = {}
    for path in sorted((run_dir / "audit_plans").glob("plan*.npz")):
        if path.with_suffix(".json").is_file():
            plan = load_plan(path)
            plans[int(plan["meta"]["plan_id"])] = plan
    executions = {}
    for path in sorted((run_dir / "audit_executions").glob("exec*.npz")):
        if path.with_suffix(".json").is_file():
            execution = load_execution(path)
            executions[int(execution["meta"]["plan_id"])] = execution
    if not plans and not executions:
        raise SystemExit("no audit records in %s (was the run made before the audit trail?)"
                         % run_dir)
    cycles = [build_cycle(plans.get(key), executions.get(key), intrinsics)
              for key in sorted(set(plans) | set(executions))]
    planner = result.get("planner") or {}
    return {
        "run": run_dir.name,
        "generated_from": str(run_dir),
        "model": planner.get("model"), "task": planner.get("task"),
        "deployment_id": planner.get("deployment_id"),
        "contract": planner.get("contract"), "speed_ratio": planner.get("speed_ratio"),
        "image_mode": intrinsics_label,
        "shaping": planner.get("shaping") or {"enabled": False},
        "prefetch": (result.get("args") or {}).get("prefetch") or 0,
        "execute": result.get("execute"), "stop_reason": result.get("stop_reason"),
        "error": result.get("error"), "arm_commands": result.get("arm_motion_commands_sent"),
        "cycles": cycles,
    }


TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>XFlow run __RUN__</title>
<style>
 body{font:13px/1.5 system-ui,sans-serif;margin:0;background:#11151c;color:#dde3ec}
 header{padding:12px 18px;background:#0b0e13;border-bottom:1px solid #222a36}
 h1{font-size:15px;margin:0 0 4px} .meta{color:#8b97a8;font-size:12px}
 .meta b{color:#dde3ec;font-weight:600}
 main{display:grid;grid-template-columns:360px 1fr;gap:16px;padding:16px}
 .panel{background:#161b24;border:1px solid #222a36;border-radius:8px;padding:12px}
 .panel h2{font-size:12px;margin:0 0 8px;color:#8b97a8;text-transform:uppercase;letter-spacing:.06em}
 canvas{display:block;width:100%;height:auto;background:#0b0e13;border-radius:4px}
 .row{display:flex;align-items:center;gap:8px;margin:8px 0}
 input[type=range]{flex:1} select{background:#0b0e13;color:#dde3ec;border:1px solid #2a3340;
   border-radius:4px;padding:4px 6px;font:inherit}
 .legend span{display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:12px}
 .swatch{width:10px;height:10px;border-radius:2px;display:inline-block}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{text-align:right;padding:3px 6px;border-bottom:1px solid #222a36}
 th:first-child,td:first-child{text-align:left}
 th{color:#8b97a8;font-weight:600} tr.sel td{background:#1d2735}
 tr:hover td{background:#1a2230;cursor:pointer}
 .bad{color:#ff8f8f} .ok{color:#7fd88f} .mut{color:#8b97a8}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}
</style>
<header>
 <h1>XFlow live run <span id="run"></span></h1>
 <div class="meta" id="meta"></div>
</header>
<main>
 <div>
  <div class="panel">
   <h2>模型看到的画面 / 预测关键点</h2>
   <canvas id="cam" width="640" height="640"></canvas>
   <div class="row"><span class="mut" id="steplabel">step</span>
    <input type="range" id="step" min="0" max="31" value="0"></div>
   <div class="legend">
    <span><i class="swatch" style="background:#4da3ff"></i>预测 左</span>
    <span><i class="swatch" style="background:#ff9f43"></i>预测 右</span>
    <span><i class="swatch" style="background:#8b97a8"></i>当前(送入)</span>
   </div>
   <div class="meta" id="caminfo"></div>
  </div>
  <div class="panel" style="margin-top:12px">
   <h2>各段概览</h2>
   <table id="table"></table>
  </div>
 </div>
 <div class="panel">
  <h2>选中段：指令 vs 实测</h2>
  <div class="row"><select id="side"><option value="0">左臂 left</option>
    <option value="1">右臂 right</option></select>
   <span class="mut" id="cyclelabel"></span></div>
  <div class="charts">
   <div><div class="meta">末端位置 x/y/z（实线=指令，虚线=实测，m）</div>
    <canvas id="pos" width="760" height="300"></canvas></div>
   <div><div class="meta">末端跟踪误差（mm）</div>
    <canvas id="err" width="760" height="300"></canvas></div>
   <div><div class="meta">预测手腕轨迹（相机系，m）</div>
    <canvas id="wrist" width="760" height="300"></canvas></div>
   <div><div class="meta">手指最大跟踪误差（rad）</div>
    <canvas id="hand" width="760" height="300"></canvas></div>
  </div>
 </div>
</main>
<script>
const DATA = __DATA__;
const C = {left:'#4da3ff', right:'#ff9f43', axis:'#2a3340', text:'#8b97a8',
           x:'#ff6b6b', y:'#7fd88f', z:'#4da3ff', cur:'#8b97a8'};
let cycle = 0, side = 0, step = 0;
const $ = id => document.getElementById(id);
const fmt = (v,d=2) => (v==null||!isFinite(v)) ? '–' : (+v).toFixed(d);

function setup(){
  $('run').textContent = DATA.run;
  $('meta').innerHTML = [
    ['模型', DATA.model], ['任务', DATA.task], ['契约', DATA.contract],
    ['speed_ratio', DATA.speed_ratio], ['图像', DATA.image_mode],
    ['平滑', DATA.shaping && DATA.shaping.enabled ? 'on' : 'off'],
    ['预取', DATA.prefetch ? DATA.prefetch : 'off'],
    ['真机执行', DATA.execute ? '是' : '影子'],
    ['停止原因', DATA.stop_reason || DATA.error || '—'],
  ].filter(p => p[1] !== undefined && p[1] !== null)
   .map(p => p[0]+' <b>'+p[1]+'</b>').join(' · ');
  buildTable();
  $('step').oninput = e => { step = +e.target.value; drawCam(); };
  $('side').onchange = e => { side = +e.target.value; drawAll(); };
  select(0);
}

function buildTable(){
  const head = '<tr><th>段</th><th>往返</th><th>规划</th><th>时长</th><th>锚定mm</th>'
             + '<th>误差p95</th><th>误差max</th><th>状态</th></tr>';
  const rows = DATA.cycles.map((c,i) => {
    const e = c.eef_error || {};
    const bad = c.status === 'rejected' || c.completed === false;
    const a = c.anchor_mm || [];
    return `<tr data-i="${i}" title="${c.reason||''}"><td>${c.plan_id}</td>`
      + `<td>${fmt(c.rtt_ms,0)}</td><td>${fmt(c.plan_ms,0)}</td>`
      + `<td>${fmt(c.duration_s)}</td><td>${fmt(a[0],1)}/${fmt(a[1],1)}</td>`
      + `<td>${fmt(e.p95*1000)}</td><td>${fmt(e.max*1000)}</td>`
      + `<td class="${bad?'bad':'ok'}">${c.status==='rejected'?'被拒':(c.executed?'已执行':'未执行')}</td></tr>`;
  }).join('');
  $('table').innerHTML = head + rows;
  $('table').querySelectorAll('tr[data-i]').forEach(tr =>
    tr.onclick = () => select(+tr.dataset.i));
}

function select(i){
  cycle = i; step = 0; $('step').value = 0;
  $('table').querySelectorAll('tr[data-i]').forEach(tr =>
    tr.classList.toggle('sel', +tr.dataset.i === i));
  const c = DATA.cycles[i];
  const n = c.predicted_px ? c.predicted_px[0].length : 1;
  $('step').max = Math.max(0, n-1);
  const bits = ['段 ' + c.plan_id];
  if (c.rtt_ms != null) bits.push('往返 ' + fmt(c.rtt_ms,0) + ' ms (denoise ' + fmt(c.denoise_ms,0) + ')');
  if (c.plan_ms != null) bits.push('规划 ' + fmt(c.plan_ms,0) + ' ms');
  if (c.stretch != null) bits.push('拉长 x' + fmt(c.stretch));
  if (c.history_sent != null) bits.push('历史帧 ' + (c.history_sent ? '有' : '无'));
  if (c.splice) bits.push('拼接: 跳过 ' + c.splice.skipped_points + ' 步');
  if (c.reason) bits.push(c.reason);
  $('cyclelabel').textContent = bits.join(' · ');
  drawAll();
}

function drawAll(){ drawCam(); drawPos(); drawErr(); drawWrist(); drawHand(); }

function drawCam(){
  const c = DATA.cycles[cycle], cv = $('cam'), g = cv.getContext('2d');
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,cv.width,cv.height);
  g.fillStyle = '#0b0e13'; g.fillRect(0,0,cv.width,cv.height);
  const paint = () => {
    const s = cv.width/320;
    g.setTransform(s,0,0,s,0,0);
    if (c.current_px) [0,1].forEach(k => skeleton(g, c.current_px[k], C.cur, 1, .55));
    if (c.predicted_px) [0,1].forEach(k =>
      skeleton(g, c.predicted_px[k][Math.min(step, c.predicted_px[k].length-1)],
               k===0?C.left:C.right, 1.6, 1));
    g.setTransform(1,0,0,1,0,0);
    const t = c.model_time_s ? c.model_time_s[Math.min(step, c.model_time_s.length-1)] : null;
    $('steplabel').textContent = 'step ' + step + (t!=null ? '  (模型 t=' + fmt(t,3) + 's)' : '');
    $('caminfo').textContent = c.image ? '' : '该段没有保存图像（可能是被拒的请求）';
  };
  if (c.image){
    const img = new Image();
    img.onload = () => { g.drawImage(img, 0, 0, cv.width, cv.height); paint(); };
    img.src = c.image;
  } else paint();
}

function skeleton(g, pts, color, width, alpha){
  if (!pts) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width;
  const BONES = __BONES__;
  g.beginPath();
  for (const [a,b] of BONES){
    const p = pts[a], q = pts[b];
    if (!p || !q || !isFinite(p[0]) || !isFinite(q[0])) continue;
    g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]);
  }
  g.stroke();
  for (const p of pts){
    if (!p || !isFinite(p[0])) continue;
    g.beginPath(); g.arc(p[0], p[1], width*1.3, 0, 6.2832); g.fill();
  }
  g.globalAlpha = 1;
}

function axes(g, cv, xs, ys, xlabel){
  const pad = {l:52, r:10, t:10, b:24};
  const W = cv.width, H = cv.height;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (!isFinite(ymin) || !isFinite(ymax)) { ymin = 0; ymax = 1; }
  if (ymax - ymin < 1e-9) { ymax = ymin + 1e-6; }
  const padY = (ymax-ymin)*0.08; ymin -= padY; ymax += padY;
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,W,H);
  g.strokeStyle = C.axis; g.fillStyle = C.text; g.lineWidth = 1;
  g.font = '11px system-ui';
  for (let k=0;k<=4;k++){
    const y = pad.t + (H-pad.t-pad.b)*k/4, v = ymax - (ymax-ymin)*k/4;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillText(v.toFixed(Math.abs(v)<1?3:2), 4, y+3);
  }
  g.fillText(xlabel, W-pad.r-40, H-6);
  const X = v => pad.l + (W-pad.l-pad.r)*(xmax===xmin?0:(v-xmin)/(xmax-xmin));
  const Y = v => pad.t + (H-pad.t-pad.b)*(1-(v-ymin)/(ymax-ymin));
  return {X, Y};
}

function line(g, xs, ys, map, color, dash){
  g.strokeStyle = color; g.lineWidth = 1.6; g.setLineDash(dash||[]);
  g.beginPath(); let started = false;
  for (let i=0;i<xs.length;i++){
    const v = ys[i];
    if (v == null || !isFinite(v)) { started = false; continue; }
    const px = map.X(xs[i]), py = map.Y(v);
    started ? g.lineTo(px,py) : g.moveTo(px,py); started = true;
  }
  g.stroke(); g.setLineDash([]);
}

function empty(cv, text){
  const g = cv.getContext('2d');
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,cv.width,cv.height);
  g.fillStyle = C.text; g.font = '12px system-ui';
  g.fillText(text, 12, 24);
}

function drawPos(){
  const c = DATA.cycles[cycle], cv = $('pos'), g = cv.getContext('2d');
  if (!c.feedback) return empty(cv, '该段没有执行记录（影子或被拒）');
  const f = c.feedback, xs = f.t;
  const all = [].concat(...[0,1,2].map(a => f.target_m[side].map(p=>p[a])))
                .concat(...[0,1,2].map(a => f.measured_m[side].map(p=>p[a])));
  const map = axes(g, cv, xs, all, 's');
  [[0,C.x],[1,C.y],[2,C.z]].forEach(([a,color]) => {
    line(g, xs, f.target_m[side].map(p=>p[a]), map, color);
    line(g, xs, f.measured_m[side].map(p=>p[a]), map, color, [4,3]);
  });
}

function drawErr(){
  const c = DATA.cycles[cycle], cv = $('err'), g = cv.getContext('2d');
  if (!c.feedback) return empty(cv, '无执行记录');
  const f = c.feedback;
  const map = axes(g, cv, f.t, [0].concat(f.error_mm[0], f.error_mm[1]), 's');
  line(g, f.t, f.error_mm[0], map, C.left);
  line(g, f.t, f.error_mm[1], map, C.right);
}

function drawWrist(){
  const c = DATA.cycles[cycle], cv = $('wrist'), g = cv.getContext('2d');
  if (!c.predicted_wrist_m) return empty(cv, '该段没有模型输出');
  const w = c.predicted_wrist_m[side], xs = c.model_time_s || w.map((_,i)=>i);
  const map = axes(g, cv, xs, [].concat(...[0,1,2].map(a=>w.map(p=>p[a]))), 's (模型时间)');
  [[0,C.x],[1,C.y],[2,C.z]].forEach(([a,color]) => line(g, xs, w.map(p=>p[a]), map, color));
  g.fillStyle = C.text; g.font='11px system-ui';
  g.fillText('x 红 / y 绿 / z 蓝', 56, 16);
}

function drawHand(){
  const c = DATA.cycles[cycle], cv = $('hand'), g = cv.getContext('2d');
  if (!c.feedback) return empty(cv, '无执行记录');
  const f = c.feedback;
  const map = axes(g, cv, f.t, [0].concat(f.hand_error_rad[0], f.hand_error_rad[1]), 's');
  line(g, f.t, f.hand_error_rad[0], map, C.left);
  line(g, f.t, f.hand_error_rad[1], map, C.right);
}

setup();
</script>
"""


def render(run_dir: Path, client_dir: Path, output: Path) -> Path:
    payload = collect(run_dir, client_dir)
    html = (TEMPLATE.replace("__RUN__", payload["run"])
            .replace("__BONES__", json.dumps(BONES))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False)))
    output.write_text(html, encoding="utf-8")
    return output


def main() -> int:
    client = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path,
                        help="run directory; default: the newest under client/runs")
    parser.add_argument("-o", "--output", type=Path, help="default: <run>/report.html")
    args = parser.parse_args()
    run_dir = args.run
    if run_dir is None:
        candidates = sorted((client / "runs").glob("live-*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("no runs under client/runs", file=sys.stderr)
            return 1
        run_dir = candidates[0]
    run_dir = run_dir.resolve()
    output = (args.output or run_dir / "report.html").resolve()
    render(run_dir, client, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
