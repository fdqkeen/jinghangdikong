# -*- coding: utf-8 -*-
"""
生成单文件交互原型「静航·低空 监管驾驶舱」
- 场景与模型参数由已验证的 sim/ 模块导出，保证与仿真结果一致
- 输出：submit/静航低空_监管驾驶舱.html（双击即开，无需 Python 环境）
用法：python make_demo.py
"""
import os
import io
import json
import base64
import numpy as np

import config as C
import scene as scene_mod
import noise as noise_mod
import route as route_mod
import isac as isac_mod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "submit"))
os.makedirs(OUT, exist_ok=True)

print("[1/4] 构建场景")
scene = scene_mod.build_scene()
grid = noise_mod.make_receiver_grid(scene, step=1)

print("[2/4] 感知层预计算（三层聚合，口径与正文一致）")
maps = isac_mod.coverage_maps(scene, alt_layers=C.ALT_LAYERS)
m100 = maps[100.0]
kpi = isac_mod.coverage_kpi(maps)          # 与 run_all.py 完全相同的聚合口径

print("[3/4] 航路优化")
CACHE = os.path.join(HERE, "..", "outputs", "opt_waypoints.json")
base = route_mod.baseline_route(grid, scene, cruise_alt=80.0, night=True, n_samples=120)
if os.path.exists(CACHE):
    wp = np.array(json.load(open(CACHE, encoding="utf-8"))["optWP"])
    bal = route_mod.evaluate(wp[1:-1].ravel(), grid, scene, night=True,
                             n_samples=120, full=True)
    print("  读取缓存的优化航路")
else:
    P, F = route_mod.nsga2(grid, scene, seed=C.SEED, pop_size=C.NSGA2_POP,
                           n_gen=C.NSGA2_GEN, night=True)
    ok = np.where(F[:, 0] <= base["length"] * 1.15)[0]
    i_c = int(ok[np.argmin(F[ok, 1])]) if len(ok) else int(np.argmin(F[:, 1]))
    bal = route_mod.evaluate(P[i_c], grid, scene, night=True, n_samples=120, full=True)
    json.dump({"optWP": [[float(v) for v in p] for p in bal["waypoints"]]},
              open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  已缓存优化航路")
print(f"  基线 {base['length']:.0f} m / {base['exposure']:.0f} 人·dB")
print(f"  优化 {bal['length']:.0f} m / {bal['exposure']:.0f} 人·dB "
      f"(降 {(1-bal['exposure']/base['exposure'])*100:.1f}%)")

print("[4/4] 生成 HTML")


def enc(arr, dtype):
    a = np.asarray(arr, dtype=dtype)
    return base64.b64encode(a.tobytes()).decode("ascii")


payload = dict(
    N=int(scene["n"]), CELL=float(scene["cell"]), AREA=float(C.AREA_M),
    bh=enc(scene["building_h"], np.float32), bh_dtype="f32",
    zone=enc(scene["zone"], np.uint8), zone_dtype="u8",
    pop=enc(grid["pop"], np.float32), pop_dtype="f32",
    bs=enc(scene["bs_pos"][:, :2], np.float32), bs_dtype="f32",
    nodes=enc(scene["node_pos"][:, :2], np.float32), nodes_dtype="f32",
    vert=enc(scene["vertiports"][:, :2], np.float32), vert_dtype="f32",
    nofly=[[float(a), float(b), float(c)] for a, b, c in scene["no_fly"]],
    pd=enc(m100["pd"], np.float32), pd_dtype="f32",
    sh=enc(np.clip(m100["sig_h"], 0, 12), np.float32), sh_dtype="f32",
    limits={"0": [50, 40], "1": [55, 45], "2": [60, 50], "3": [65, 55], "4": [70, 55]},
    aircraft=[
        {"name": "微型机（Mini 类）", "lw": 91.0, "lwh": 86.0},
        {"name": "商用多旋翼", "lw": C.LW_CRUISE_DBA, "lwh": C.LW_HOVER_DBA},
        {"name": "eVTOL", "lw": C.LW_EVTOL_DBA, "lwh": C.LW_EVTOL_DBA - 5.0},
        {"name": "传统直升机", "lw": C.LW_HELI_DBA, "lwh": C.LW_HELI_DBA - 5.0},
    ],
    baseWP=[[float(v) for v in p] for p in base["waypoints"]],
    optWP=[[float(v) for v in p] for p in bal["waypoints"]],
    alpha=float(C.ALPHA_DB_PER_KM),
    kpi={k: float(v) for k, v in kpi.items()},
    recon={"hybrid": 2.76, "phys": 3.03, "interp": 3.63},
)
payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>静航·低空 — 城市低空监管驾驶舱</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6f8;color:#2c2c2a;padding:18px}
.wrap{max-width:1240px;margin:0 auto}
header{background:#fff;border:1px solid #e3e5e8;border-radius:12px;padding:16px 20px;margin-bottom:14px}
h1{font-size:19px;font-weight:600;color:#0C447C}
.sub{font-size:12.5px;color:#5F5E5A;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
.kpi{background:#fff;border:1px solid #e3e5e8;border-radius:10px;padding:12px 14px}
.kpi .v{font-size:21px;font-weight:600;color:#185FA5}
.kpi .l{font-size:11.5px;color:#5F5E5A;margin-top:3px}
.kpi .t{font-size:10.5px;color:#888780;margin-top:2px}
.main{display:grid;grid-template-columns:1fr 330px;gap:14px}
.card{background:#fff;border:1px solid #e3e5e8;border-radius:12px;padding:14px}
.mapcard{display:flex;flex-direction:column;align-items:center}
canvas{border-radius:8px;cursor:crosshair;max-width:100%}
.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.tab{padding:6px 12px;border:1px solid #d3d1c7;border-radius:7px;font-size:12.5px;cursor:pointer;background:#fff;color:#5F5E5A}
.tab.on{background:#185FA5;color:#fff;border-color:#185FA5}
.ctl{margin-bottom:13px}
.ctl label{display:block;font-size:12px;color:#5F5E5A;margin-bottom:5px}
.ctl input[type=range]{width:100%}
.val{float:right;color:#185FA5;font-weight:600}
select{width:100%;padding:6px 8px;border:1px solid #d3d1c7;border-radius:7px;font-size:12.5px;background:#fff}
.seg{display:flex;gap:0;border:1px solid #d3d1c7;border-radius:7px;overflow:hidden}
.seg button{flex:1;padding:6px;border:0;background:#fff;font-size:12.5px;cursor:pointer;color:#5F5E5A}
.seg button.on{background:#185FA5;color:#fff}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 4px;text-align:right;border-bottom:1px solid #eeeff0}
th:first-child,td:first-child{text-align:left}
th{color:#5F5E5A;font-weight:500}
.good{color:#1D9E75;font-weight:600}
.bad{color:#D85A30;font-weight:600}
.note{font-size:11px;color:#888780;line-height:1.6;margin-top:10px;padding-top:10px;border-top:1px solid #eeeff0}
.legend{display:flex;align-items:center;gap:8px;font-size:11px;color:#5F5E5A;margin-top:8px}
.bar{width:150px;height:11px;border-radius:3px}
h3{font-size:13px;font-weight:600;color:#2c2c2a;margin-bottom:9px}
.rw{display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid #f0f1f2}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>静航 · 低空 — 城市低空飞行器监管驾驶舱</h1>
  <div class="sub">基于 6G 通感算智融合 · 感知 → 噪声场重构 → 安全-效率-静音航路优化 · 交互原型（数值由仿真内核导出，与提交包一致）</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v" id="k1">—</div><div class="l">检测概率达标率</div><div class="t">IMT-2030 要求 ≥95%</div></div>
  <div class="kpi"><div class="v" id="k2">—</div><div class="l">定位精度 水平/垂直</div><div class="t">要求 ≤5 m / ≤8 m</div></div>
  <div class="kpi"><div class="v" id="k3">—</div><div class="l">噪声场重构误差</div><div class="t">目标 ≤3 dB</div></div>
  <div class="kpi"><div class="v" id="k4">—</div><div class="l">噪声暴露下降</div><div class="t">目标 ≥30%</div></div>
  <div class="kpi"><div class="v" id="k5">—</div><div class="l">受影响人口</div><div class="t">相对基线</div></div>
</div>

<div class="main">
  <div class="card mapcard">
    <div class="tabs">
      <div class="tab on" data-m="scene">场景</div>
      <div class="tab" data-m="pd">感知覆盖</div>
      <div class="tab" data-m="sig">定位精度</div>
      <div class="tab" data-m="noise">噪声场</div>
      <div class="tab" data-m="exceed">超标量</div>
      <div class="tab" data-m="cmp">基线 vs 优化</div>
    </div>
    <canvas id="cv" width="560" height="560"></canvas>
    <div class="legend"><span id="lg0">低</span><div class="bar" id="lgbar"></div><span id="lg1">高</span><span id="lgu" style="margin-left:10px"></span></div>
  </div>

  <div>
    <div class="card" style="margin-bottom:14px">
      <h3>仿真参数</h3>
      <div class="ctl"><label>机型 / 声功率级 <span class="val" id="vac"></span></label>
        <select id="ac"></select></div>
      <div class="ctl"><label>巡航高度 <span class="val" id="valt">80 m</span></label>
        <input type="range" id="alt" min="40" max="300" step="10" value="80"></div>
      <div class="ctl"><label>评价时段</label>
        <div class="seg"><button id="bday">昼间</button><button id="bnight" class="on">夜间</button></div></div>
      <div class="ctl"><label>声学校准节点 <span class="val" id="vnode">40</span></label>
        <input type="range" id="node" min="10" max="100" step="10" value="40"></div>
      <div class="ctl"><label>显示图层叠加</label>
        <div class="seg"><button id="bbs" class="on">基站</button><button id="bnf" class="on">禁飞区</button></div></div>
    </div>

    <div class="card" style="margin-bottom:14px">
      <h3>单架次效果（当前参数）</h3>
      <table>
        <tr><th>指标</th><th>基线(80m直线)</th><th>优化航路</th></tr>
        <tr><td>航程 (m)</td><td id="r1a">—</td><td id="r1b">—</td></tr>
        <tr><td>噪声暴露 (人·dB)</td><td id="r2a">—</td><td id="r2b">—</td></tr>
        <tr><td>受影响人口</td><td id="r3a">—</td><td id="r3b">—</td></tr>
        <tr><td>超标面积 (km²)</td><td id="r4a">—</td><td id="r4b">—</td></tr>
        <tr><td>最大超标 (dB)</td><td id="r5a">—</td><td id="r5b">—</td></tr>
      </table>
      <div class="note" id="verdict"></div>
    </div>

    <div class="card">
      <h3>工程结论</h3>
      <div class="rw"><span>提高巡航高度（80→300 m）</span><span class="good" id="c1">—</span></div>
      <div class="rw"><span>复用 6G 站址新增感知成本</span><span class="good">≈ 0</span></div>
      <div class="rw"><span>声学节点密度</span><span id="c2">—</span></div>
      <div class="rw"><span>取证链</span><span class="good">轨迹+声学+时间戳</span></div>
      <div class="note">
        本页数值由提交包内 <code>sim/</code> 仿真内核导出，属设定参数下的仿真结果，非外场实测；
        参数分级与溯源见《数据溯源与可信度分级》。
      </div>
    </div>
  </div>
</div>
</div>

<script>
const D = __PAYLOAD__;
const N=D.N, CELL=D.CELL, AREA=D.AREA;

function dec(b64, dtype){
  const bin = atob(b64); const len = bin.length;
  const buf = new ArrayBuffer(len); const v = new Uint8Array(buf);
  for(let i=0;i<len;i++) v[i]=bin.charCodeAt(i);
  if(dtype==="f32") return new Float32Array(buf);
  return new Uint8Array(buf);
}
const BH = dec(D.bh,"f32"), ZONE = dec(D.zone,"u8"), POP = dec(D.pop,"f32");
const BS = dec(D.bs,"f32"), NODES = dec(D.nodes,"f32"), VERT = dec(D.vert,"f32");
const PD = dec(D.pd,"f32"), SH = dec(D.sh,"f32");

const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
const SC = cv.width / (N*CELL);
let mode="scene", night=true, lw=D.aircraft[1].lw, lwh=D.aircraft[1].lwh,
    altV=80, nNode=40, showBS=true, showNF=true;
const NSAMP=120;   // 与 Python 仿真内核一致
const HOVER_ALT=60; // 低于此高度按悬停声功率级计（与 route.py CRUISE_SWITCH_ALT 一致）

// 基线航路：起降点间直线 + 给定巡航高度（结构与 route.baseline_route 一致）
function baseWP(alt){
  const ax=400, ay=600, bx=3600, by=3400;
  return [[ax,ay,30],[ax+(bx-ax)*0.15, ay+(by-ay)*0.15, alt],
          [ax+(bx-ax)*0.85, ay+(by-ay)*0.85, alt],[bx,by,30]];
}

const ZC = [[238,243,222],[186,213,244],[250,238,218],[241,239,232],[234,235,232]];
function zoneRGB(z){return ZC[z]||[240,240,240];}
function lerpColor(t, stops){
  t=Math.max(0,Math.min(1,t));
  for(let i=0;i<stops.length-1;i++){
    const a=stops[i], b=stops[i+1];
    if(t>=a[0]&&t<=b[0]){
      const k=(t-a[0])/((b[0]-a[0])||1);
      return [Math.round(a[1]+(b[1]-a[1])*k),Math.round(a[2]+(b[2]-a[2])*k),Math.round(a[3]+(b[3]-a[3])*k)];
    }
  }
  return [stops[stops.length-1][1],stops[stops.length-1][2],stops[stops.length-1][3]];
}
const HEAT=[[0,255,255,204],[0.25,145,207,235],[0.5,250,199,117],[0.75,216,90,48],[1,120,25,25]];
const VIR =[[0,68,1,84],[0.33,59,82,139],[0.66,33,145,140],[1,253,231,37]];
const MAG =[[0,0,0,4],[0.33,81,18,124],[0.66,183,55,121],[1,252,253,191]];

function lpField(sx,sy,hs,lwv){
  const o=new Float32Array(N*N); const dz=Math.max(hs-1.5,1.0);
  for(let i=0;i<N;i++){ const x=(i+0.5)*CELL;
    for(let j=0;j<N;j++){ const y=(j+0.5)*CELL;
      const dx=x-sx, dy=y-sy; const R=Math.sqrt(dx*dx+dy*dy+dz*dz);
      o[i*N+j]= lwv - 20*Math.log10(R) - 11 - D.alpha*R/1000;
    }}
  return o;
}
function sampleTrack(wp,n){
  const seg=[]; let tot=0;
  for(let i=0;i<wp.length-1;i++){
    const d=Math.hypot(wp[i+1][0]-wp[i][0],wp[i+1][1]-wp[i][1],wp[i+1][2]-wp[i][2]);
    seg.push(d); tot+=d;
  }
  const out=[]; const cum=[0];
  for(let i=0;i<seg.length;i++) cum.push(cum[i]+seg[i]);
  for(let k=0;k<n;k++){
    const t=tot*k/(n-1); let i=0;
    while(i<seg.length-1 && cum[i+1]<t) i++;
    const f=seg[i]>0?(t-cum[i])/seg[i]:0;
    out.push([wp[i][0]+(wp[i+1][0]-wp[i][0])*f, wp[i][1]+(wp[i+1][1]-wp[i][1])*f, wp[i][2]+(wp[i+1][2]-wp[i][2])*f]);
  }
  return out;
}
function limitAt(idx){
  const z=ZONE[idx];
  const L=D.limits[String(z)]||[55,45];
  return night?L[1]:L[0];
}
function evalWP(wp, lwv, nsamp, lwhv){
  const tr=sampleTrack(wp,nsamp);
  const lmax=new Float32Array(N*N).fill(-999);
  let len=0;
  for(let i=0;i<wp.length-1;i++) len+=Math.hypot(wp[i+1][0]-wp[i][0],wp[i+1][1]-wp[i][1],wp[i+1][2]-wp[i][2]);
  for(const p of tr){
    const w=(lwhv!==undefined && p[2]<HOVER_ALT)?lwhv:lwv;
    const f=lpField(p[0],p[1],p[2], w);
    for(let k=0;k<N*N;k++) if(f[k]>lmax[k]) lmax[k]=f[k];
  }
  let expo=0, aff=0, area=0, mx=0;
  const ca=(CELL*CELL)/1e6;
  for(let k=0;k<N*N;k++){
    const e=Math.max(0, lmax[k]-limitAt(k));
    if(e>0){expo+=POP[k]*e; aff+=POP[k]; area+=ca; if(e>mx)mx=e;}
  }
  return {lmax, len, expo, aff, area, mx, tr};
}

let cache={};
function computeAll(){
  const key=lw+"|"+altV+"|"+(night?1:0);
  if(cache.key!==key){
    cache={key, base:evalWP(baseWP(altV),lw,NSAMP,lwh),
                opt:evalWP(D.optWP,lw,NSAMP,lwh)};
  }
  return cache;
}

function draw(){
  const R=computeAll();
  const img=ctx.createImageData(N*CELL*SC|0, N*CELL*SC|0);
  const W=img.width;
  let unit="", lo="", hi="", stops=HEAT;
  for(let py=0;py<W;py++){
    for(let px=0;px<W;px++){
      const i=Math.min(N-1,Math.floor(px/SC/CELL)), j=Math.min(N-1,Math.floor(py/SC/CELL));
      const k=i*N+j; let c;
      if(mode==="scene"){
        const h=BH[k]/120;
        const z=zoneRGB(ZONE[k]);
        const g=Math.round(255-70*h);
        c=[Math.round(z[0]*0.45+g*0.55),Math.round(z[1]*0.45+g*0.55),Math.round(z[2]*0.45+g*0.55)];
      } else if(mode==="pd"){ c=lerpColor((PD[k]-0.5)/0.5,HEAT); }
      else if(mode==="sig"){ c=lerpColor(SH[k]/10,VIR); }
      else if(mode==="noise"){ c=lerpColor((R.base.lmax[k]-25)/50,MAG); }
      else if(mode==="exceed"){ c=lerpColor(Math.max(0,R.base.lmax[k]-limitAt(k))/25,HEAT); }
      else {
        const d=R.base.lmax[k]-R.opt.lmax[k];
        const t=(d+30)/60;
        c=lerpColor(t,[[0,216,90,48],[0.5,245,245,245],[1,29,158,117]]);
      }
      const o=(py*W+px)*4;
      img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=255;
    }
  }
  ctx.putImageData(img,0,0);

  if(mode==="scene"||mode==="cmp"){
    if(showNF){
      ctx.strokeStyle="#D85A30"; ctx.setLineDash([5,4]); ctx.lineWidth=1.4;
      for(const f of D.nofly){ ctx.beginPath(); ctx.arc(f[0]*SC,f[1]*SC,f[2]*SC,0,6.2832); ctx.stroke(); }
      ctx.setLineDash([]);
    }
  }
  if(showBS){
    ctx.fillStyle="rgba(24,95,165,0.9)";
    for(let i=0;i<BS.length/2;i++){ ctx.beginPath(); ctx.arc(BS[2*i]*SC,BS[2*i+1]*SC,2.6,0,6.2832); ctx.fill(); }
  }
  ctx.fillStyle="#1D9E75"; ctx.strokeStyle="#fff"; ctx.lineWidth=0.8;
  for(let i=0;i<nNode && i<NODES.length/2;i++){ ctx.beginPath(); ctx.rect(NODES[2*i]*SC-2.6,NODES[2*i+1]*SC-2.6,5.2,5.2); ctx.fill(); ctx.stroke(); }
  ctx.fillStyle="#BA7517"; ctx.strokeStyle="#fff"; ctx.lineWidth=1;
  for(let i=0;i<VERT.length/2;i++){
    ctx.beginPath();
    ctx.moveTo(VERT[2*i]*SC,VERT[2*i+1]*SC-7); ctx.lineTo(VERT[2*i]*SC+6,VERT[2*i+1]*SC+5);
    ctx.lineTo(VERT[2*i]*SC-6,VERT[2*i+1]*SC+5); ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  if(mode==="cmp"||mode==="noise"||mode==="exceed"){
    for(const [wp,col,dash] of [[baseWP(altV),"#D85A30",[7,5]],[D.optWP,"#185FA5",[]]]){
      ctx.strokeStyle=col; ctx.lineWidth=2.6; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(wp[0][0]*SC,wp[0][1]*SC);
      for(let i=1;i<wp.length;i++) ctx.lineTo(wp[i][0]*SC,wp[i][1]*SC);
      ctx.stroke(); ctx.setLineDash([]);
    }
  }
  if(mode==="scene"){lo="低";hi="高（建筑）";stops=[[0,250,248,240],[1,150,90,40]];}
  else if(mode==="pd"){lo="0.5";hi="1.0";stops=HEAT;unit="检测概率";}
  else if(mode==="sig"){lo="0";hi="10 m";stops=VIR;unit="σ_h";}
  else if(mode==="noise"){lo="25";hi="75 dB(A)";stops=MAG;unit="A 声级";}
  else if(mode==="exceed"){lo="0";hi="25 dB";stops=HEAT;unit="超标量";}
  else {lo="优化后更吵";hi="优化后更静";stops=[[0,216,90,48],[0.5,245,245,245],[1,29,158,117]];unit="声级差";}

  const bar=document.getElementById("lgbar");
  let g="linear-gradient(90deg";
  for(let t=0;t<=1.0001;t+=0.1){const c=lerpColor(t,stops);g+=",rgb("+c[0]+","+c[1]+","+c[2]+")";}
  bar.style.background=g+")";
  document.getElementById("lg0").textContent=lo;
  document.getElementById("lg1").textContent=hi;
  document.getElementById("lgu").textContent=unit;
  return R;
}

function refresh(){
  const R=draw();
  const b=R.base,o=R.opt;
  const f=(x)=>x.toLocaleString("zh-CN",{maximumFractionDigits:0});
  document.getElementById("r1a").textContent=f(b.len); document.getElementById("r1b").textContent=f(o.len);
  document.getElementById("r2a").textContent=f(b.expo); document.getElementById("r2b").textContent=f(o.expo);
  document.getElementById("r3a").textContent=f(b.aff); document.getElementById("r3b").textContent=f(o.aff);
  document.getElementById("r4a").textContent=b.area.toFixed(2); document.getElementById("r4b").textContent=o.area.toFixed(2);
  document.getElementById("r5a").textContent=b.mx.toFixed(1); document.getElementById("r5b").textContent=o.mx.toFixed(1);
  const cut=(1-o.expo/b.expo)*100, dl=(o.len/b.len-1)*100;
  const cutEl=document.getElementById("k4");
  cutEl.textContent=(cut>0?"-":"")+Math.abs(cut).toFixed(1)+"%";
  cutEl.className="v "+(cut>=30?"":"");
  cutEl.style.color=cut>=30?"#1D9E75":"#D85A30";
  document.getElementById("k5").textContent=f(o.aff)+" 人";
  const v=document.getElementById("verdict");
  v.innerHTML="优化航路：航程 <b>"+(dl>=0?"+":"")+dl.toFixed(1)+"%</b>，"
    +"噪声暴露 <b style='color:"+(cut>=30?"#1D9E75":"#D85A30")+"'>-"+cut.toFixed(1)+"%</b>"
    +(cut>=30?"　✓ 满足 K8（≥30%）":"　✗ 未达 K8");
  const c1=document.getElementById("c1");
  const e80=evalWP(baseWP(80),lw,NSAMP,lwh).expo;
  const e300=evalWP(baseWP(300),lw,NSAMP,lwh).expo;
  c1.textContent="-"+((1-e300/e80)*100).toFixed(0)+"% 暴露";
  document.getElementById("c2").textContent=nNode+" 个 / "+(N*N)+" 栅格 ("+(nNode/(N*N)*100).toFixed(2)+"%)";
}

document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); mode=t.dataset.m; refresh();
});
const acSel=document.getElementById("ac");
D.aircraft.forEach((a,i)=>{const o=document.createElement("option");o.value=i;o.textContent=a.name+"  Lw="+a.lw.toFixed(1)+" dB(A)";acSel.appendChild(o);});
acSel.value=1;
acSel.onchange=()=>{lw=D.aircraft[+acSel.value].lw;lwh=D.aircraft[+acSel.value].lwh;
  document.getElementById("vac").textContent="Lw="+lw.toFixed(1)+" dB(A)";refresh();};
document.getElementById("vac").textContent="Lw="+lw.toFixed(1)+" dB(A)";
document.getElementById("alt").oninput=(e)=>{altV=+e.target.value;document.getElementById("valt").textContent=altV+" m";refresh();};
document.getElementById("node").oninput=(e)=>{nNode=+e.target.value;document.getElementById("vnode").textContent=nNode;refresh();};
document.getElementById("bday").onclick=()=>{night=false;document.getElementById("bday").classList.add("on");document.getElementById("bnight").classList.remove("on");refresh();};
document.getElementById("bnight").onclick=()=>{night=true;document.getElementById("bnight").classList.add("on");document.getElementById("bday").classList.remove("on");refresh();};
document.getElementById("bbs").onclick=(e)=>{showBS=!showBS;e.target.classList.toggle("on",showBS);refresh();};
document.getElementById("bnf").onclick=(e)=>{showNF=!showNF;e.target.classList.toggle("on",showNF);refresh();};

document.getElementById("k1").textContent=(D.kpi.coverage_pd*100).toFixed(1)+"%";
document.getElementById("k2").textContent=D.kpi.median_sig_h.toFixed(2)+" / "+D.kpi.median_sig_v.toFixed(2)+" m";
document.getElementById("k3").textContent=D.recon.hybrid.toFixed(2)+" dB";
refresh();
</script>
</body>
</html>
"""

html = HTML.replace("__PAYLOAD__", payload_json)
p = os.path.join(OUT, "静航低空_监管驾驶舱.html")
with io.open(p, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n完成 -> {p}  ({len(html)/1024:.0f} KB)")
