"""tpilot ui — one console for both agents: Data (dataset-gate) -> Plan (training agent) -> Runs.

    tpilot ui            # http://127.0.0.1:8770
The current dataset is carried between steps, every agent action shows up in the activity feed, and
nothing trains or stops without a confirmation in the page. Standard library only; binds to 127.0.0.1.
"""
from __future__ import annotations

import csv
import io
import json
import mimetypes
import subprocess
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import launch, planner, spec as specmod, status

HOME = Path.cwd()
LOCK = threading.Lock()
STATE = {"busy": None, "activity": [], "gate": None, "plan": None, "review": None}


def act(agent: str, text: str, level: str = "info"):
    with LOCK:
        STATE["activity"].append({"t": time.strftime("%H:%M:%S"), "agent": agent, "text": text, "level": level})
        STATE["activity"] = STATE["activity"][-400:]


class _Tee(io.TextIOBase):
    def __init__(self, agent):
        self.agent = agent

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                act(self.agent, line.strip())
        return len(s)


def _job(name: str, agent: str, fn, *args):
    """Run fn in a thread; one job at a time; results land in STATE by fn itself."""
    def run():
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            act(agent, f"{type(exc).__name__}: {exc}", "error")
            act(agent, traceback.format_exc(limit=2), "error")
        finally:
            with LOCK:
                STATE["busy"] = None
    with LOCK:
        if STATE["busy"]:
            return False
        STATE["busy"] = name
    threading.Thread(target=run, daemon=True).start()
    return True


# ── jobs ─────────────────────────────────────────────────────────────────────
def gate_job(p: dict):
    from dsgate import checks, report
    from dsgate.core import load
    ds, out = p["dataset"], Path(p["dataset"]) / "_dsgate"
    if p.get("mode") == "agent":
        from dsgate.agent import run_agent
        act("data", f"data agent ({p.get('llm') or 'qwen2.5:3b'}) is auditing {Path(ds).name} …")
        with redirect_stdout(_Tee("data")):
            r = run_agent(ds, p.get("llm") or "qwen2.5:3b", p.get("model") or None, "cpu", str(out), verbose=True)
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        expl = (out / "agent_report.md").read_text(encoding="utf-8") if (out / "agent_report.md").exists() else ""
        plan = (out / "fix_plan.md").read_text(encoding="utf-8") if (out / "fix_plan.md").exists() else ""
        g = {"verdict": r["verdict"], "results": rep["results"], "explanation": expl, "fix_plan": plan}
    else:
        act("data", f"quick check of {Path(ds).name} …")
        d = load(ds, p.get("manifest") or None)
        opts = {"out_dir": str(out), "model": p.get("model") or None, "device": "cpu"}
        out.mkdir(parents=True, exist_ok=True)
        res = []
        for name, fn in checks.CHECKS.items():
            try:
                r = fn(d, opts)
            except Exception as exc:  # noqa: BLE001
                r = checks.Result(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}")
            res.append(r)
            act("data", f"{r.status:4} {name}: {r.summary[:140]}", {"FAIL": "bad", "WARN": "warn"}.get(r.status, "info"))
        v = report.verdict(res)
        report.write(res, out, str(Path(ds).resolve()), v)
        g = {"verdict": v, "explanation": "", "fix_plan": "",
             "results": [{"check": r.check, "status": r.status, "summary": r.summary, "details": r.details,
                          "artifacts": r.artifacts} for r in res]}
    g.update(dataset=ds, manifest=p.get("manifest") or "", out=str(out))
    with LOCK:
        STATE["gate"] = g
    act("data", f"verdict: {g['verdict']}", "good" if g["verdict"] == "PASSED" else "bad")


def plan_job(p: dict):
    act("train", f"training agent ({p.get('llm') or 'qwen2.5:3b'}) is planning a run for {Path(p['dataset']).name} …")
    past = [status.status(d) for d in sorted((HOME / "runs").iterdir()) if d.is_dir()][-5:]
    sp, notes = planner.plan(p["dataset"], p["base"], p["goal"], p.get("name") or "run", p.get("device") or "0",
                             p.get("llm") or "qwen2.5:3b", past)
    sp.gate_manifest = p.get("manifest") or None
    sp.train_python = p.get("train_python") or None
    for n in notes:
        act("train", f"code corrected: {n}", "warn")
    with LOCK:
        STATE["plan"] = {"spec": asdict(sp), "notes": notes}
    act("train", "plan ready — review the settings, then approve", "good")


def launch_job(d: dict):
    sp = specmod.from_dict(d)
    act("train", f"launching {sp.name}: running dataset-gate first …")
    res = launch.launch(sp, HOME / "runs")
    if res["ok"]:
        act("train", f"{sp.name} started (pid {res['pid']}) — follow it in Runs", "good")
    else:
        act("train", f"NOT launched: {res['reason']}", "bad")
        for f in res.get("failed_checks", []):
            act("data", f"FAIL {f['check']}: {f['summary'][:140]}", "bad")


def review_job(run: str, llm: str):
    d = HOME / "runs" / run
    st, sp = status.status(d), specmod.load(d / "run.yaml")
    if st.get("alive") or st.get("error_tail") or not st.get("epochs_done"):
        act("train", "nothing to review yet (still running, failed or no epochs)", "warn")
        return
    act("train", f"training agent is reviewing {run} …")
    rv = planner.review(st, sp, llm or "qwen2.5:3b")
    with LOCK:
        STATE["review"] = {"run": run, "summary": rv["summary"], "notes": rv["notes"],
                           "next_spec": asdict(rv["next_spec"]) if rv["next_spec"] else None}
    act("train", "review ready — you can load its proposal into Plan", "good")


# ── runs ─────────────────────────────────────────────────────────────────────
def run_rows() -> list[dict]:
    out = []
    rd = HOME / "runs"
    for d in sorted((x for x in rd.iterdir() if x.is_dir()), reverse=True) if rd.exists() else []:
        st = status.status(d)
        hist = []
        res = d / "train" / "results.csv"
        if res.exists():
            for r in csv.DictReader(open(res, encoding="utf-8")):
                r = {k.strip(): v.strip() for k, v in r.items()}
                hist.append(float(r.get("metrics/mAP50(B)", 0) or 0))
        gate = json.loads((d / "gate.json").read_text()) if (d / "gate.json").exists() else {}
        sp = specmod.load(d / "run.yaml") if (d / "run.yaml").exists() else None
        state = ("running" if st["alive"] else "failed" if st.get("error_tail") else
                 "blocked" if gate.get("verdict") == "BLOCKED" else "finished" if st["epochs_done"] else
                 "not started")
        out.append({"name": d.name, "state": state, "epochs_done": st["epochs_done"],
                    "epochs": sp.epochs if sp else None, "best": st.get("best", {}), "map50": hist,
                    "gate": gate.get("verdict"), "error": st.get("error_tail"), "weights": st.get("weights"),
                    "device": sp.device if sp else None})
    return out


def stop_run(name: str) -> str:
    d = HOME / "runs" / name
    pid = int((d / "pid").read_text()) if (d / "pid").exists() else None
    if not pid or not status.alive(pid):
        return "not running"
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import os
        import signal
        os.kill(pid, signal.SIGTERM)
    act("train", f"stopped {name} (pid {pid}) at your request", "warn")
    return "stopped"


# ── page ─────────────────────────────────────────────────────────────────────
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Training Console</title><style>
:root{--bg:#f6f5f2;--fg:#1c1c1a;--muted:#75726b;--line:#e2dfd7;--card:#fff;--acc:#2b59d1;--accfg:#fff;--ok:#2d7a46;--warn:#a87410;--bad:#b3261e;--chip:#eeebe4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141518;--fg:#e8e6e1;--muted:#9a978f;--line:#2c2e33;--card:#1d1f23;--acc:#7ea3ff;--accfg:#0e1220;--chip:#26282d}}
:root[data-theme="dark"]{--bg:#141518;--fg:#e8e6e1;--muted:#9a978f;--line:#2c2e33;--card:#1d1f23;--acc:#7ea3ff;--accfg:#0e1220;--chip:#26282d}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:17px;margin:0} .chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:12.5px;color:var(--muted);max-width:44ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.steps{display:flex;gap:6px;margin-left:auto} .steps button{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:999px;padding:6px 14px;cursor:pointer;font:inherit}
.steps button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)} .steps .n{opacity:.7;margin-right:4px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;padding:16px 18px;max-width:1400px;margin:0 auto}
@media (max-width:980px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
h2{font-size:16px;margin:0 0 10px} .m{color:var(--muted)} label{display:block;font-size:12.5px;color:var(--muted);margin:10px 0 3px}
input,select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--fg);font:inherit}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px} .g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media (max-width:640px){.g2,.g3{grid-template-columns:1fr}}
.btn{display:inline-flex;align-items:center;gap:6px;margin:12px 8px 0 0;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);font:inherit;font-weight:600;cursor:pointer}
.btn.sec{background:transparent;color:var(--fg);border-color:var(--line);font-weight:500} .btn.danger{background:var(--bad);border-color:var(--bad);color:#fff}
.btn:disabled{opacity:.45;cursor:not-allowed}
.pill{display:inline-block;min-width:46px;text-align:center;border-radius:999px;font-size:11.5px;font-weight:600;padding:1px 8px;color:#fff;margin-right:8px}
.PASS,.good,.finished,.PASSED{background:var(--ok)} .WARN,.warn{background:var(--warn)} .FAIL,.bad,.failed,.blocked,.BLOCKED{background:var(--bad)}
.SKIP,.info,.not{background:var(--muted)} .running{background:var(--acc);color:var(--accfg)}
.row{display:flex;align-items:flex-start;gap:4px;padding:7px 0;border-bottom:1px solid var(--line)} .row:last-child{border:0}
details summary{cursor:pointer;color:var(--muted);font-size:12.5px} pre{white-space:pre-wrap;font-size:12px;margin:6px 0 0}
.verdict{font-size:22px;font-weight:700;margin:4px 0 8px} .verdict.PASSED{background:none;color:var(--ok)} .verdict.BLOCKED{background:none;color:var(--bad)}
.feed{font:12.5px/1.45 ui-monospace,Consolas,monospace;max-height:calc(100vh - 150px);overflow:auto} .feed div{padding:3px 0;border-bottom:1px dashed var(--line)}
.feed .who{font-weight:700;margin:0 6px} .feed .bad{background:none;color:var(--bad)} .feed .warn{background:none;color:var(--warn)} .feed .good{background:none;color:var(--ok)} .feed .info,.feed .error{background:none}
.feed .error{color:var(--bad)} .side{position:sticky;top:70px;align-self:start}
.busy{display:inline-flex;gap:8px;align-items:center;font-size:13px;color:var(--muted)} .dot{width:8px;height:8px;border-radius:50%;background:var(--acc);animation:p 1s infinite alternate}
@keyframes p{to{opacity:.25}} .note{border-left:3px solid var(--warn);padding:6px 10px;background:var(--chip);border-radius:6px;font-size:13px;margin:8px 0}
.hint{font-size:11.5px;color:var(--muted)} img.sheet{max-width:100%;border:1px solid var(--line);border-radius:8px;margin-top:8px}
dialog{border:1px solid var(--line);border-radius:12px;background:var(--card);color:var(--fg);max-width:520px;width:92vw} dialog::backdrop{background:rgba(0,0,0,.45)}
.runcard{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:12px 0;border-bottom:1px solid var(--line)} .runcard:last-child{border:0}
svg.spark{width:140px;height:34px} .empty{color:var(--muted);padding:12px 0}
</style></head><body>
<header><h1>Training Console</h1><span class="chip" id="dsChip">no dataset selected</span><span id="busy"></span>
<nav class="steps"><button data-s="data" class="on"><span class="n">1</span>Data</button><button data-s="plan"><span class="n">2</span>Plan</button><button data-s="runs"><span class="n">3</span>Runs</button></nav></header>
<div class="wrap"><main>
<section id="s-data"><div class="card"><h2>Check the dataset</h2><p class="m">The data agent runs dataset-gate: leakage, broken labels, missing classes, suspicious empty tiles. Training is only allowed on a dataset that passes.</p>
<label>Dataset folder (data.yaml, images/, labels/)</label><input id="dataset" placeholder="C:\path\to\dataset">
<div class="g2"><div><label>Manifest CSV (optional: frame, capture_time, lat, lon)</label><input id="manifest"></div>
<div><label>Detector .pt (optional: missing-label & class-swap checks)</label><input id="model"></div></div>
<button class="btn" onclick="gate('quick')">Quick check</button><button class="btn sec" onclick="gate('agent')">Ask the data agent (explains + fix plan)</button></div>
<div class="card" id="gateOut" hidden></div></section>

<section id="s-plan" hidden><div class="card"><h2>Plan a training run</h2><p class="m">The training agent proposes settings. Code clamps every value to a safe range; you edit and approve. Nothing starts without your confirmation.</p>
<div class="g2"><div><label>Base weights (.pt)</label><input id="base"></div><div><label>Run name</label><input id="name" value="run"></div></div>
<label>Goal (plain English)</label><input id="goal" value="High recall on small aerial animals; this GPU is shared, keep memory low">
<div class="g3"><div><label>Device</label><select id="device"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></div>
<div><label>Planner LLM (Ollama)</label><input id="llm" value="qwen2.5:3b"></div><div><label>Training Python (ultralytics + torch)</label><input id="train_python"></div></div>
<button class="btn" id="planBtn" onclick="plan()">Ask the training agent for a plan</button><span class="hint" id="planGate"></span></div>
<div class="card" id="planOut" hidden></div></section>

<section id="s-runs" hidden><div class="card"><h2>Runs</h2><p class="m">Live progress; refreshes on its own. Review a finished run to get the agent's next proposal, then load it into Plan.</p><div id="runsOut"></div></div>
<div class="card" id="reviewOut" hidden></div></section>
</main>
<aside class="side"><div class="card"><h2>Activity</h2><p class="hint">Everything both agents do, as it happens.</p><div class="feed" id="feed"><div class="m">Nothing yet.</div></div></div></aside></div>

<dialog id="confirm"><h2 id="cTitle">Confirm</h2><div id="cBody" class="m"></div>
<div style="text-align:right"><button class="btn sec" onclick="confirmDlg.close()">Cancel</button><button class="btn" id="cOk">Confirm</button></div></dialog>
<script>
const $ = id => document.getElementById(id), esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const confirmDlg = $("confirm"); let S = {}, seenAct = 0, step = "data";
const FIELDS = [["imgsz","Image size",320,1600,32],["epochs","Epochs",1,300,1],["patience","Patience",3,50,1],["batch","Batch",1,64,1],
 ["lr0","Learning rate",0.00001,0.05,0.0001],["fliplr","Flip L-R",0,1,0.05],["flipud","Flip U-D",0,1,0.05],["degrees","Rotation °",0,30,1],
 ["mosaic","Mosaic",0,1,0.05],["scale","Scale jitter",0,0.9,0.05],["min_free_vram_gb","Min free GPU GB",1,24,0.5]];
function save(){ try{ const f={}; for(const k of ["dataset","manifest","model","base","name","goal","device","llm","train_python"]) f[k]=$(k).value; localStorage.setItem("tp-form",JSON.stringify(f)); }catch(e){} }
try{ const f=JSON.parse(localStorage.getItem("tp-form")||"{}"); for(const k in f) if($(k)&&f[k]) $(k).value=f[k]; }catch(e){}
document.querySelectorAll("input,select").forEach(e=>e.addEventListener("change",()=>{save();chip();}));
function chip(){ $("dsChip").textContent = $("dataset").value ? "dataset: " + $("dataset").value : "no dataset selected"; }
function go(s){ step=s; for(const x of ["data","plan","runs"]) $("s-"+x).hidden = x!==s; document.querySelectorAll(".steps button").forEach(b=>b.classList.toggle("on",b.dataset.s===s)); if(s==="runs") loadRuns(); render(); }
document.querySelectorAll(".steps button").forEach(b=>b.onclick=()=>go(b.dataset.s));
async function api(path, body){ const r = await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const j = await r.json().catch(()=>({error:"bad response"})); if(!r.ok) toast(j.error||"request failed"); return r.ok ? j : null; }
function toast(t){ const d=document.createElement("div"); d.className="bad"; d.innerHTML=`<span class="who">ui</span>${esc(t)}`; $("feed").appendChild(d); $("feed").scrollTop=1e9; }
function ask(title, html){ return new Promise(res=>{ $("cTitle").textContent=title; $("cBody").innerHTML=html; $("cOk").onclick=()=>{confirmDlg.close();res(true)}; confirmDlg.onclose=()=>res(false); confirmDlg.showModal(); }); }
async function gate(mode){ save(); chip(); if(!$("dataset").value.trim()) return toast("enter a dataset folder first");
  await api("/api/gate",{mode,dataset:$("dataset").value.trim(),manifest:$("manifest").value.trim(),model:$("model").value.trim(),llm:$("llm").value.trim()}); }
async function plan(){ save(); if(!$("base").value.trim()) return toast("enter base weights (.pt)");
  await api("/api/plan",{dataset:$("dataset").value.trim(),manifest:$("manifest").value.trim(),base:$("base").value.trim(),goal:$("goal").value,name:$("name").value,device:$("device").value,llm:$("llm").value,train_python:$("train_python").value.trim()}); }
function gateHtml(g){
  const rows = g.results.map(r=>`<div class="row"><span class="pill ${r.status}">${r.status}</span><div><b>${esc(r.check)}</b> — ${esc(r.summary)}
   ${r.details&&Object.keys(r.details).length?`<details><summary>evidence</summary><pre>${esc(JSON.stringify(r.details,null,1)).slice(0,4000)}</pre></details>`:""}
   ${(r.artifacts||[]).filter(a=>/\.(jpg|png)$/i.test(a)).map(a=>`<img class="sheet" loading="lazy" src="/file?path=${encodeURIComponent(a)}" alt="">`).join("")}</div></div>`).join("");
  const next = g.verdict==="PASSED" ? `<button class="btn" onclick="go('plan')">Continue to Plan →</button>` :
    `<div class="note">Training is blocked until the FAIL items are fixed. Fix the data, then check again.</div>`;
  return `<div class="verdict ${g.verdict}">${g.verdict}</div><div class="m">${esc(g.dataset)}</div>${next}
   ${g.explanation?`<details open><summary>the data agent's explanation</summary><pre>${esc(g.explanation)}</pre></details>`:""}
   ${g.fix_plan?`<details><summary>fix plan (proposed, not applied)</summary><pre>${esc(g.fix_plan)}</pre></details>`:""}<div style="margin-top:10px">${rows}</div>`; }
function planHtml(p){ const s=p.spec;
  const inputs = FIELDS.map(([k,l,lo,hi,st])=>`<div><label>${l} <span class="hint">${lo}–${hi}</span></label><input type="number" data-k="${k}" min="${lo}" max="${hi}" step="${st}" value="${s[k]??""}"></div>`).join("");
  return `<h2>Proposed run: ${esc(s.name)}</h2>${p.notes.length?p.notes.map(n=>`<div class="note">code corrected: ${esc(n)}</div>`).join(""):""}
   ${s.notes?`<p class="m"><b>Agent's reasoning:</b> ${esc(s.notes)}</p>`:""}
   <div class="g3">${inputs}<div><label>Optimizer</label><select data-k="optimizer">${["AdamW","SGD","Adam","auto"].map(o=>`<option ${o===s.optimizer?"selected":""}>${o}</option>`).join("")}</select></div>
   <div><label>Time cap, hours <span class="hint">optional</span></label><input type="number" data-k="time_hours" min="0.05" max="24" step="0.25" value="${s.time_hours??""}"></div></div>
   <button class="btn" onclick="approve()">Approve & launch…</button><span class="hint">The data gate runs again at launch.</span>`; }
async function approve(){ const sp = Object.assign({}, S.plan.spec);
  document.querySelectorAll("#planOut [data-k]").forEach(e=>{ const v=e.value; sp[e.dataset.k] = e.tagName==="SELECT" ? v : (v===""?null:Number(v)); });
  const ok = await ask("Launch this run?", `<p><b>${esc(sp.name)}</b> on <b>${esc(sp.device==="cpu"?"CPU":"GPU "+sp.device)}</b></p>
    <p>${sp.epochs} epochs max (stops early after ${sp.patience} without improvement)${sp.time_hours?`, at most ${sp.time_hours} h`:""}; image size ${sp.imgsz}, batch ${sp.batch}, lr ${sp.lr0}.</p>
    <p>Dataset: ${esc(sp.dataset)}<br>dataset-gate runs first; a BLOCKED dataset will not train. On GPU it waits for at least ${sp.min_free_vram_gb} GB free.</p>`);
  if(ok){ await api("/api/launch", sp); setTimeout(()=>go("runs"), 900); } }
function spark(v){ if(!v.length) return ""; const mx=Math.max(...v,0.01), w=140,h=34, pts=v.map((y,i)=>`${(i/(Math.max(v.length-1,1)))*w},${h-2-(y/mx)*(h-4)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="mAP50 per epoch"><polyline fill="none" stroke="currentColor" stroke-width="2" points="${pts}"/></svg>`; }
async function loadRuns(){ const j = await api("/api/runs"); if(!j) return;
  $("runsOut").innerHTML = j.runs.length ? j.runs.map(r=>`<div class="runcard"><div><span class="pill ${r.state.split(" ")[0]}">${r.state}</span><b>${esc(r.name)}</b>
     <div class="m">epoch ${r.epochs_done}${r.epochs?"/"+r.epochs:""} · ${esc(r.device==="cpu"?"CPU":"GPU")} · gate ${esc(r.gate||"–")}${r.best&&r.best["mAP50(B)"]!==undefined?` · best mAP50 ${r.best["mAP50(B)"]} · recall ${r.best["recall(B)"]}`:""}</div>
     ${r.error?`<details><summary>error</summary><pre>${esc(r.error)}</pre></details>`:""}${r.weights?`<div class="hint">weights: ${esc(r.weights)}</div>`:""}</div>
     <div style="text-align:right">${spark(r.map50)}<div>${r.state==="running"?`<button class="btn danger" onclick="stopRun('${r.name}')">Stop…</button>`:""}
     ${r.state==="finished"?`<button class="btn sec" onclick="review('${r.name}')">Review</button>`:""}</div></div></div>`).join("") : `<div class="empty">No runs yet — plan one in step 2.</div>`; }
async function stopRun(n){ if(await ask("Stop this run?", `<p><b>${esc(n)}</b> will be killed. Its last saved weights stay on disk.</p>`)){ await api("/api/stop",{run:n}); loadRuns(); } }
async function review(n){ await api("/api/review",{run:n,llm:$("llm").value}); }
function loadProposal(){ if(!S.review||!S.review.next_spec) return; S.plan = {spec:S.review.next_spec, notes:S.review.notes||[]}; $("planOut").hidden=false; $("planOut").innerHTML=planHtml(S.plan); go("plan"); }
function render(){
  $("busy").innerHTML = S.busy ? `<span class="busy"><span class="dot"></span>${esc(S.busy)} …</span>` : "";
  document.querySelectorAll(".btn").forEach(b=>{ if(!b.closest("dialog")&&!b.classList.contains("danger")) b.disabled = !!S.busy; });
  if(S.gate){ $("gateOut").hidden=false; if($("gateOut").dataset.v!==JSON.stringify([S.gate.verdict,S.gate.dataset,S.gate.results.length,S.gate.explanation.length])){ $("gateOut").innerHTML=gateHtml(S.gate); $("gateOut").dataset.v=JSON.stringify([S.gate.verdict,S.gate.dataset,S.gate.results.length,S.gate.explanation.length]); } }
  const passed = S.gate && S.gate.verdict==="PASSED" && S.gate.dataset===$("dataset").value.trim();
  $("planGate").textContent = passed ? "✓ dataset passed the gate" : "Check the dataset in step 1 first (it is re-checked at launch anyway).";
  if(S.plan && $("planOut").dataset.n !== S.plan.spec.name+S.plan.notes.length+(S.plan.spec.notes||"").length){ $("planOut").hidden=false; $("planOut").innerHTML=planHtml(S.plan); $("planOut").dataset.n=S.plan.spec.name+S.plan.notes.length+(S.plan.spec.notes||"").length; }
  if(S.review){ $("reviewOut").hidden=false; $("reviewOut").innerHTML=`<h2>Review of ${esc(S.review.run)}</h2><pre>${esc(S.review.summary)}</pre>${S.review.next_spec?`<button class="btn" onclick="loadProposal()">Load the proposed next run into Plan →</button>`:"<p class='m'>No next run proposed.</p>"}`; }
}
async function poll(){ const j = await api("/api/state"); if(j){ S=j;
   const ev=j.activity.slice(seenAct); if(seenAct===0&&ev.length) $("feed").innerHTML=""; seenAct=j.activity.length;
   for(const e of ev){ const d=document.createElement("div"); d.className=e.level; d.innerHTML=`<span class="m">${e.t}</span><span class="who">${e.agent==="data"?"data agent":e.agent==="train"?"training agent":esc(e.agent)}</span>${esc(e.text)}`; $("feed").appendChild(d); }
   if(ev.length) $("feed").scrollTop=1e9; render(); }
  if(step==="runs") loadRuns(); setTimeout(poll, S.busy?800:2500); }
chip(); poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj, ctype: str = "application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict | None:
        try:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def _allowed(self, p: Path) -> bool:
        """Files may be served only from this console's runs/plans or a dataset's _dsgate report folder."""
        p = p.resolve()
        roots = [(HOME / "runs").resolve(), (HOME / "plans").resolve()]
        with LOCK:
            g = STATE.get("gate")
        if g:
            roots.append(Path(g["out"]).resolve())
        return any(r == p or r in p.parents for r in roots) and p.is_file()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            with LOCK:
                return self._send(200, STATE)
        if u.path == "/api/runs":
            return self._send(200, {"runs": run_rows()})
        if u.path == "/file":
            p = Path(unquote(parse_qs(u.query).get("path", [""])[0]))
            if not self._allowed(p):
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path, b = urlparse(self.path).path, self._body()
        if b is None:
            return self._send(400, {"error": "malformed request"})
        if path == "/api/gate":
            if not (Path(b.get("dataset", "")) / "images").is_dir():
                return self._send(400, {"error": "that folder has no images/ — is it a YOLO dataset?"})
            ok = _job("checking the dataset", "data", gate_job, b)
        elif path == "/api/plan":
            if not (Path(b.get("dataset", "")) / "data.yaml").is_file():
                return self._send(400, {"error": "pick a dataset (with data.yaml) in step 1 first"})
            if not str(b.get("base", "")).endswith(".pt"):
                return self._send(400, {"error": "base weights must be a .pt file"})
            ok = _job("planning a run", "train", plan_job, b)
        elif path == "/api/launch":
            ok = _job("launching", "train", launch_job, b)
        elif path == "/api/review":
            ok = _job("reviewing a run", "train", review_job, b.get("run", ""), b.get("llm", ""))
        elif path == "/api/stop":
            return self._send(200, {"result": stop_run(str(b.get("run", "")))})
        else:
            return self._send(404, {"error": "not found"})
        return self._send(200 if ok else 409, {"ok": ok} if ok else {"error": "an agent is already working — wait for it to finish"})


def serve(port: int = 8770, home: Path | None = None):
    global HOME
    HOME = Path(home or Path.cwd()).resolve()
    (HOME / "runs").mkdir(parents=True, exist_ok=True)
    (HOME / "plans").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training Console on http://127.0.0.1:{port}  (home: {HOME})  Ctrl-C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
