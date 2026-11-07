"""tpilot ui — one console for both agents: Data (dataset-gate) -> Plan (training agent) -> Runs.

    tpilot ui            # http://127.0.0.1:8770
A live card shows which agent is working, its stage, what the model is writing and why; the activity feed keeps
the history of both agents and of what you did. Drop a dataset folder / .zip, a manifest or weights onto the
fields (or paste paths). Your rules file is shown and enforced; nothing trains or stops without a confirmation in
the page. Standard library only; binds to 127.0.0.1. Shared widgets come from dsgate.web.
"""
from __future__ import annotations

import csv
import json
import mimetypes
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dsgate import rules as rules_mod
from dsgate.web import KIT_CSS, KIT_JS, Live, finish_upload, save_upload

from . import launch, planner, spec as specmod, status

HOME = Path.cwd()
LIVE = Live()
STATE = {"gate": None, "plan": None, "review": None}


class _Cancelled(Exception):
    pass


def _job(job: str, agent: str, stages: list[str], fn, *args, cancellable: bool = False) -> bool:
    """One job at a time, in a thread; the live card follows it."""
    if not LIVE.start(job, agent, stages, cancellable):
        return False

    def run():
        try:
            fn(*args)
        except _Cancelled:
            LIVE.act("you", "stopped at your request — nothing was changed", "warn", "you")
        except Exception as exc:  # noqa: BLE001
            LIVE.act(agent, f"error: {type(exc).__name__}: {exc}", "bad", "warn")
            LIVE.act(agent, traceback.format_exc(limit=2), "bad", "note")
        finally:
            LIVE.finish()
    threading.Thread(target=run, daemon=True).start()
    return True


# ── jobs ─────────────────────────────────────────────────────────────────────
def gate_job(p: dict):
    from dsgate import checks, report
    from dsgate.core import load
    ds, out = p["dataset"], Path(p["dataset"]) / "_dsgate"
    expl, plan = "", ""
    if p.get("mode") == "agent":
        from dsgate.agent import Cancelled, run_agent
        try:
            r = run_agent(ds, p.get("llm") or "qwen2.5:3b", p.get("model") or None, "cpu", str(out), verbose=False,
                          on_event=LIVE.on_event("data"), should_stop=LIVE.should_stop,
                          manifest=p.get("manifest") or None)
        except Cancelled:
            raise _Cancelled() from None
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        results, v, notes = rep["results"], r["verdict"], rep.get("rules_applied", [])
        expl = r["facts"] + (f"\n\nThe model's explanation:\n{r['explanation']}" if r["explanation"] else
                             f"\n\n(model's explanation dropped: {r['explanation_dropped']})"
                             if r["explanation_dropped"] else "")
        if r["skipped_by_model"]:
            expl += f"\n\nThe model skipped {', '.join(r['skipped_by_model'])}; code ran them."
        plan = (out / "fix_plan.md").read_text(encoding="utf-8") if (out / "fix_plan.md").exists() else ""
    else:
        LIVE.stage("Load", f"reading {Path(ds).name}")
        d = load(ds, p.get("manifest") or None)
        out.mkdir(parents=True, exist_ok=True)
        LIVE.stage("Run checks", "running every check")

        def each(r):
            LIVE.act("data", f"{r.status} {r.check}: {r.summary[:150]}",
                     {"FAIL": "bad", "WARN": "warn"}.get(r.status, "info"), "result")
            if LIVE.should_stop():
                raise _Cancelled()
        res, notes = checks.run(d, {"out_dir": str(out), "model": p.get("model") or None, "device": "cpu"},
                                on_result=each)
        for n in notes:
            LIVE.act("data", f"your rule applied: {n}", "info", "stage")
        v = report.verdict(res)
        report.write(res, out, str(Path(ds).resolve()), v, notes)
        LIVE.stage("Verdict", f"{v}: computed by code from the check results")
        results = [{"check": r.check, "status": r.status, "summary": r.summary, "details": r.details,
                    "artifacts": r.artifacts} for r in res]
    STATE["gate"] = {"verdict": v, "results": results, "explanation": expl, "fix_plan": plan, "rules_applied": notes,
                     "dataset": ds, "manifest": p.get("manifest") or "", "out": str(out)}
    LIVE.act("data", f"verdict: {v}" + ("" if v == "PASSED" else " — training is blocked until the FAILs are fixed"),
             "good" if v == "PASSED" else "bad", "stage")


def plan_job(p: dict):
    past = [status.status(d) for d in sorted((HOME / "runs").iterdir()) if d.is_dir()][-5:]
    sp, notes = planner.plan(p["dataset"], p["base"], p["goal"], p.get("name") or "run", p.get("device") or "0",
                             p.get("llm") or "qwen2.5:3b", past, on_event=LIVE.on_event("train"))
    sp.gate_manifest = p.get("manifest") or None
    sp.train_python = p.get("train_python") or None
    STATE["plan"] = {"spec": asdict(sp), "notes": notes}
    LIVE.act("train", "plan ready — review the settings, then approve", "good", "stage")


def launch_job(d: dict, override: bool):
    sp = specmod.from_dict(d)
    res = launch.launch(sp, HOME / "runs", override_window=override,
                        on_step=lambda st, tx: (LIVE.stage(st, tx), LIVE.act("train", f"{st}: {tx}", "info", "stage")))
    for n in res.get("notes", []):
        LIVE.act("train", n, "info", "result")
    if res["ok"]:
        LIVE.act("train", f"{sp.name} started (pid {res['pid']}) — follow it in Runs", "good", "stage")
    else:
        LIVE.act("train", f"NOT launched: {res['reason']}", "bad", "warn")
        for f in res.get("failed_checks", []):
            LIVE.act("data", f"FAIL {f['check']}: {f['summary'][:150]}", "bad", "result")


def review_job(run: str, llm: str):
    d = HOME / "runs" / run
    st, sp = status.status(d), specmod.load(d / "run.yaml")
    if st.get("alive") or st.get("error_tail") or not st.get("epochs_done"):
        LIVE.act("train", "nothing to review yet (still running, failed or no epochs)", "warn", "warn")
        return
    LIVE.stage("Read run", f"{st['epochs_done']} epochs, best at epoch {st.get('best_epoch')}")
    LIVE.stage("Model reviews", f"{llm or 'qwen2.5:3b'} is reading the metrics")
    rv = planner.review(st, sp, llm or "qwen2.5:3b", on_event=LIVE.on_event("train"))
    STATE["review"] = {"run": run, "summary": rv["summary"], "notes": rv["notes"],
                       "next_spec": asdict(rv["next_spec"]) if rv["next_spec"] else None}
    LIVE.stage("Ready for you", "load the proposal into Plan if you want it")
    LIVE.act("train", "review ready — you can load its proposal into Plan", "good", "stage")


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
    LIVE.act("you", f"stopped {name} (pid {pid})", "warn", "you")
    return "stopped"


def rules_view() -> dict:
    try:
        r = rules_mod.load()
        ok, why = rules_mod.in_window(r)
        return {"name": r.get("name") or ("your rules" if r else ""), "path": r.get("_path", ""),
                "lines": rules_mod.summary(r), "window": {"ok": ok, "text": why},
                "limits": specmod.rule_limits()}
    except Exception as exc:  # noqa: BLE001 — a broken rules file must be visible
        return {"name": "rules file has an error", "path": str(rules_mod.path()), "lines": [str(exc)],
                "window": {"ok": True, "text": ""}, "limits": specmod.LIMITS}


# ── page ─────────────────────────────────────────────────────────────────────
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Training Console</title><style>""" + KIT_CSS + r"""
.logo{background:#2a57cf} .chip{background:var(--soft);border-radius:999px;padding:3px 11px;font-size:12.5px;color:var(--muted);max-width:46ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.steps{display:flex;gap:4px;margin-left:auto;background:var(--soft);padding:3px;border-radius:999px}
.steps button{border:0;background:transparent;color:var(--muted);border-radius:999px;padding:6px 14px;cursor:pointer;font:inherit;font-weight:550}
.steps button.on{background:var(--card);color:var(--fg);box-shadow:var(--shadow)} .steps .n{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:var(--line);font-size:11px;margin-right:6px}
.steps button.on .n{background:var(--acc);color:var(--accfg)} .steps button.ok .n{background:var(--ok);color:#fff}
.runcard{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:12px 0;border-bottom:1px solid var(--line)} .runcard:last-child{border:0}
svg.spark{width:150px;height:36px;color:var(--acc)} .empty{color:var(--muted);padding:12px 0}
</style></head><body>
<header class="top"><div class="brand"><span class="logo">↗</span>Training Console</div><span class="chip" id="dsChip">no dataset yet</span>
<nav class="steps"><button data-s="data" class="on"><span class="n">1</span>Data</button><button data-s="plan"><span class="n">2</span>Plan</button><button data-s="runs"><span class="n">3</span>Runs</button></nav></header>
<div class="wrap"><main>
<div class="card now" id="now" hidden></div>

<section id="s-data"><div class="card"><h2>1 · Check the dataset</h2><p class="m">The data agent runs dataset-gate: leakage, broken labels, missing classes,
suspicious empty tiles. Training only starts on a dataset that passes, and it is checked again at launch.</p>
<label>Dataset folder (data.yaml, images/, labels/)</label><div class="drop" id="dz-dataset"><input id="dataset" placeholder="C:\path\to\dataset"></div>
<div class="g2"><div><label>Manifest CSV <span class="hint">optional: frame, capture_time, lat, lon, task</span></label><div class="drop" id="dz-manifest"><input id="manifest"></div></div>
<div><label>Detector .pt <span class="hint">optional: missing-label and class-swap checks</span></label><div class="drop" id="dz-model"><input id="model"></div></div></div>
<button class="btn" onclick="gate('agent')">Ask the data agent</button><button class="btn sec" onclick="gate('quick')">Quick check (no LLM)</button></div>
<div class="card" id="gateOut" hidden></div></section>

<section id="s-plan" hidden><div class="card"><h2>2 · Plan a training run</h2><p class="m">The training agent proposes settings with your preferences in mind.
Code clamps every value to its allowed range (your rules can narrow them); you edit and approve.</p>
<label>Base weights (.pt)</label><div class="drop" id="dz-base"><input id="base"></div>
<div class="g2"><div><label>Run name</label><input id="name" value="run"></div><div><label>Device</label><select id="device"><option value="0">GPU 0</option><option value="cpu">CPU</option></select></div></div>
<label>Goal (plain English)</label><input id="goal" value="High recall on small aerial animals; this GPU is shared, keep memory low">
<details style="margin-top:10px"><summary>advanced</summary><div class="g2"><div><label>Planner LLM (Ollama)</label><input id="llm" value="qwen2.5:3b"></div>
<div><label>Training Python (ultralytics + torch)</label><input id="train_python"></div></div></details>
<button class="btn" id="planBtn" onclick="plan()">Ask the training agent for a plan</button><span class="hint" id="planGate"></span></div>
<div class="card" id="planOut" hidden></div></section>

<section id="s-runs" hidden><div class="card"><h2>3 · Runs</h2><p class="m">Live progress; refreshes on its own. Review a finished run to get the agent's next proposal.</p><div id="runsOut"></div></div>
<div class="card" id="reviewOut" hidden></div></section>
</main>
<aside class="side"><div class="card"><h2>Activity</h2><div class="feedtools"><button class="on" onclick="Kit.filterFeed($('feed'),'all',this)">all</button>
<button onclick="Kit.filterFeed($('feed'),'data',this)">data agent</button><button onclick="Kit.filterFeed($('feed'),'train',this)">training agent</button><button onclick="Kit.filterFeed($('feed'),'you',this)">you</button></div>
<div class="feed" id="feed"><div class="m">Everything both agents do shows up here.</div></div></div>
<div class="card" id="rulesCard" hidden><h2>Your rules</h2><p class="hint" id="rulesPath"></p><div id="rulesWin"></div><div id="rulesList"></div></div></aside></div>

<dialog id="confirm"><h2 id="cTitle">Confirm</h2><div id="cBody" class="m"></div>
<div style="text-align:right"><button class="btn sec" onclick="confirmDlg.close()">Cancel</button><button class="btn" id="cOk">Confirm</button></div></dialog>
<script>""" + KIT_JS + r"""
const $ = id => document.getElementById(id), esc = Kit.esc, NAMES = {data:"Data agent", train:"Training agent", you:"You"};
const confirmDlg = $("confirm"); let S = {}, step = "data", R = {limits:{}};
const FORM = ["dataset","manifest","model","base","name","goal","device","llm","train_python"];
const FIELDS = [["imgsz","Image size",32],["epochs","Epochs",1],["patience","Patience",1],["batch","Batch",1],["lr0","Learning rate",0.0001],
 ["fliplr","Flip left-right",0.05],["flipud","Flip up-down",0.05],["degrees","Rotation °",1],["mosaic","Mosaic",0.05],["scale","Scale jitter",0.05],["min_free_vram_gb","Min free GPU GB",0.5]];
Kit.onToast(t => { const d=document.createElement("div"); d.className="ev warn bad"; d.innerHTML=`<span class="ic">!</span><div class="tx">${esc(t)}</div>`; $("feed").appendChild(d); $("feed").scrollTop=1e9; });
function save(){ try{ const f={}; for(const k of FORM) f[k]=$(k).value; localStorage.setItem("tp-form",JSON.stringify(f)); }catch(e){} }
try{ const f=JSON.parse(localStorage.getItem("tp-form")||"{}"); for(const k in f) if($(k)&&f[k]) $(k).value=f[k]; }catch(e){}
document.querySelectorAll("input,select").forEach(e=>e.addEventListener("change",()=>{save();chip();}));
Kit.dropField($("dz-dataset"),"dataset"); Kit.dropField($("dz-manifest"),"csv"); Kit.dropField($("dz-model"),"pt"); Kit.dropField($("dz-base"),"pt");
function chip(){ $("dsChip").textContent = $("dataset").value ? "dataset: " + $("dataset").value.split(/[\\/]/).filter(Boolean).pop() : "no dataset yet"; $("dsChip").title=$("dataset").value; }
function go(s){ step=s; for(const x of ["data","plan","runs"]) $("s-"+x).hidden = x!==s; document.querySelectorAll(".steps button").forEach(b=>b.classList.toggle("on",b.dataset.s===s));
  try{ history.replaceState(null,"","#"+s); }catch(e){} if(s==="runs") loadRuns(); render(); }
document.querySelectorAll(".steps button").forEach(b=>b.onclick=()=>go(b.dataset.s));
function ask(title, html){ return new Promise(res=>{ $("cTitle").textContent=title; $("cBody").innerHTML=html; $("cOk").onclick=()=>{confirmDlg.close();res(true)}; confirmDlg.onclose=()=>res(false); confirmDlg.showModal(); }); }
async function gate(mode){ save(); chip(); await Kit.api("/api/gate",{mode,dataset:$("dataset").value.trim(),manifest:$("manifest").value.trim(),model:$("model").value.trim(),llm:$("llm").value.trim()}); poll(); }
async function plan(){ save(); await Kit.api("/api/plan",{dataset:$("dataset").value.trim(),manifest:$("manifest").value.trim(),base:$("base").value.trim(),goal:$("goal").value,name:$("name").value,device:$("device").value,llm:$("llm").value,train_python:$("train_python").value.trim()}); poll(); }
function gateHtml(g){
  const rows = g.results.map(r=>`<div class="row"><span class="pill ${r.status}">${r.status}</span><div><b>${esc(r.check)}</b> — ${esc(r.summary)}
   ${r.details&&Object.keys(r.details).length?`<details><summary>evidence</summary><pre>${esc(JSON.stringify(r.details,null,1)).slice(0,4000)}</pre></details>`:""}
   ${(r.artifacts||[]).filter(a=>/\.(jpg|png)$/i.test(a)).map(a=>`<img class="sheet" loading="lazy" src="/file?path=${encodeURIComponent(a)}" alt="${esc(r.check)} sheet">`).join("")}</div></div>`).join("");
  const next = g.verdict==="PASSED" ? `<button class="btn" onclick="go('plan')">Continue to Plan →</button>` : `<div class="note">Training is blocked until the FAIL items are fixed. Fix the data, then check again.</div>`;
  return `<div class="verdict ${g.verdict}">${g.verdict}</div><div class="m">${esc(g.dataset)}</div>${next}
   ${(g.rules_applied||[]).map(n=>`<div class="rule">Your rule applied: ${esc(n)}</div>`).join("")}
   ${g.explanation?`<h3>Summary</h3><pre>${esc(g.explanation)}</pre>`:""}
   ${g.fix_plan?`<details><summary>fix plan (proposed, not applied)</summary><pre>${esc(g.fix_plan)}</pre></details>`:""}<h3>Checks</h3>${rows}`; }
function planHtml(p){ const s=p.spec, L=R.limits||{};
  const inputs = FIELDS.map(([k,l,st])=>{ const [lo,hi]=L[k]||["",""]; return `<div><label>${l} <span class="hint">${lo}–${hi}</span></label><input type="number" data-k="${k}" min="${lo}" max="${hi}" step="${st}" value="${s[k]??""}"></div>`; }).join("");
  return `<h2>Proposed run: ${esc(s.name)}</h2>
   ${s.notes?`<div class="quote"><span class="hint">the agent's reasoning:</span> ${esc(s.notes)}</div>`:""}
   ${p.notes.map(n=>`<div class="note">code corrected: ${esc(n)}</div>`).join("")}
   <div class="g3">${inputs}<div><label>Optimizer</label><select data-k="optimizer">${["AdamW","SGD","Adam","auto"].map(o=>`<option ${o===s.optimizer?"selected":""}>${o}</option>`).join("")}</select></div>
   <div><label>Time cap, hours <span class="hint">optional</span></label><input type="number" data-k="time_hours" min="0.05" max="24" step="0.25" value="${s.time_hours??""}"></div></div>
   <button class="btn" onclick="approve()">Approve &amp; launch…</button><span class="hint">Your rules and the data gate are checked again at launch.</span>`; }
async function approve(){ const sp = Object.assign({}, S.plan.spec);
  document.querySelectorAll("#planOut [data-k]").forEach(e=>{ const v=e.value; sp[e.dataset.k] = e.tagName==="SELECT" ? v : (v===""?null:Number(v)); });
  const gpu = sp.device!=="cpu", out = gpu && R.window && !R.window.ok;
  const ok = await ask("Launch this run?", `<p><b>${esc(sp.name)}</b> on <b>${gpu?"GPU "+esc(sp.device):"CPU"}</b></p>
    <p>${sp.epochs} epochs max (stops early after ${sp.patience} without improvement)${sp.time_hours?`, at most ${sp.time_hours} h`:""}; image size ${sp.imgsz}, batch ${sp.batch}, learning rate ${sp.lr0}.</p>
    <p>Dataset: ${esc(sp.dataset)}<br>dataset-gate runs first; a BLOCKED dataset will not train.${gpu?` It waits for at least ${sp.min_free_vram_gb} GB free GPU memory.`:""}</p>
    ${out?`<div class="note"><b>Outside your training window</b> (${esc(R.window.text)}).<br><label style="display:flex;gap:8px;align-items:center;color:var(--fg)"><input type="checkbox" id="ovr" style="width:auto"> The GPU is free right now (nobody else is using it): launch anyway</label></div>`:""}`);
  if(!ok) return; const override = out && $("ovr") && $("ovr").checked;
  if(out && !override) return Kit.api("/api/launch",{spec:sp,override:false}).then(poll);
  await Kit.api("/api/launch",{spec:sp,override}); setTimeout(()=>go("runs"),900); poll(); }
function spark(v){ if(!v.length) return ""; const mx=Math.max(...v,0.01), w=150,h=36, pts=v.map((y,i)=>`${(i/(Math.max(v.length-1,1)))*w},${h-3-(y/mx)*(h-6)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="mAP50 per epoch"><polyline fill="none" stroke="currentColor" stroke-width="2" points="${pts}"/></svg>`; }
async function loadRuns(){ const j = await Kit.api("/api/runs"); if(!j) return;
  $("runsOut").innerHTML = j.runs.length ? j.runs.map(r=>`<div class="runcard"><div><span class="pill ${r.state.split(" ")[0]}">${r.state}</span><b>${esc(r.name)}</b>
     <div class="m">epoch ${r.epochs_done}${r.epochs?"/"+r.epochs:""} · ${r.device==="cpu"?"CPU":"GPU"} · gate ${esc(r.gate||"–")}${r.best&&r.best["mAP50(B)"]!==undefined?` · best mAP50 ${r.best["mAP50(B)"]} · recall ${r.best["recall(B)"]}`:""}</div>
     ${r.error?`<details><summary>error</summary><pre>${esc(r.error)}</pre></details>`:""}${r.weights?`<div class="hint">weights: ${esc(r.weights)}</div>`:""}</div>
     <div style="text-align:right">${spark(r.map50)}<div>${r.state==="running"?`<button class="btn danger small" onclick="stopRun('${esc(r.name)}')">Stop…</button>`:""}
     ${r.state==="finished"?`<button class="btn sec small" onclick="review('${esc(r.name)}')">Review</button>`:""}</div></div></div>`).join("") : `<div class="empty">No runs yet — plan one in step 2.</div>`; }
async function stopRun(n){ if(await ask("Stop this run?", `<p><b>${esc(n)}</b> will be killed. Its last saved weights stay on disk.</p>`)){ await Kit.api("/api/stop",{run:n}); loadRuns(); } }
async function review(n){ await Kit.api("/api/review",{run:n,llm:$("llm").value}); poll(); }
function loadProposal(){ if(!S.review||!S.review.next_spec) return; S.plan = {spec:S.review.next_spec, notes:S.review.notes||[]}; $("planOut").hidden=false; $("planOut").innerHTML=planHtml(S.plan); $("planOut").dataset.n="review"; go("plan"); }
function render(){
  document.querySelectorAll("main .btn").forEach(b=>{ if(!b.closest(".now")&&!b.classList.contains("danger")) b.disabled = !!S.busy; });
  if(S.gate){ const k=JSON.stringify([S.gate.verdict,S.gate.dataset,S.gate.results.length,S.gate.explanation.length]); if($("gateOut").dataset.v!==k){ $("gateOut").hidden=false; $("gateOut").innerHTML=gateHtml(S.gate); $("gateOut").dataset.v=k; } }
  const passed = S.gate && S.gate.verdict==="PASSED" && S.gate.dataset===$("dataset").value.trim();
  document.querySelector('.steps [data-s="data"]').classList.toggle("ok", !!passed);
  $("planGate").textContent = passed ? "  ✓ dataset passed the gate" : "  Check the dataset in step 1 first (it is re-checked at launch anyway).";
  if(S.plan){ const k=S.plan.spec.name+S.plan.notes.length+(S.plan.spec.notes||"").length+JSON.stringify(R.limits); if($("planOut").dataset.n!==k && $("planOut").dataset.n!=="review"){ $("planOut").hidden=false; $("planOut").innerHTML=planHtml(S.plan); $("planOut").dataset.n=k; } }
  if(S.review){ $("reviewOut").hidden=false; $("reviewOut").innerHTML=`<h2>Review of ${esc(S.review.run)}</h2><pre>${esc(S.review.summary)}</pre>${S.review.next_spec?`<button class="btn" onclick="loadProposal()">Load the proposed next run into Plan →</button>`:"<p class='m'>No next run proposed.</p>"}`; }
}
let timer=null;
async function poll(){ clearTimeout(timer); const j = await Kit.api("/api/state");
  if(j){ S=j; R=j.rules; Kit.renderNow($("now"), j.now, NAMES, j.cancelling); Kit.renderFeed($("feed"), j.activity, NAMES);
    $("rulesCard").hidden = !R.lines.length; $("rulesPath").textContent = R.path||"";
    $("rulesWin").innerHTML = R.window.text ? `<div class="${R.window.ok?"rule":"note"}">GPU: ${esc(R.window.text)}${R.window.ok?" — inside the window now":""}</div>` : "";
    $("rulesList").innerHTML = R.lines.map(l=>`<div class="rule">${esc(l)}</div>`).join(""); render(); }
  if(step==="runs") loadRuns(); timer=setTimeout(poll, j&&j.busy?600:2500); }
const h0 = location.hash.slice(1); chip(); go(["data","plan","runs"].includes(h0) ? h0 : "data"); poll();
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
        g = STATE.get("gate")
        if g:
            roots.append(Path(g["out"]).resolve())
        return any(r == p or r in p.parents for r in roots) and p.is_file()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._send(200, {**LIVE.snapshot(), **STATE, "rules": rules_view()})
        if u.path == "/api/runs":
            return self._send(200, {"runs": run_rows()})
        if u.path == "/file":
            p = Path(unquote(parse_qs(u.query).get("path", [""])[0]))
            if not self._allowed(p):
                return self._send(404, {"error": "not found"})
            return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self._send(404, {"error": "not found"})
        q = parse_qs(u.query)
        try:
            save_upload(HOME, q.get("id", [""])[0], q.get("rel", [""])[0], self.rfile,
                        int(self.headers.get("Content-Length", 0)))
        except (ValueError, OSError) as exc:
            return self._send(400, {"error": str(exc)})
        self._send(200, {"ok": True})

    def do_POST(self):
        path, b = urlparse(self.path).path, self._body()
        if b is None:
            return self._send(400, {"error": "malformed request"})
        if path == "/api/cancel":
            LIVE.cancel = bool(LIVE.busy)
            if LIVE.busy:
                LIVE.act("you", "asked to stop — the agent stops after its current step", "warn", "you")
            return self._send(200, {"ok": True})
        if path == "/api/upload/done":
            try:
                r = finish_upload(HOME, str(b.get("id", "")), str(b.get("want", "dataset")))
            except (ValueError, OSError, KeyError) as exc:
                return self._send(400, {"error": str(exc)})
            LIVE.act("you", f"dropped files → {r['path']} ({r['files']} files)", "info", "you")
            return self._send(200, r)
        if path == "/api/stop":
            return self._send(200, {"result": stop_run(str(b.get("run", "")))})
        busy = {"error": "an agent is already working — wait for it or press Stop"}
        if path == "/api/gate":
            ds = str(b.get("dataset", "")).strip()
            if not ds:
                return self._send(400, {"error": "drop a dataset folder or paste its path first"})
            if not (Path(ds) / "images").is_dir():
                return self._send(400, {"error": "that folder has no images/ — is it a YOLO dataset?"})
            from dsgate.agent import STAGES
            agent = b.get("mode") == "agent"
            ok = _job("auditing the dataset" if agent else "quick check", "data",
                      STAGES if agent else ["Load", "Run checks", "Verdict"], gate_job, b, cancellable=True)
            if ok:
                LIVE.act("you", f"asked the data agent to {'audit' if agent else 'quick-check'} {Path(ds).name}",
                         "info", "you")
        elif path == "/api/plan":
            if not (Path(b.get("dataset", "")) / "data.yaml").is_file():
                return self._send(400, {"error": "pick a dataset (with data.yaml) in step 1 first"})
            if not str(b.get("base", "")).endswith(".pt"):
                return self._send(400, {"error": "base weights must be a .pt file — drop one or paste its path"})
            ok = _job("planning a run", "train", planner.PLAN_STAGES, plan_job, b)
            if ok:
                LIVE.act("you", f"asked the training agent for a plan: “{b.get('goal', '')[:80]}”", "info", "you")
        elif path == "/api/launch":
            sp, override = b.get("spec") or {}, bool(b.get("override"))
            ok = _job("launching", "train", ["Rules", "Data gate", "GPU", "Start"], launch_job, sp, override)
            if ok:
                LIVE.act("you", f"approved launch of {sp.get('name')}" + (" (outside the window, override)"
                                                                          if override else ""), "info", "you")
        elif path == "/api/review":
            ok = _job("reviewing a run", "train", ["Read run", "Model reviews", "Ready for you"], review_job,
                      b.get("run", ""), b.get("llm", ""))
        else:
            return self._send(404, {"error": "not found"})
        return self._send(200 if ok else 409, {"ok": True} if ok else busy)


def serve(port: int = 8770, home: Path | None = None):
    global HOME
    HOME = Path(home or Path.cwd()).resolve()
    (HOME / "runs").mkdir(parents=True, exist_ok=True)
    (HOME / "plans").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training Console on http://127.0.0.1:{port}  (home: {HOME})  close this window to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
