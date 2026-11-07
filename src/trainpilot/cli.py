"""tpilot — a cautious local training agent that works in tandem with dataset-gate.

  tpilot plan   --dataset D --base W.pt --goal "..." --name N   LLM proposes a spec -> plans/N.yaml (review it)
  tpilot run    plans/N.yaml [--dry-run]                         gate -> GPU check -> render -> launch detached
  tpilot status [RUN]                                            progress of one run, or all runs
  tpilot review RUN                                              LLM summary + next-run proposal (not launched)
Nothing launches without an explicit `tpilot run` on a spec file a person has read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import launch, planner, spec as specmod, status

HOME = Path.cwd()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="tpilot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=str(HOME), help="folder holding plans/ and runs/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True, help="base weights .pt")
    p.add_argument("--goal", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--manifest", help="manifest CSV for dataset-gate's time/GPS checks")
    p.add_argument("--train-python", help="python with ultralytics+torch for the run")
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("--dry-run", action="store_true", help="gate + render only, don't train")
    r.add_argument("--override-window", action="store_true", help="GPU is free: launch outside your rules' window")
    s = sub.add_parser("status")
    s.add_argument("run", nargs="?")
    v = sub.add_parser("review")
    v.add_argument("run")
    v.add_argument("--llm", default="qwen2.5:3b")
    e = sub.add_parser("eval", help="score the planner's raw proposals and the reviews")
    e.add_argument("--dataset", action="append", required=True)
    e.add_argument("--base", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--repeats", type=int, default=2)
    e.add_argument("--llm", default="qwen2.5:3b")
    u = sub.add_parser("ui", help="one console for both agents: Data -> Plan -> Runs")
    u.add_argument("--port", type=int, default=8770)
    a = ap.parse_args(argv)
    if a.cmd == "ui":
        from .ui import serve
        serve(a.port, Path(a.home))
        return 0
    home = Path(a.home)
    if a.cmd == "eval":
        from .evaluate import evaluate
        r = evaluate(a.dataset, a.base, a.out, a.repeats, a.llm, str(home))
        for k, v in r["summary"].items():
            print(f"{k:40} {v}")
        return 0
    (home / "plans").mkdir(parents=True, exist_ok=True)
    (home / "runs").mkdir(parents=True, exist_ok=True)

    if a.cmd == "plan":
        past = [status.status(d) for d in sorted((home / "runs").iterdir()) if d.is_dir()][-5:]
        sp, notes = planner.plan(a.dataset, a.base, a.goal, a.name, a.device, a.llm, past)
        sp.gate_manifest, sp.train_python = a.manifest, a.train_python
        out = home / "plans" / f"{sp.name}.yaml"
        planner.save_plan(sp, notes, out)
        print(out.read_text(encoding="utf-8"))
        print(f"plan written to {out} — read it, edit if needed, then:  tpilot run {out}")
        return 0
    if a.cmd == "run":
        sp = specmod.load(a.spec)
        res = launch.launch(sp, home / "runs", a.dry_run, a.override_window,
                            on_step=lambda st, tx: print(f"  {st}: {tx}", flush=True))
        print(json.dumps(res, indent=1, default=str))
        return 0 if res["ok"] else 1
    if a.cmd == "status":
        runs = [home / "runs" / a.run] if a.run else sorted(d for d in (home / "runs").iterdir() if d.is_dir())
        for d in runs:
            print(json.dumps(status.status(d), default=str))
        return 0
    if a.cmd == "review":
        d = home / "runs" / a.run
        st, sp = status.status(d), specmod.load(d / "run.yaml")
        if st.get("alive"):
            print("still training — review it when it has finished")
            return 1
        if st.get("error_tail") or not st.get("epochs_done"):
            print("this run failed or produced no epochs — nothing to review, and no next run is proposed.\n"
                  + (st.get("error_tail") or "(no results.csv)"))
            return 1
        rv = planner.review(st, sp, a.llm)
        print(rv["summary"])
        if rv["next_spec"] is not None:
            out = home / "plans" / f"{rv['next_spec'].name}.yaml"
            planner.save_plan(rv["next_spec"], rv["notes"], out)
            print(f"\nproposed next run (NOT launched): {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
