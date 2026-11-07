"""Evaluate the training agent: how good are its plans and reviews before code cleans them up?

    tpilot eval --dataset D [--dataset D2] --base W.pt --out report_dir [--repeats 2]

Plans: every goal x dataset, `repeats` times. From the model's RAW proposal (before clamping) we score
  valid             it answered with usable JSON (no fallback to defaults)
  corrections       how many values code had to change; how many of those your rules' limits caught
  preferences       batch <= 4, lr0 <= 0.005, both flips > 0 (your rules file), reasoning present
  goal-sensitive    the "quick check" goal gets fewer epochs than the "overnight" goal
  consistency       repeats of the same request give the same settings
Reviews: every finished run in runs/: did the next proposal change at most two settings, as instructed?
Writes planner_eval.json and planner_eval.html. Everything runs locally.
"""
from __future__ import annotations

import html
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from . import planner, spec as specmod, status

GOALS = {
    "quick": "Quick sanity check that training runs at all; I only have 10 minutes",
    "recall": "High recall on small aerial animals; this GPU is shared, keep memory low",
    "overnight": "Best possible model; it can train overnight for 8 hours",
    "precision": "Too many false alarms on empty tiles; improve precision without losing much recall",
    "small_data": "Very little data; avoid overfitting",
}
KEYS = ["imgsz", "epochs", "patience", "batch", "lr0", "optimizer", "fliplr", "flipud", "degrees", "mosaic", "scale"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def evaluate(datasets: list[str], base: str, out: str, repeats: int = 2, llm: str = "qwen2.5:3b",
             home: str | None = None, log=print) -> dict:
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    rows = []
    for di, ds in enumerate(datasets):
        for g, goal in GOALS.items():
            for rep in range(repeats):
                ev = {"raw": None, "reason": None}

                def on_event(e):
                    if e["kind"] == "proposal":
                        ev["raw"] = e.get("data")
                    elif e["kind"] == "reason":
                        ev["reason"] = e.get("text")
                t0 = time.time()
                sp, notes = planner.plan(ds, base, goal, f"eval_{g}", "0", llm, [], on_event=on_event)
                raw = ev["raw"] or {}
                valid = bool(ev["raw"]) and not any(n.startswith("planner fell back") or n.startswith("invalid")
                                                    for n in notes)
                b, lr, fl, fu = (_num(raw.get(k)) for k in ("batch", "lr0", "fliplr", "flipud"))
                row = {"dataset": f"dataset{di + 1}", "goal": g, "repeat": rep, "valid": valid,
                       "corrections": len(notes), "rule_limit_hits": sum("by your rules" in n for n in notes),
                       "corrected_fields": sorted({n.split(":")[0] for n in notes if ":" in n}),
                       "pref_batch_ok": b is not None and b <= 4, "pref_lr_ok": lr is not None and lr <= 0.005,
                       "pref_flips_ok": bool(fl and fu and fl > 0 and fu > 0),
                       "reasoning": bool(ev["reason"] and len(ev["reason"].split()) >= 8),
                       "raw": {k: raw.get(k) for k in KEYS}, "final": {k: getattr(sp, k) for k in KEYS},
                       "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                log(f"  {row['dataset']} {g:10} #{rep} valid={valid} corrections={len(notes)} "
                    f"raw batch={raw.get('batch')} lr0={raw.get('lr0')} epochs={raw.get('epochs')} · {row['seconds']} s")
    reviews = []
    runs = Path(home or ".") / "runs"
    for d in sorted(x for x in runs.iterdir() if x.is_dir()) if runs.exists() else []:
        st = status.status(d)
        if st.get("alive") or st.get("error_tail") or not st.get("epochs_done") or not (d / "run.yaml").exists():
            continue
        sp = specmod.load(d / "run.yaml")
        rv = planner.review(st, sp, llm)
        nxt = rv["next_spec"]
        changed = [k for k in KEYS if nxt is not None and getattr(nxt, k) != getattr(sp, k)]
        reviews.append({"run": d.name, "proposed": nxt is not None, "changed": changed,
                        "follows_max_two": nxt is not None and len(changed) <= 2,
                        "has_model_comment": "Model's comment:" in rv["summary"]})
        log(f"  review {d.name}: changed {changed}")
    summ = summarise(rows, reviews, llm)
    (out_p / "planner_eval.json").write_text(json.dumps({"summary": summ, "rows": rows, "reviews": reviews},
                                                        indent=1, default=str), encoding="utf-8")
    (out_p / "planner_eval.html").write_text(to_html(summ, rows, reviews), encoding="utf-8")
    return {"summary": summ, "rows": rows, "reviews": reviews}


def summarise(rows: list[dict], reviews: list[dict], llm: str) -> dict:
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    ep = lambda g: [_num(r["raw"]["epochs"]) for r in rows if r["goal"] == g and _num(r["raw"]["epochs"])]  # noqa: E731
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["goal"]), []).append(json.dumps(r["raw"], sort_keys=True))
    consistent = [len(set(v)) == 1 for v in groups.values() if len(v) > 1]
    fields = {}
    for r in rows:
        for f in r["corrected_fields"]:
            fields[f] = fields.get(f, 0) + 1
    return {
        "llm": llm, "plans": len(rows),
        "valid_json_pct": pct([r["valid"] for r in rows]),
        "plans_needing_any_correction_pct": pct([r["corrections"] > 0 for r in rows]),
        "corrections_per_plan": round(statistics.mean(r["corrections"] for r in rows), 2) if rows else None,
        "plans_caught_by_your_limits_pct": pct([r["rule_limit_hits"] > 0 for r in rows]),
        "most_corrected_fields": dict(sorted(fields.items(), key=lambda kv: -kv[1])[:6]),
        "pref_batch_le_4_pct": pct([r["pref_batch_ok"] for r in rows]),
        "pref_lr0_le_0.005_pct": pct([r["pref_lr_ok"] for r in rows]),
        "pref_both_flips_pct": pct([r["pref_flips_ok"] for r in rows]),
        "reasoning_given_pct": pct([r["reasoning"] for r in rows]),
        "median_epochs_quick_goal": statistics.median(ep("quick")) if ep("quick") else None,
        "median_epochs_overnight_goal": statistics.median(ep("overnight")) if ep("overnight") else None,
        "same_settings_on_repeat_pct": pct(consistent),
        "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else None,
        "reviews": len(reviews), "reviews_following_max_two_changes_pct": pct([r["follows_max_two"] for r in reviews]),
        "reviews_with_model_comment_pct": pct([r["has_model_comment"] for r in reviews]),
    }


def to_html(s: dict, rows: list[dict], reviews: list[dict]) -> str:
    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in cells) + "</tr>"
    kv = "".join(tr([k.replace("_", " "), v]) for k, v in s.items())
    body = "".join(tr([r["dataset"], r["goal"], r["repeat"], "yes" if r["valid"] else "NO", r["corrections"],
                       ", ".join(r["corrected_fields"]), r["raw"]["batch"], r["raw"]["lr0"], r["raw"]["epochs"],
                       r["raw"]["fliplr"], r["raw"]["flipud"], r["seconds"]]) for r in rows)
    rv = "".join(tr([r["run"], ", ".join(r["changed"]) or "–", "yes" if r["follows_max_two"] else "NO"]) for r in reviews)
    return f"""<!doctype html><meta charset="utf-8"><title>Planner evaluation</title>
<style>body{{font:14px/1.5 system-ui;max-width:1100px;margin:0 auto;padding:20px}} table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:5px 8px;text-align:left}} th{{background:#f3f3f0}}</style>
<h1>trainpilot planner evaluation</h1><h2>Summary</h2><table>{kv}</table>
<h2>Plans (raw model values, before clamping)</h2><table>{tr(["dataset", "goal", "#", "valid", "corrections", "fields", "batch", "lr0", "epochs", "fliplr", "flipud", "s"], "th")}{body}</table>
<h2>Reviews</h2><table>{tr(["run", "settings changed", "≤ 2 changes"], "th")}{rv}</table>"""
