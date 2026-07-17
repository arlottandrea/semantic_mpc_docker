#!/usr/bin/env python3
"""Train and benchmark dual-reliability MLP architecture ablations."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
sys.path.insert(0, str(SEMANTIC_SRC))
from semantic_mpc_package.casadi_mlp_sensor import load_trained_weights, trained_mlp_sensor


VARIANTS = {
    "simple_small": dict(architecture="simple", hidden_size=16, hidden_layers=2, yaw_harmonics=2),
    "simple_large": dict(architecture="simple", hidden_size=128, hidden_layers=4, yaw_harmonics=2),
    "resnet": dict(architecture="resnet", hidden_size=32, hidden_layers=2, yaw_harmonics=2),
    "neural_ode": dict(architecture="neural_ode", hidden_size=32, hidden_layers=2, yaw_harmonics=2,
                       ode_steps=3, ode_dt=0.25),
    "enhanced": dict(architecture="enhanced", hidden_size=48, hidden_layers=3, yaw_harmonics=3),
}


def training_config(output_dir, label, variant, epochs):
    cfg = dict(
        input_csvs={
            "raw": str(ROOT.parent / "ros1" / "datasets" / "fog" / "5m" / "raw" / "TreeDatasetCNN.csv"),
            "ripe": str(ROOT.parent / "ros1" / "datasets" / "fog" / "5m" / "ripe" / "TreeDatasetCNN.csv"),
        },
        output_dir=str(output_dir / label), replace_existing_checkpoints=True,
        input_dim=3, output_dim=1, single_output_label="accuracy_{}".format(label),
        loss_function="mse", output_temperature=0.4, include_alignment_features=True,
        yaw_threshold_deg=30.0, yaw_gate_slope=50.0, threshold=5.0, gate_slope=10.0,
        validation_split=0.35, batch_size=64, epochs=epochs, learning_rate=1e-4,
        min_detections_for_visibility=6, minimum_detection_score=0.62,
        evidence_count_midpoint=4.75, evidence_count_steepness=2.05,
        visibility_loss_weight=1.0, semantic_loss_weight=1.0,
        augment_fraction=0.0, augment_distance_margin=5.0, num_workers=0,
    )
    cfg.update(variant)
    return {"seed": 42 if label == "raw" else 43, "device": "cpu", "training": cfg}


def run_checked(command, cwd=ROOT):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("command failed:\n{}\n{}".format(result.stdout, result.stderr))
    return result.stdout, time.perf_counter() - started


def best_checkpoint(directory):
    paths = list(Path(directory).glob("best_model_epoch_*.pth"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one best checkpoint in {}".format(directory))
    return paths[0]


def benchmark_casadi(raw_checkpoint, ripe_checkpoint, cache_dir, repeats=500):
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sensor = trained_mlp_sensor(
        16, raw_checkpoint, ripe_checkpoint_path=ripe_checkpoint,
        weights_cache_path=cache_dir / "raw.npz", ripe_weights_cache_path=cache_dir / "ripe.npz",
        output_temperature=0.4,
        yaw_harmonics=int(json.loads((raw_checkpoint.parent / "training_metadata.json").read_text())["yaw_harmonics"]),
        include_alignment_features=True,
    )
    build_ms = 1e3 * (time.perf_counter() - started)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 3)); features[:, :2] *= 2.0; features[:, 2] *= 0.2
    sensor(ca.DM(features))
    samples = []
    for _ in range(repeats):
        tick = time.perf_counter(); sensor(ca.DM(features)); samples.append(1e6 * (time.perf_counter() - tick))
    return build_ms, float(np.mean(samples)), float(np.percentile(samples, 95))


def field_benchmark(name, raw_checkpoint, output_root, tree_count, iterations, horizon):
    output_dir = output_root / name / "multi_tree"
    script = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "tools" / "plots" / "scripts" / "plot_nmpc_multi_tree_diagnostic.py"
    command = [sys.executable, str(script), "--output-dir", str(output_dir),
               "--checkpoint", str(raw_checkpoint), "--tree-count", str(tree_count),
               "--active-targets", str(tree_count), "--iterations", str(iterations),
               "--horizon", str(horizon), "--information-gain-weight", "20",
               "--entropy-time-pressure-weight", "20", "--attraction-weight", "0.2"]
    _, wall_s = run_checked(command)
    rows = list(csv.DictReader((output_dir / "nmpc_multi_tree_diagnostic.csv").open(newline="", encoding="utf-8")))
    last = rows[-1]
    confidence = 0
    for idx in range(tree_count):
        confidence += max(float(last["tree{}_belief_ripe".format(idx)]),
                          float(last["tree{}_belief_raw".format(idx)])) >= 0.9975245006578829
    solve = np.asarray([float(row["solve_time_s"]) for row in rows[1:]], dtype=float)
    return dict(final_entropy=float(last["total_entropy_bits"]),
                entropy_reduction=tree_count - float(last["total_entropy_bits"]),
                tracked_trees=int(confidence), min_tree_distance=min(float(r["min_tree_distance"]) for r in rows),
                optimizer_mean_ms=1e3 * float(solve.mean()), optimizer_p95_ms=1e3 * float(np.percentile(solve, 95)),
                field_wall_s=wall_s, field_csv=str(output_dir / "nmpc_multi_tree_diagnostic.csv"),
                field_plot=str(output_dir / "nmpc_multi_tree_diagnostic.png"))


def one_tree_plots(name, raw_checkpoint, output_root):
    output_dir = output_root / name / "one_tree"
    script = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "tools" / "plots" / "scripts" / "plot_nmpc_one_tree_diagnostic.py"
    run_checked([sys.executable, str(script), "--output-dir", str(output_dir), "--checkpoint", str(raw_checkpoint),
                 "--best-view-grid-size", "31", "--best-view-yaw-steps", "19", "--heatmap-grid-size", "51",
                 "--interactive-heatmap-grid-size", "31", "--interactive-yaw-steps", "19"])
    return dict(one_tree_plot=str(output_dir / "nmpc_one_tree_diagnostic.png"),
                heatmap_plot=str(output_dir / "nmpc_one_tree_mlp_heatmap.png"),
                heatmap_interactive=str(output_dir / "nmpc_one_tree_mlp_heatmap_interactive.html"))


def write_reports(output_root, records):
    csv_path = output_root / "ablation_summary.csv"
    keys = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(records)

    names = [r["variant"] for r in records]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(names, [r["validation_loss_mean"] for r in records]); axes[0, 0].set_title("Mean validation loss")
    axes[0, 1].bar(names, [r["optimizer_mean_ms"] for r in records]); axes[0, 1].set_title("CasADi NMPC mean solve [ms]")
    axes[1, 0].bar(names, [r["entropy_reduction"] for r in records]); axes[1, 0].set_title("Field entropy reduction [bits]")
    axes[1, 1].bar(names, [r["tracked_trees"] for r in records]); axes[1, 1].set_title("Tracked trees")
    for ax in axes.flat: ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_root / "ablation_summary.png", dpi=180); plt.close(fig)

    curves = []
    curve_fig, curve_axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for record in records:
        for label_index, label in enumerate(("raw", "ripe")):
            history_path = output_root / record["variant"] / "models" / label / "training_history.csv"
            history = list(csv.DictReader(history_path.open(newline="", encoding="utf-8")))
            epochs = [int(row["epoch"]) for row in history]
            train = [float(row["train_loss"]) for row in history]
            validation = [float(row["validation_loss"]) for row in history]
            curve_axes[label_index].plot(epochs, train, alpha=0.45, label=record["variant"] + " train")
            curve_axes[label_index].plot(epochs, validation, label=record["variant"] + " val")
            curves.append((record["variant"], label, epochs, train, validation))
    for axis, label in zip(curve_axes, ("raw", "ripe")):
        axis.set_title("{} reliability training".format(label)); axis.set_xlabel("epoch"); axis.set_ylabel("loss")
        axis.set_yscale("log"); axis.grid(alpha=0.25); axis.legend(fontsize=7, ncol=2)
    curve_fig.savefig(output_root / "training_curves.png", dpi=180); plt.close(curve_fig)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        interactive = make_subplots(rows=2, cols=2, subplot_titles=("Validation loss", "NMPC solve ms", "Entropy reduction", "Tracked trees"))
        interactive.add_bar(x=names, y=[r["validation_loss_mean"] for r in records], name="val loss", row=1, col=1)
        interactive.add_bar(x=names, y=[r["optimizer_mean_ms"] for r in records], name="solve mean", row=1, col=2)
        interactive.add_bar(x=names, y=[r["entropy_reduction"] for r in records], name="entropy reduction", row=2, col=1)
        interactive.add_bar(x=names, y=[r["tracked_trees"] for r in records], name="tracked", row=2, col=2)
        interactive.update_layout(title="Dual-MLP architecture ablation", height=800)
        interactive.write_html(str(output_root / "ablation_summary_interactive.html"), include_plotlyjs=True)
        training_plot = make_subplots(rows=1, cols=2, subplot_titles=("Raw training", "Ripe training"))
        for variant, label, epochs, train, validation in curves:
            col = 1 if label == "raw" else 2
            training_plot.add_scatter(x=epochs, y=train, mode="lines", name=variant + " train", legendgroup=variant,
                                      opacity=0.45, row=1, col=col)
            training_plot.add_scatter(x=epochs, y=validation, mode="lines", name=variant + " val", legendgroup=variant,
                                      row=1, col=col)
        training_plot.update_yaxes(type="log"); training_plot.update_layout(title="Training monitor", height=600)
        training_plot.write_html(str(output_root / "training_curves_interactive.html"), include_plotlyjs=True)
    except ImportError:
        payload = json.dumps(records)
        (output_root / "ablation_summary_interactive.html").write_text("""<!doctype html><meta charset='utf-8'>
<title>Dual MLP ablation</title><style>body{font:14px system-ui;margin:28px;background:#111;color:#eee}.bar{height:28px;background:#27a7d8;margin:8px 0;padding:5px;color:#fff}.row{margin:14px 0}select{padding:6px}</style>
<h1>Dual-MLP architecture ablation</h1><label>Metric <select id='metric'></select></label><div id='chart'></div>
<script>const data=%s;const metrics=['validation_loss_mean','casadi_inference_mean_us','optimizer_mean_ms','entropy_reduction','tracked_trees','min_tree_distance'];
const sel=document.querySelector('#metric');metrics.forEach(m=>sel.add(new Option(m,m)));function draw(){let m=sel.value,max=Math.max(...data.map(d=>+d[m]));chart.innerHTML=data.map(d=>`<div class=row><b>${d.variant}</b><div class=bar style="width:${Math.max(2,80*d[m]/max)}%%" title="${d[m]}">${(+d[m]).toFixed(4)}</div></div>`).join('')}sel.onchange=draw;draw();</script>""" % payload, encoding="utf-8")
        curve_payload = json.dumps([dict(variant=v, label=l, epochs=e, train=t, validation=val)
                                    for v, l, e, t, val in curves])
        (output_root / "training_curves_interactive.html").write_text("""<!doctype html><meta charset='utf-8'>
<title>Training monitor</title><style>body{font:14px system-ui;margin:28px;background:#111;color:#eee}canvas{background:#fff}select{padding:6px;margin:8px}</style>
<h1>Interactive training monitor</h1><select id='variant'></select><select id='label'><option>raw</option><option>ripe</option></select><br><canvas id='plot' width='1000' height='550'></canvas>
<script>const data=%s,V=[...new Set(data.map(d=>d.variant))],s=document.querySelector('#variant'),c=plot.getContext('2d');V.forEach(v=>s.add(new Option(v,v)));
function draw(){let d=data.find(x=>x.variant==s.value&&x.label==label.value),all=d.train.concat(d.validation),lo=Math.min(...all),hi=Math.max(...all);c.clearRect(0,0,1000,550);c.strokeStyle='#ddd';c.strokeRect(60,20,900,480);
function line(a,color){c.strokeStyle=color;c.lineWidth=3;c.beginPath();a.forEach((y,i)=>{let x=60+900*i/(a.length-1),py=500-480*(Math.log(y)-Math.log(lo))/(Math.log(hi)-Math.log(lo));i?c.lineTo(x,py):c.moveTo(x,py)});c.stroke()}line(d.train,'#2980b9');line(d.validation,'#e74c3c');c.fillStyle='#111';c.fillText('blue=train, red=validation; log scale',65,535)}s.onchange=label.onchange=draw;draw();</script>""" % curve_payload, encoding="utf-8")

    lines = ["# Dual-MLP architecture ablation", "", "All variants use identical data, split policy, targets and NMPC scenario.", "",
             "| Variant | Params | Val loss | CasADi inference mean us | NMPC mean ms | Entropy reduction | Tracked | Min distance m |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append("| {variant} | {parameters} | {validation_loss_mean:.6f} | {casadi_inference_mean_us:.1f} | {optimizer_mean_ms:.1f} | {entropy_reduction:.3f} | {tracked_trees} | {min_tree_distance:.3f} |".format(**r))
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    heatmaps = [
        {"variant": record["variant"],
         "path": str(Path(record["heatmap_interactive"]).relative_to(output_root)).replace("\\", "/")}
        for record in records
    ]
    (output_root / "heatmaps_interactive.html").write_text("""<!doctype html><html><head><meta charset='utf-8'>
<title>Interactive MLP heatmap comparison</title><style>html,body{margin:0;font:14px system-ui;background:#111;color:#eee}
header{position:sticky;top:0;z-index:10;padding:12px 18px;background:#20242a;display:flex;align-items:center;gap:16px;box-shadow:0 2px 8px #0008}
input{width:min(620px,55vw)}.models{display:grid;grid-template-columns:1fr;gap:14px;padding:14px}.model{background:#20242a;border:1px solid #39414b;border-radius:8px;overflow:hidden}
.model h2{margin:0;padding:10px 16px;font-size:17px}.model iframe{display:block;border:0;width:100%%;height:650px;background:#fff}</style></head>
<body><header><strong>All dual-MLP heatmaps</strong><label>Global relative yaw <input id='yaw' type='range' min='0' max='18' value='9' step='1'></label><strong id='yawValue'>0.0 deg</strong></header>
<main class='models' id='models'></main><script>const maps=%s,root=document.querySelector('#models'),slider=document.querySelector('#yaw'),label=document.querySelector('#yawValue');
maps.forEach((m,i)=>{let section=document.createElement('section');section.className='model';section.innerHTML=`<h2>${m.variant}</h2><iframe data-index="${i}" src="${m.path}"></iframe>`;root.appendChild(section)});
function sync(){label.textContent=(-180+20*Number(slider.value)).toFixed(1)+' deg';document.querySelectorAll('iframe').forEach(frame=>{try{let doc=frame.contentDocument,s=doc&&doc.querySelector('#yawSlider');if(s){s.value=slider.value;s.dispatchEvent(new Event('input'));let controls=doc.querySelector('.controls');if(controls)controls.style.display='none'}}catch(e){}})}
slider.addEventListener('input',sync);document.querySelectorAll('iframe').forEach(frame=>frame.addEventListener('load',sync));sync();</script></body></html>"""
        % json.dumps(heatmaps), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "dual_mlp_ablation"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--tree-count", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args(); output_root = Path(args.output_dir); output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.variants:
        variant_dir = output_root / name; config_dir = variant_dir / "configs"; config_dir.mkdir(parents=True, exist_ok=True)
        train_seconds = 0.0
        for label in ("raw", "ripe"):
            config = training_config(variant_dir / "models", label, VARIANTS[name], args.epochs)
            path = config_dir / "train_{}.yaml".format(label); path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            _, elapsed = run_checked([sys.executable, "-m", "mlp_pipeline.train", "--config", str(path)])
            train_seconds += elapsed
        raw = best_checkpoint(variant_dir / "models" / "raw"); ripe = best_checkpoint(variant_dir / "models" / "ripe")
        raw_meta = json.loads((raw.parent / "training_metadata.json").read_text()); ripe_meta = json.loads((ripe.parent / "training_metadata.json").read_text())
        build_ms, inference_us, inference_p95_us = benchmark_casadi(raw, ripe, variant_dir / "casadi_cache")
        field = field_benchmark(name, raw, output_root, args.tree_count, args.iterations, args.horizon)
        plots = one_tree_plots(name, raw, output_root)
        parameter_state = load_trained_weights(raw, variant_dir / "casadi_cache" / "raw.npz")
        parameters = sum(value.size for key, value in parameter_state.items()
                         if key.endswith("weight") or key.endswith("bias"))
        record = dict(variant=name, parameters=2 * parameters, training_seconds=train_seconds,
                      raw_validation_loss=raw_meta["best_validation_loss"], ripe_validation_loss=ripe_meta["best_validation_loss"],
                      validation_loss_mean=0.5 * (raw_meta["best_validation_loss"] + ripe_meta["best_validation_loss"]),
                      casadi_build_ms=build_ms, casadi_inference_mean_us=inference_us, casadi_inference_p95_us=inference_p95_us,
                      **field, **plots)
        records.append(record); print(json.dumps(record, indent=2))
    write_reports(output_root, records)


if __name__ == "__main__":
    main()
