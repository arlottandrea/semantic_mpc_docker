#!/usr/bin/env python3
"""Benchmark NMPC scaling over active-tree count and prediction horizon."""

import argparse, csv, json, subprocess, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/semantic_mpc/semantic_mpc/tools/plots/scripts/plot_nmpc_multi_tree_diagnostic.py"

def run_case(args, active, horizon, output):
    # Keep the scene size equal to the requested active set.  Otherwise a
    # nominal "2 active trees" case still builds and propagates beliefs for
    # every tree in the largest scene, which measures a different quantity.
    tree_count = active
    command=[sys.executable,str(SCRIPT),"--output-dir",str(output),"--checkpoint",str(args.checkpoint),
             "--tree-count",str(tree_count),"--active-targets",str(active),"--horizon",str(horizon),
             "--iterations",str(args.iterations),"--ipopt-max-iter",str(args.ipopt_max_iter),
             "--ipopt-max-cpu-time",str(args.ipopt_max_cpu_time),"--information-gain-weight","20",
             "--entropy-time-pressure-weight","20","--attraction-weight","0.2"]
    started=time.perf_counter()
    try:
        result=subprocess.run(command,cwd=str(ROOT),text=True,capture_output=True,timeout=args.case_timeout)
    except subprocess.TimeoutExpired:
        wall=time.perf_counter()-started
        return dict(active_trees=active,horizon=horizon,batch_size=active*horizon,status="timeout",
                    wall_time_s=wall,solve_mean_ms=np.nan,solve_p95_ms=np.nan,solve_max_ms=np.nan,
                    final_entropy=np.nan,entropy_reduction=np.nan,tracked_trees=np.nan,min_tree_distance=np.nan,
                    csv_path="",plot_path="")
    wall=time.perf_counter()-started
    if result.returncode:
        return dict(active_trees=active,horizon=horizon,batch_size=active*horizon,status="error",
                    wall_time_s=wall,solve_mean_ms=np.nan,solve_p95_ms=np.nan,solve_max_ms=np.nan,
                    final_entropy=np.nan,entropy_reduction=np.nan,tracked_trees=np.nan,min_tree_distance=np.nan,
                    csv_path="",plot_path="")
    rows=list(csv.DictReader((output/"nmpc_multi_tree_diagnostic.csv").open(newline="",encoding="utf-8")))
    solve=np.asarray([float(r["solve_time_s"]) for r in rows[1:]],dtype=float);last=rows[-1]
    tracked=sum(max(float(last[f"tree{i}_belief_ripe"]),float(last[f"tree{i}_belief_raw"]))>=.9975245006578829 for i in range(tree_count))
    return dict(active_trees=active,horizon=horizon,batch_size=active*horizon,status="ok",wall_time_s=wall,
                solve_mean_ms=1e3*float(solve.mean()),solve_p95_ms=1e3*float(np.percentile(solve,95)),solve_max_ms=1e3*float(solve.max()),
                final_entropy=float(last["total_entropy_bits"]),entropy_reduction=tree_count-float(last["total_entropy_bits"]),
                tracked_trees=int(tracked),min_tree_distance=min(float(r["min_tree_distance"]) for r in rows),
                csv_path=str(output/"nmpc_multi_tree_diagnostic.csv"),plot_path=str(output/"nmpc_multi_tree_diagnostic.png"))

def reports(root,records,active_values,horizons):
    with (root/"ablation_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    metrics=[("solve_mean_ms","Mean solve [ms]"),("solve_p95_ms","P95 solve [ms]"),("entropy_reduction","Entropy reduction [bits]"),("tracked_trees","Tracked trees")]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    matrices={}
    for ax,(key,title) in zip(axes.flat,metrics):
        matrix=np.asarray([[next(r[key] for r in records if r["active_trees"]==a and r["horizon"]==h) for h in horizons] for a in active_values],dtype=float);matrices[key]=matrix
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap="viridis");ax.set_xticks(range(len(horizons)),horizons);ax.set_yticks(range(len(active_values)),active_values);ax.set_xlabel("horizon");ax.set_ylabel("active trees");ax.set_title(title);fig.colorbar(image,ax=ax)
        for i in range(len(active_values)):
            for j in range(len(horizons)):
                value=matrix[i,j];ax.text(j,i,"{:.1f}".format(value) if np.isfinite(value) else "TIMEOUT",ha="center",va="center",color="white",fontsize=8)
    fig.savefig(root/"active_tree_horizon_ablation.png",dpi=180);plt.close(fig)
    lines=["# Active-tree × horizon NMPC ablation","","| Active trees | Horizon | Batch | Status | Mean ms | P95 ms | Entropy reduction | Tracked | Min distance |","|---:|---:|---:|:---|---:|---:|---:|---:|---:|"]
    for r in records:lines.append("| {active_trees} | {horizon} | {batch_size} | {status} | {solve_mean_ms:.2f} | {solve_p95_ms:.2f} | {entropy_reduction:.3f} | {tracked_trees} | {min_tree_distance:.3f} |".format(**r))
    lines.extend(["", "## Benchmark assumptions and limits", "",
                  "- Each scene contains exactly the requested number of active trees; inactive trees are not retained in the symbolic graph.",
                  "- IPOPT is capped by `ipopt_max_iter` and `ipopt_max_cpu_time` for each solve.",
                  "- A whole diagnostic case is marked `timeout` after `case_timeout`; timeout cells are not numeric measurements.",
                  "- Information-gain and entropy-pressure weights are fixed to 20, while attraction is fixed to 0.2.",
                  "- Short runs can show zero entropy reduction when the vehicle does not reach an informative pose."])
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default=str(ROOT/"outputs/active_tree_horizon_ablation"));p.add_argument("--checkpoint",default=str(ROOT/"models/nmpc/fog/class_conditioned/5m/best_model_epoch_37.pth"));p.add_argument("--active-trees",nargs="+",type=int,default=[2,5,25,50,100]);p.add_argument("--horizons",nargs="+",type=int,default=[2,5,10,25]);p.add_argument("--iterations",type=int,default=5);p.add_argument("--ipopt-max-iter",type=int,default=8);p.add_argument("--ipopt-max-cpu-time",type=float,default=.12);p.add_argument("--case-timeout",type=float,default=60.0);args=p.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True);records=[]
    for active in args.active_trees:
        for horizon in args.horizons:
            out=root/f"active_{active}_horizon_{horizon}";out.mkdir(parents=True,exist_ok=True);record=run_case(args,active,horizon,out);records.append(record);print(json.dumps(record,indent=2))
    reports(root,records,args.active_trees,args.horizons)
if __name__=="__main__":main()
