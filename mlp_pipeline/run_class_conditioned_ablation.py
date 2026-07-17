#!/usr/bin/env python3
"""End-to-end architecture ablation for the class-conditioned surrogate."""

import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"))
from semantic_mpc_package.casadi_mlp_sensor import load_trained_weights, trained_mlp_sensor

VARIANTS = {
    "simple_small": dict(architecture="simple", hidden_size=16, hidden_layers=2, yaw_harmonics=2),
    "simple_large": dict(architecture="simple", hidden_size=128, hidden_layers=4, yaw_harmonics=2),
    "resnet": dict(architecture="resnet", hidden_size=32, hidden_layers=2, yaw_harmonics=2),
    "neural_ode": dict(architecture="neural_ode", hidden_size=32, hidden_layers=2, yaw_harmonics=2, ode_steps=3, ode_dt=0.25),
    "enhanced": dict(architecture="enhanced", hidden_size=48, hidden_layers=3, yaw_harmonics=3),
}

def run(command):
    env=dict(os.environ);env["KMP_DUPLICATE_LIB_OK"]="TRUE";env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
    started=time.perf_counter(); result=subprocess.run(command,cwd=str(ROOT),text=True,capture_output=True,env=env)
    if result.returncode: raise RuntimeError(result.stdout+"\n"+result.stderr)
    return result.stdout,time.perf_counter()-started

def config_for(directory, variant, epochs):
    training=dict(model_type="class_conditioned",input_csvs={"raw":r"C:\ros1\datasets\fog\5m\raw\TreeDatasetCNN.csv","ripe":r"C:\ros1\datasets\fog\5m\ripe\TreeDatasetCNN.csv"},
      output_dir=str(directory),replace_existing_checkpoints=True,input_dim=4,output_dim=2,loss_function="headwise_bce",
      include_alignment_features=True,output_temperature=.3,yaw_threshold_deg=30.,yaw_gate_slope=70.,threshold=5.,gate_slope=14.,
      validation_split=.35,batch_size=64,epochs=epochs,learning_rate=.0001,min_detections_for_visibility=6,minimum_detection_score=.62,
      evidence_count_midpoint=4.75,evidence_count_steepness=2.05,visibility_loss_weight=1.,semantic_loss_weight=1.,augment_fraction=0.,augment_distance_margin=5.,num_workers=0)
    training.update(variant); return {"seed":42,"device":"cpu","training":training}

def checkpoint(directory):
    paths=list(directory.glob("best_model_epoch_*.pth")); assert len(paths)==1,paths; return paths[0]

def casadi_benchmark(model_path, cache, repeats=500):
    started=time.perf_counter(); sensor=trained_mlp_sensor(16,model_path,cache,yaw_harmonics=2); build=1e3*(time.perf_counter()-started)
    rng=np.random.default_rng(4); x=rng.normal(size=(16,3));x[:,:2]*=2;x[:,2]*=.2;sensor(ca.DM(x));samples=[]
    for _ in range(repeats):
        tick=time.perf_counter();sensor(ca.DM(x));samples.append(1e6*(time.perf_counter()-tick))
    return build,float(np.mean(samples)),float(np.percentile(samples,95))

def diagnostic_outputs(name, model_path, root, trees, iterations, horizon):
    one=root/name/"one_tree"; multi=root/name/"multi_tree"
    one_script=ROOT/"src/semantic_mpc/semantic_mpc/tools/plots/scripts/plot_nmpc_one_tree_diagnostic.py"
    multi_script=ROOT/"src/semantic_mpc/semantic_mpc/tools/plots/scripts/plot_nmpc_multi_tree_diagnostic.py"
    run([sys.executable,str(one_script),"--output-dir",str(one),"--checkpoint",str(model_path),"--best-view-grid-size","31","--best-view-yaw-steps","19","--heatmap-grid-size","51","--interactive-heatmap-grid-size","31","--interactive-yaw-steps","19"])
    _,wall=run([sys.executable,str(multi_script),"--output-dir",str(multi),"--checkpoint",str(model_path),"--tree-count",str(trees),"--active-targets",str(trees),"--iterations",str(iterations),"--horizon",str(horizon),"--information-gain-weight","20","--entropy-time-pressure-weight","20","--attraction-weight","0.2"])
    rows=list(csv.DictReader((multi/"nmpc_multi_tree_diagnostic.csv").open(newline="",encoding="utf-8")));last=rows[-1]
    solve=np.asarray([float(r["solve_time_s"]) for r in rows[1:]]); tracked=sum(max(float(last[f"tree{i}_belief_ripe"]),float(last[f"tree{i}_belief_raw"]))>=.9975245006578829 for i in range(trees))
    return dict(final_entropy=float(last["total_entropy_bits"]),entropy_reduction=trees-float(last["total_entropy_bits"]),tracked_trees=int(tracked),
      min_tree_distance=min(float(r["min_tree_distance"]) for r in rows),optimizer_mean_ms=1e3*float(solve.mean()),optimizer_p95_ms=1e3*float(np.percentile(solve,95)),field_wall_s=wall,
      heatmap_interactive=str(one/"nmpc_one_tree_mlp_heatmap_interactive.html"),heatmap_plot=str(one/"nmpc_one_tree_mlp_heatmap.png"),one_tree_plot=str(one/"nmpc_one_tree_diagnostic.png"),multi_tree_plot=str(multi/"nmpc_multi_tree_diagnostic.png"))

def dataset_fit(name, cfg, model, root):
    output=root/name/"dataset_model_fit.png"; stats=root/name/"dataset_model_fit.json"
    command=[sys.executable,"-m","mlp_pipeline.visualize","--config",str(cfg),"--checkpoint",str(model),"--raw-csv",r"C:\ros1\datasets\fog\5m\raw\TreeDatasetCNN.csv","--ripe-csv",r"C:\ros1\datasets\fog\5m\ripe\TreeDatasetCNN.csv","--output",str(output),"--stats-output",str(stats)]
    env=dict(os.environ);env["KMP_DUPLICATE_LIB_OK"]="TRUE";env["OMP_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1"
    try:
        result=subprocess.run(command,cwd=str(ROOT),env=env,timeout=20,
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if result.returncode and not output.is_file(): raise RuntimeError("dataset/model visualization failed")
    except subprocess.TimeoutExpired:
        # Some Windows OpenMP combinations hang during interpreter teardown
        # after matplotlib has already flushed valid artifacts.
        if not output.is_file() or not stats.is_file(): raise
    return str(output)

def reports(root, records):
    with (root/"ablation_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    names=[r["variant"] for r in records];fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for ax,values,title in [(axes[0,0],[r["validation_loss"] for r in records],"Validation loss"),(axes[0,1],[r["optimizer_mean_ms"] for r in records],"NMPC mean solve [ms]"),(axes[1,0],[r["entropy_reduction"] for r in records],"Entropy reduction [bits]"),(axes[1,1],[r["tracked_trees"] for r in records],"Tracked trees")]:
        ax.bar(names,values);ax.set_title(title);ax.tick_params(axis="x",rotation=25);ax.grid(axis="y",alpha=.25)
    fig.savefig(root/"ablation_summary.png",dpi=180);plt.close(fig)
    lines=["# Class-conditioned MLP architecture ablation","","| Variant | Params | Val loss | CasADi us | NMPC ms | Entropy reduction | Tracked |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:lines.append("| {variant} | {parameters} | {validation_loss:.6f} | {casadi_mean_us:.1f} | {optimizer_mean_ms:.1f} | {entropy_reduction:.3f} | {tracked_trees} |".format(**r))
    lines.extend(["", "Generate the row-wise dataset/model comparison with:", "",
                  "`python -m mlp_pipeline.plot_class_conditioned_ablation_fit --ablation-dir {}`".format(root),
                  "", "## Fixed assumptions and implementation constraints", "",
                  "- All variants use seed 42, CPU training, the same raw/ripe CSV files, split (35%), batch size (64), and learning rate (1e-4).",
                  "- The reliability target uses a 30 deg yaw gate, temperature 0.3, minimum visibility count 6, minimum detection score 0.62, and evidence-count calibration midpoint 4.75 / steepness 2.05.",
                  "- The NMPC comparison fixes information-gain and entropy-pressure weights to 20 and attraction to 0.2.",
                  "- OpenMP and MKL are forced to one thread for reproducible timing; `KMP_DUPLICATE_LIB_OK` is enabled as a Windows runtime workaround.",
                  "- Dataset-fit subprocesses have a 20 s teardown timeout; an already flushed plot is accepted if the Windows OpenMP process hangs during shutdown."])
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    maps=[]
    for record in records:
        source=Path(record["heatmap_interactive"])
        maps.append({"variant":record["variant"],"html":source.read_text(encoding="utf-8")})
    (root/"heatmaps_interactive.html").write_text("""<!doctype html><meta charset='utf-8'><title>Class-conditioned heatmaps</title><style>body{margin:0;background:#111;color:#eee;font:14px system-ui}header{position:sticky;top:0;z-index:9;background:#222;padding:14px}input{width:55vw}.m{margin:14px;background:#222}.m h2{padding:10px;margin:0}.m iframe{border:0;width:100%%;height:650px;background:#fff}</style><header><b>All class-conditioned heatmaps</b> &nbsp; Global yaw <input id=s type=range min=0 max=18 value=9><b id=v>0 deg</b></header><main id=r></main><script>const maps=%s;maps.forEach(m=>{const section=document.createElement('section'),title=document.createElement('h2'),frame=document.createElement('iframe');section.className='m';title.textContent=m.variant;frame.srcdoc=m.html;frame.onload=sync;section.append(title,frame);r.append(section)});function sync(){v.textContent=(-180+20*s.value)+' deg';document.querySelectorAll('iframe').forEach(f=>{try{let q=f.contentDocument.querySelector('#yawSlider');if(q){q.value=s.value;q.dispatchEvent(new Event('input'));let controls=f.contentDocument.querySelector('.controls');if(controls)controls.style.display='none'}}catch(e){}})}s.oninput=sync;sync();</script>"""%json.dumps(maps),encoding="utf-8")
    # The aggregate is standalone; keep one user-facing interactive artifact.
    for record in records:
        Path(record["heatmap_interactive"]).unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default=str(ROOT/"outputs/class_conditioned_ablation"));p.add_argument("--epochs",type=int,default=40);p.add_argument("--tree-count",type=int,default=5);p.add_argument("--iterations",type=int,default=60);p.add_argument("--horizon",type=int,default=5);p.add_argument("--variants",nargs="*",choices=VARIANTS,default=list(VARIANTS));a=p.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);records=[]
    for name in a.variants:
        d=root/name;d.mkdir(parents=True,exist_ok=True);cfg=d/"train.yaml";cfg.write_text(yaml.safe_dump(config_for(d/"model",VARIANTS[name],a.epochs),sort_keys=False),encoding="utf-8")
        _,train_s=run([sys.executable,"-m","mlp_pipeline.train","--config",str(cfg)]);model=checkpoint(d/"model");meta=json.loads((d/"model/training_metadata.json").read_text());fit=str(d/"dataset_model_fit.png");cache=d/"weights.npz";build,mean,p95=casadi_benchmark(model,cache)
        state=load_trained_weights(model,cache);params=sum(v.size for k,v in state.items() if k.endswith("weight") or k.endswith("bias"));diag=diagnostic_outputs(name,model,root,a.tree_count,a.iterations,a.horizon)
        rec=dict(variant=name,parameters=params,training_seconds=train_s,validation_loss=meta["best_validation_loss"],casadi_build_ms=build,casadi_mean_us=mean,casadi_p95_us=p95,dataset_fit=fit,**diag);records.append(rec);print(json.dumps(rec,indent=2))
    reports(root,records)
if __name__=="__main__":main()
