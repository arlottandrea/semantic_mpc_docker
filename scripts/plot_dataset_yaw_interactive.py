#!/usr/bin/env python3
"""Create a standalone interactive XY/yaw viewer for raw and ripe datasets."""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
sys.path.insert(0, str(SEMANTIC_SRC))
from semantic_mpc_package.perception_protocol import tree_observation_scores


def wrap_angle(values):
    return np.arctan2(np.sin(values), np.cos(values))


def read_points(paths, minimum_tree_detections, minimum_score):
    points = []
    for label, path in paths:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    observation = tree_observation_scores(
                        ast.literal_eval(row["Ripe_scores"]),
                        ast.literal_eval(row["Raw_scores"]),
                        minimum_score=minimum_score,
                        minimum_tree_detections=minimum_tree_detections,
                    )
                    correct_score = observation[1] if label == "raw" else observation[0]
                    points.append(
                        (float(row["x"]), float(row["y"]), float(row["yaw"]), label,
                         float(correct_score) if np.isfinite(correct_score) else None)
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    if not points:
        raise ValueError("no valid x,y,yaw rows found")
    return points


def build_payload(points):
    yaw = wrap_angle(np.asarray([point[2] for point in points], dtype=float))
    frames = np.unique(np.round(yaw, 6))
    frames.sort()
    if len(frames) > 1:
        circular = np.sort(wrap_angle(frames))
        gaps = np.diff(circular)
        tolerance = 0.51 * float(np.median(gaps[gaps > 1e-8])) if np.any(gaps > 1e-8) else np.deg2rad(1.0)
    else:
        tolerance = np.deg2rad(1.0)
    return {
        "x": [point[0] for point in points],
        "y": [point[1] for point in points],
        "yawDeg": np.rad2deg(yaw).round(4).tolist(),
        "label": [point[3] for point in points],
        "score": [point[4] for point in points],
        "framesDeg": np.rad2deg(frames).round(4).tolist(),
        "toleranceDeg": float(np.rad2deg(tolerance)),
    }


def write_html(path, payload):
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training dataset by yaw</title>
<style>
body{font-family:Arial,sans-serif;margin:18px;background:#f8fafc;color:#111827}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
input{width:min(620px,70vw)} .panels{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:14px}
.panel{background:white;padding:10px;border:1px solid #cbd5e1;border-radius:6px}.panel h3{margin:0 0 8px}
canvas{background:white;border:1px solid #cbd5e1;width:100%;height:auto}
.legend{margin-top:8px}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 5px 0 14px}
</style></head><body>
<h2>Tree-level sensor reliability targets — XY slices by yaw</h2>
<div class="controls"><label for="slider">Yaw</label><input id="slider" type="range" min="0" value="0" step="1">
<strong id="yaw"></strong><span id="count"></span></div>
<div class="panels">
<div class="panel"><h3>P(obs=raw | true=raw)</h3><canvas id="rawPlot" width="720" height="650"></canvas></div>
<div class="panel"><h3>P(obs=ripe | true=ripe)</h3><canvas id="ripePlot" width="720" height="650"></canvas></div>
</div>
<div class="legend">Color: purple=0, teal=0.5, yellow=1. Gray = insufficient tree-level evidence. Only points in the selected yaw slice are shown.</div>
<script>
const p=__PAYLOAD__, s=document.getElementById('slider');
s.max=p.framesDeg.length-1;
const xmin=Math.min(...p.x),xmax=Math.max(...p.x),ymin=Math.min(...p.y),ymax=Math.max(...p.y),pad=45;
function diff(a,b){let d=a-b;while(d>180)d-=360;while(d<-180)d+=360;return Math.abs(d)}
function color(v){if(v===null)return '#94a3b8';const t=Math.max(0,Math.min(1,v));const stops=[[68,1,84],[33,145,140],[253,231,37]],q=t*2,i=Math.min(1,Math.floor(q)),f=q-i;return `rgb(${stops[i].map((a,k)=>Math.round(a+(stops[i+1][k]-a)*f)).join(',')})`}
function drawPanel(id,label,yd){const c=document.getElementById(id),ctx=c.getContext('2d');const px=x=>pad+(x-xmin)/Math.max(1e-9,xmax-xmin)*(c.width-2*pad),py=y=>c.height-pad-(y-ymin)/Math.max(1e-9,ymax-ymin)*(c.height-2*pad);ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#94a3b8';ctx.strokeRect(pad,pad,c.width-2*pad,c.height-2*pad);let valid=0,total=0;for(let i=0;i<p.x.length;i++){if(p.label[i]!==label||diff(p.yawDeg[i],yd)>p.toleranceDeg)continue;total++;if(p.score[i]!==null)valid++;ctx.fillStyle=color(p.score[i]);ctx.beginPath();ctx.arc(px(p.x[i]),py(p.y[i]),4,0,2*Math.PI);ctx.fill();}ctx.fillStyle='#334155';ctx.font='13px Arial';ctx.fillText(`valid ${valid}/${total}`,pad,c.height-14);return [valid,total]}
function draw(){const yd=p.framesDeg[+s.value],raw=drawPanel('rawPlot','raw',yd),ripe=drawPanel('ripePlot','ripe',yd);document.getElementById('yaw').textContent=`${yd.toFixed(1)}°`;document.getElementById('count').textContent=`valid raw ${raw[0]}/${raw[1]} · ripe ${ripe[0]}/${ripe[1]}`;}
s.addEventListener('input',draw);draw();
</script></body></html>'''
    Path(path).write_text(html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--ripe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-tree-detections", type=int, default=5)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = read_points(
        (("raw", args.raw), ("ripe", args.ripe)),
        args.minimum_tree_detections,
        args.minimum_score,
    )
    payload = build_payload(points)
    write_html(output, payload)
    print("Wrote {} points across {} yaw frames to {}".format(len(points), len(payload["framesDeg"]), output))


if __name__ == "__main__":
    main()
