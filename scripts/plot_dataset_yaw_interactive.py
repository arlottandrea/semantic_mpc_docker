#!/usr/bin/env python3
"""Create a standalone interactive XY/yaw viewer for raw and ripe datasets."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def wrap_angle(values):
    return np.arctan2(np.sin(values), np.cos(values))


def read_points(paths):
    points = []
    for label, path in paths:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    points.append((float(row["x"]), float(row["y"]), float(row["yaw"]), label))
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
        "framesDeg": np.rad2deg(frames).round(4).tolist(),
        "toleranceDeg": float(np.rad2deg(tolerance)),
    }


def write_html(path, payload):
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training dataset by yaw</title>
<style>
body{font-family:Arial,sans-serif;margin:18px;background:#f8fafc;color:#111827}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
input{width:min(620px,70vw)} canvas{background:white;border:1px solid #cbd5e1;max-width:92vw;height:auto}
.legend{margin-top:8px}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 5px 0 14px}
</style></head><body>
<h2>Training dataset — XY slices by yaw</h2>
<div class="controls"><label for="slider">Yaw</label><input id="slider" type="range" min="0" value="0" step="1">
<strong id="yaw"></strong><span id="count"></span></div>
<canvas id="plot" width="900" height="760"></canvas>
<div class="legend"><span class="dot" style="background:#f97316"></span>raw
<span class="dot" style="background:#22c55e"></span>ripe</div>
<script>
const p=__PAYLOAD__, s=document.getElementById('slider'), c=document.getElementById('plot'), ctx=c.getContext('2d');
s.max=p.framesDeg.length-1;
const xmin=Math.min(...p.x),xmax=Math.max(...p.x),ymin=Math.min(...p.y),ymax=Math.max(...p.y),pad=45;
function diff(a,b){let d=a-b;while(d>180)d-=360;while(d<-180)d+=360;return Math.abs(d)}
function px(x){return pad+(x-xmin)/Math.max(1e-9,xmax-xmin)*(c.width-2*pad)}
function py(y){return c.height-pad-(y-ymin)/Math.max(1e-9,ymax-ymin)*(c.height-2*pad)}
function draw(){const yd=p.framesDeg[+s.value];ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle='#94a3b8';ctx.strokeRect(pad,pad,c.width-2*pad,c.height-2*pad);
let raw=0,ripe=0;for(let i=0;i<p.x.length;i++){if(diff(p.yawDeg[i],yd)>p.toleranceDeg)continue;const isRaw=p.label[i]==='raw';ctx.fillStyle=isRaw?'rgba(249,115,22,.72)':'rgba(34,197,94,.72)';ctx.beginPath();ctx.arc(px(p.x[i]),py(p.y[i]),3,0,2*Math.PI);ctx.fill();if(isRaw)raw++;else ripe++;}
ctx.fillStyle='#334155';ctx.font='13px Arial';ctx.fillText(`x [${xmin.toFixed(2)}, ${xmax.toFixed(2)}]`,pad,c.height-14);ctx.save();ctx.translate(15,c.height-pad);ctx.rotate(-Math.PI/2);ctx.fillText(`y [${ymin.toFixed(2)}, ${ymax.toFixed(2)}]`,0,0);ctx.restore();
document.getElementById('yaw').textContent=`${yd.toFixed(1)}°`;document.getElementById('count').textContent=`raw ${raw} · ripe ${ripe} · total ${raw+ripe}`;}
s.addEventListener('input',draw);draw();
</script></body></html>'''
    Path(path).write_text(html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--ripe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = read_points((("raw", args.raw), ("ripe", args.ripe)))
    payload = build_payload(points)
    write_html(output, payload)
    print("Wrote {} points across {} yaw frames to {}".format(len(points), len(payload["framesDeg"]), output))


if __name__ == "__main__":
    main()
