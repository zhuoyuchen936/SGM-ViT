#!/usr/bin/env python3
"""Compact A1 ablation summary across 4 variants."""
import json
import os

BASE = "/home/pdongaa/workspace/SGM-ViT/results/ablation_a1"
VARIANTS = ["V0", "V1", "V2", "V3"]

header = "Variant   | KITTI EPE / D1    | SF  EPE / bad1    | ETH3D EPE / bad1  | Mid EPE / bad2"
print(header)
print("-" * len(header))
for v in VARIANTS:
    path = os.path.join(BASE, v, "summary.json")
    d = json.load(open(path))
    r = d["per_dataset"]
    k = r["kitti"]
    s = r["sceneflow"]
    e = r["eth3d"]
    m = r["middlebury"]
    kepe = k["epe"]; kd1 = k.get("d1", 0)
    sepe = s["epe"]; sbad1 = s.get("bad1", 0)
    eepe = e["epe"]; ebad1 = e.get("bad1", 0)
    mepe = m["epe"]; mbad2 = m.get("bad2", 0)
    print(f"{v:<10}| {kepe:.3f} / {kd1:.2f}%    | {sepe:.3f} / {sbad1:.2f}%    | {eepe:.3f} / {ebad1:.2f}%    | {mepe:.3f} / {mbad2:.2f}%")

# Also write a flat table for paper
out = {}
for v in VARIANTS:
    path = os.path.join(BASE, v, "summary.json")
    d = json.load(open(path))
    r = d["per_dataset"]
    out[v] = {ds: {"epe": r[ds]["epe"], list(r[ds].keys())[-1] if r[ds].get("d1_key") else "metric": r[ds].get(r[ds].get("d1_key", ""), 0)} for ds in r}
with open(os.path.join(BASE, "combined.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote", os.path.join(BASE, "combined.json"))
