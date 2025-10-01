# make_colored_ply.py
import json, sys, numpy as np

IN, OUT, CLASS = sys.argv[1], sys.argv[2], sys.argv[3]  # json -> ply, class name
data = json.load(open(IN))
P = np.asarray(data["points"], float)
s = np.asarray(data["scores"][CLASS], float)
s = (s - s.min()) / (np.ptp(s) + 1e-8)

# red/gray palette like your render
low = np.array([206, 212, 216]) / 255.0  # light gray
high = np.array([220, 30, 38])   / 255.0  # red
RGB = (low[None,:] * (1 - s[:,None]) + high[None,:] * s[:,None])

# write ASCII PLY with per-vertex color
with open(OUT, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(P)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for (x,y,z),(r,g,b) in zip(P, (RGB*255).round().astype(int)):
        f.write(f"{x} {y} {z} {r} {g} {b}\n")
