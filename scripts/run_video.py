"""Discover the law of a real moving scene: a video file or a live webcam.

    python scripts/run_video.py --video cockatoo.mp4
    python scripts/run_video.py --camera 0 --frames 200          (webcam; Windows works)

Frames are turned grey and shrunk by --scale. The first half is the training split, the next
20% decides which terms are accepted, the last 30% is the test the law never saw. The law is
compared with 'nothing changes' (hold-last) and with the fixed-library regression.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
from luotain2.stream import Stream
from luotain2.discover import Discoverer, nmse
from luotain2.baselines import default_library, stlsq


def read_frames(a):
    cap = cv2.VideoCapture(a.camera if a.video is None else a.video)
    frames = []
    while len(frames) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (g.shape[1] // a.scale, g.shape[0] // a.scale), interpolation=cv2.INTER_AREA)
        frames.append(g.astype(np.float64) / 255.0)
    cap.release()
    if len(frames) < 30:
        sys.exit("Fewer than 30 frames could be read.")
    return np.array(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=280)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--max-lag", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    F = read_frames(a)
    T = len(F)
    i1, i2 = int(0.5 * T), int(0.7 * T)
    L = a.max_lag
    tr = Stream(F[:i1], "dynamics", L, a.margin, periodic=False)
    va = Stream(F[i1 - L:i2], "dynamics", L, a.margin, periodic=False)
    te = Stream(F[i2 - L:], "dynamics", L, a.margin, periodic=False)
    hold = nmse(te.y, te.frames[te.targets - 1][:, te.mask].ravel())
    print(f"{T} frames of {F.shape[1]}x{F.shape[2]}. Hold-last test NMSE {hold:.4e}")
    t0 = time.time()
    lib = stlsq(tr, va, default_library(max_lag=L))
    e_lib = nmse(te.y, te.design(lib.terms) @ lib.coef)
    print(f"fixed library: {len(lib.terms)} terms, test NMSE {e_lib:.4e}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    d = Discoverer(mode="dynamics", max_lag=L, margin=a.margin)
    law = d.run(tr, va, verbose=True)
    e_law = nmse(te.y, te.design(law.terms) @ law.coef)
    print(f"luotain2 law: {len(law.terms)} terms, test NMSE {e_law:.4e}  ({time.time()-t0:.0f}s)")
    print("   ", law.describe(4))
    out = dict(source=a.video or f"camera {a.camera}", frames=T, size=list(F.shape[1:]),
               hold_last=hold, library_terms=len(lib.terms), library_test=e_lib,
               law_terms=len(law.terms), law_test=e_law, law=law.describe(4),
               steps=[{k: v for k, v in s.items() if k not in ("resid_frame", "corr_map", "added")} for s in law.history])
    path = a.out or os.path.join("results", "video_" + os.path.splitext(os.path.basename(a.video or "camera"))[0] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
