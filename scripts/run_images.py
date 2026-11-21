"""Still images: discover each image's causal law and score it as a lossless predictor.

Every image is cut into horizontal bands: the law is fitted on the top band, accepted or
rejected on the next band, and scored on the bottom band it never saw. Compared against
the JPEG-LS median predictor and two fixed least-squares stencils (radius 1 and radius 2)
fitted on the same top band. Also checks a synthetic texture whose true law is known.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import skimage.data as skd
from skimage.color import rgb2gray
from luotain2.stream import Stream
from luotain2.discover import Discoverer, lstsq, nmse
from luotain2.coding import med_predict, entropy_bits, causal_stencil, bits_per_pixel
from luotain2.worlds import causal_ar_texture
from luotain2.grammar import term_name

IMAGES = ["brick", "grass", "gravel", "camera", "moon", "coins", "text", "page", "astronaut", "coffee"]


def load(name):
    a = getattr(skd, name)()
    if a.ndim == 3:
        a = (rgb2gray(a) * 255).round().astype(np.uint8)
    return a


def splits(H, W, margin):
    rows = np.arange(H)[:, None] * np.ones((1, W), bool)
    lo, hi = margin, H - margin
    a, b = lo + int(0.45 * (hi - lo)), lo + int(0.62 * (hi - lo))
    return (rows >= lo) & (rows < a), (rows >= a) & (rows < b), (rows >= b) & (rows < hi)


def run_image(img, margin=32, verbose=False):
    img = np.asarray(img)
    f = img.astype(np.float64)
    H, W = f.shape
    r_tr, r_va, r_te = splits(H, W, margin)
    S = lambda r: Stream(f, mode="static-causal", margin=margin, periodic=False, region=r)
    tr, va, te = S(r_tr), S(r_va), S(r_te)
    d = Discoverer(mode="static-causal", margin=margin, max_terms=24)
    t0 = time.time()
    law = d.run(tr, va, verbose=verbose)
    secs = time.time() - t0
    out = {"shape": [H, W], "seconds": secs, "law": law.describe(4), "n_terms": len(law.terms),
           "terms": [term_name(t) for t in law.terms],
           "offsets": [[t[3], t[4]] for t in law.terms],
           "probes": law.probes, "killed": law.killed,
           "reach": max([max(abs(t[3]), abs(t[4])) for t in law.terms] or [0])}
    if img.dtype == np.uint8:
        u8 = img
        med = med_predict(f.astype(np.int64))
        out["bpp_med"] = entropy_bits((u8.astype(np.int64) - med)[r_te & te.mask])
        out["bpp_raw"] = entropy_bits(u8[te.mask].astype(np.int64))
        for name, stencil in (("ls_r1", causal_stencil(1)), ("ls_r2", causal_stencil(2))):
            c = lstsq(tr.design(stencil), tr.y)
            out[f"bpp_{name}"] = bits_per_pixel(te, stencil, c, u8)
            out[f"nmse_{name}"] = nmse(te.y, te.design(stencil) @ c)
        out["bpp_law"] = bits_per_pixel(te, law.terms, law.coef, u8)
        out["law_bytes"] = 4 * len(law.terms) + 4    # offsets (2 bytes) + 16-bit coefficient each
        out["bpp_law_incl_rule"] = out["bpp_law"] + 8 * out["law_bytes"] / te.mask.sum()
    out["nmse_law"] = nmse(te.y, te.design(law.terms) @ law.coef)
    return out, law


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=IMAGES)
    ap.add_argument("--out", default="results/images.json")
    a = ap.parse_args()
    res = {}
    # ground truth first: a texture made by a known causal law
    tex, truth = causal_ar_texture(seed=7, n=256)
    o, law = run_image(tex, margin=16)
    o["truth"] = " ".join(f"{c:+.3f}*{term_name(t)}" for t, c in truth.items())
    o["exact"] = set(law.terms) == set(truth)
    res["synthetic AR texture"] = o
    print(f"synthetic AR texture exact={o['exact']}  {o['law']}  | truth {o['truth']}", flush=True)
    for name in a.images:
        o, law = run_image(load(name))
        res[name] = o
        print(f"{name:10s} terms {o['n_terms']:2d} reach {o['reach']:2d}  bpp: raw {o['bpp_raw']:.3f}  MED {o['bpp_med']:.3f}"
              f"  LS r1 {o['bpp_ls_r1']:.3f}  LS r2 {o['bpp_ls_r2']:.3f}  law {o['bpp_law']:.3f}  ({o['seconds']:.0f}s)", flush=True)
        print("           ", o["law"], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
