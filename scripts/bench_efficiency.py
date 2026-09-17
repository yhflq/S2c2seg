#!/usr/bin/env python3
import argparse
import glob
import os.path as osp
import statistics
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image

from models import S2C2Config
from models import css as css_module
from models import scoring, spatial_refine
from open_clip import create_model, tokenizer as clip_tokenizer

STAGE_KEYS = (
    "CLIP encoding (C)",
    "CLIPSeg inference (C)",
    "CSS selection",
    "CLIP encoding (C')",
    "CLIPSeg inference (C')",
    "CSG fusion",
    "Semantic-spatial alignment",
    "Post-processing",
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=20)
    p.add_argument("--tile", type=int, default=336)
    p.add_argument("--clipseg-size", type=int, default=352)
    p.add_argument("--image-dir", default="data/VOC2012/JPEGImages")
    p.add_argument("--vocab", default="configs/cls_voc21.txt")
    p.add_argument("--out", default="results/bench_efficiency.md")
    p.add_argument("--mode", choices=("twostage", "nocss"), default="twostage",
                   help="twostage=two-stage reference schedule (CSS compression); nocss=full-vocabulary control")
    p.add_argument("--tag", default=None, help="CSV row label; defaults to the vocabulary file name")
    p.add_argument("--csv", default=None, help="append one summary row to this CSV (used by the sweep)")
    return p.parse_args()


def class_names(path):
    with open(path) as f:
        return [line.split("; ")[0].strip() for line in f if line.strip()]


def load_models(dev):
    clip = create_model("ViT-B/16", pretrained="openai",
                        precision="fp16").eval().to(dev)
    for q in clip.parameters():
        q.requires_grad = False
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    seg = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").half().eval().to(dev)
    for q in seg.parameters():
        q.requires_grad = False
    return clip, proc, seg


def to_tiles(pil, tile):
    """Center-crop one tile x tile window; returns (CLIP-normalised tensor, PIL tile)."""
    w, h = pil.size
    scale = tile / min(w, h)
    pil = pil.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    left, top = (pil.size[0] - tile) // 2, (pil.size[1] - tile) // 2
    pil = pil.crop((left, top, left + tile, top + tile))
    x = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8)
    x = x.view(tile, tile, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), pil, x


class Bench:
    def __init__(self, args, dev="cuda"):
        self.args = args
        self.dev = dev
        self.clip, self.proc, self.seg = load_models(dev)
        self.classes = class_names(args.vocab)
        self.cfg = S2C2Config(tau=0.4, k_min=6, k_max=20,
                              rerank_weights=(1.0, 1.0, 1.0), t_max=0,
                              global_score="patch_mean", residual_beta=0.0,
                              adaptive_admit=False)
        self.seg_text = []
        for name in self.classes:
            tok = self.proc.tokenizer([name], return_tensors="pt", padding=True)
            self.seg_text.append({k: v.to(dev) for k, v in tok.items()})


    @torch.no_grad()
    def clip_encoding(self, img, indices):
        names = [self.classes[i] for i in indices]
        tok = clip_tokenizer.tokenize(["a photo of a %s" % n
                                       for n in names]).to(self.dev)
        text = self.clip.encode_text(tok)
        text = text / text.norm(dim=-1, keepdim=True)
        patches, cls_feat = self.clip.visual(
            img.half().to(self.dev), None, 1.2, 3.0, return_cls=True)
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
        grid = int(patches.shape[1] ** 0.5)
        dense = (patches @ text.T).permute(0, 2, 1).reshape(
            1, len(names), grid, grid)
        return dense[0].float(), (cls_feat @ text.T)[0].float()

    @torch.no_grad()
    def clipseg_sequential(self, pixel_values, indices):
        logits = []
        for i in indices:
            out = self.seg(pixel_values=pixel_values, **self.seg_text[i])
            l = out.logits
            logits.append(l if l.ndim == 2 else l[0])
        return torch.sigmoid(torch.stack(logits).float())

    def run_image(self, pil):
        args = self.args
        img, tile_pil, raw = to_tiles(pil, args.tile)
        pv = self.proc.image_processor(
            images=[tile_pil.resize((args.clipseg_size,) * 2, Image.BILINEAR)],
            return_tensors="pt")["pixel_values"].half().to(self.dev)
        all_idx = list(range(len(self.classes)))
        t = {}
        torch.cuda.synchronize()
        mark = time.perf_counter()

        def lap(name):
            nonlocal mark
            now = time.perf_counter()
            t[name] = (now - mark) * 1000
            mark = now

        dense, cls_cos = self.clip_encoding(img, all_idx)
        lap("CLIP encoding (C)")
        probs = self.clipseg_sequential(pv, all_idx)
        lap("CLIPSeg inference (C)")
        if self.args.mode == "nocss":
            # Control: no CSS, full-vocabulary fusion, no second-stage encoding
            selected = all_idx
            dense_s, probs_s = dense, probs
            t["CSS selection"] = 0.0
            t["CLIP encoding (C')"] = 0.0
            t["CLIPSeg inference (C')"] = 0.0
            mark = time.perf_counter()
        else:
            s_glob = cls_cos
            s_spat = probs.mean(dim=(-2, -1))
            glob_list = [float(s_glob[i].item()) for i in all_idx]
            spat_list = [float(probs[i].mean().item()) for i in all_idx]
            s_conf, _ = scoring.cross_view_confidence(
                s_glob, s_spat)
            conf_list = [float(s_conf[i].item()) for i in all_idx]
            del glob_list, spat_list, conf_list
            sel = css_module.select(s_glob, s_spat, s_conf, self.cfg)
            selected = sel["selected"].tolist()
            lap("CSS selection")
            dense_s, cls_s = self.clip_encoding(img, selected)
            lap("CLIP encoding (C')")
            probs_s = self.clipseg_sequential(pv, selected)
            lap("CLIPSeg inference (C')")
        g = F.interpolate(dense_s.unsqueeze(0),
                          size=probs_s.shape[-2:], mode="bilinear")[0]
        support = [float(g[i].mean().item()) for i in range(g.shape[0])]
        median = sorted(support)[len(support) // 2]
        w = torch.tensor([1.0 / (1.0 + torch.e ** (-(s - median) / 0.1))
                          for s in support], device=self.dev)
        fused = g + 0.6 * w[:, None, None] * probs_s
        lap("CSG fusion")
        cls_probs = (fused * 40.0).softmax(dim=0)
        refined = spatial_refine.refine(
            raw.to(self.dev), cls_probs, 3, max_side=args.clipseg_size)
        lap("Semantic-spatial alignment")
        out_size = (pil.size[1], pil.size[0])
        up = F.interpolate(refined.unsqueeze(0), size=out_size,
                           mode="bilinear")[0]
        max_prob, pred = up.max(dim=0)
        pred[max_prob < 0.2] = 0
        pred = pred.cpu().numpy()
        max_prob = max_prob.cpu().numpy()
        torch.cuda.synchronize()
        lap("Post-processing")

        t["Per refinement round"] = (t["CLIP encoding (C')"]
                                     + t["CLIPSeg inference (C')"]
                                     + t["CSG fusion"])
        t["Total R0"] = sum(t[k] for k in STAGE_KEYS)
        t["|C'|"] = float(len(selected))
        return t


def main():
    args = parse_args()
    dev = "cuda"
    bench = Bench(args, dev)
    files = sorted(glob.glob(osp.join(args.image_dir, "*.jpg")))[:args.images]
    if not files:
        raise SystemExit("no images under %s" % args.image_dir)
    pils = [Image.open(f).convert("RGB") for f in files]

    for pil in pils[:3]:
        bench.run_image(pil)
    torch.cuda.reset_peak_memory_stats()

    records = [bench.run_image(pil) for pil in pils]
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    reserved_mb = torch.cuda.max_memory_reserved() / 2 ** 20

    med = {k: statistics.median(r[k] for r in records) for k in records[0]}
    params = sum(p.numel() for p in bench.clip.parameters()) \
        + sum(p.numel() for p in bench.seg.parameters())

    lines = ["# Efficiency benchmark results", "",
             "- images: %d (%s), tile=%d, fp16, device=%s, mode=%s"
             % (len(records), args.image_dir, args.tile, dev, args.mode),
             "- vocabulary size |C|: %d, measured median |C'|: %.0f"
             % (len(bench.classes), med["|C'|"]), "",
             "| stage | median (ms) |",
             "|---|---:|"]
    for key in STAGE_KEYS + ("Total R0", "Per refinement round"):
        lines.append("| %s | %.1f |" % (key, med[key]))
    lines += ["",
              "- peak GPU memory: allocated %.0f MB / reserved %.0f MB"
              % (peak_mb, reserved_mb),
              "- parameters: %.0fM" % (params / 1e6)]
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")

    if args.csv:
        import csv as csv_module
        tag = args.tag or osp.splitext(osp.basename(args.vocab))[0]
        header = (["tag", "mode", "num_classes", "cprime_median"]
                  + list(STAGE_KEYS)
                  + ["total_r0_ms", "per_round_ms",
                     "peak_alloc_mb", "peak_reserved_mb", "images"])
        row = ([tag, args.mode, len(bench.classes), "%.0f" % med["|C'|"]]
               + ["%.1f" % med[k] for k in STAGE_KEYS]
               + ["%.1f" % med["Total R0"],
                  "%.1f" % med["Per refinement round"],
                  "%.0f" % peak_mb, "%.0f" % reserved_mb, len(records)])
        write_header = not osp.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv_module.writer(f, lineterminator="\n")
            if write_header:
                writer.writerow(header)
            writer.writerow(row)


if __name__ == "__main__":
    main()
