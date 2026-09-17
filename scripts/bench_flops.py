#!/usr/bin/env python3
"""FLOPs accounting (torch.profiler; MAC-dominated terms).

Unit measurements:
  CLIP ViT-B/16 visual (336 tile), CLIP text encoding (single prompt),
  CLIPSeg full forward (single prompt; per-prompt unit of the two-stage reference schedule),
  CLIPSeg vision-only (once per window in the released pipeline),
  CLIPSeg decoder-only (per-prompt unit in the released pipeline),
  DINO ViT-B/8 (the VFM of ProxyCLIP; baseline component).

Formulas (per 336x336 window):
  ProxyCLIP baseline        = CLIP_vis + DINO
  released pipeline (ours)  = CLIP_vis + DINO + CLIPSeg_vis + CLIPSeg_dec x C
  two-stage reference    = 2 x CLIP_vis + CLIPSeg_full x (C + C')
  Text encoding CLIP_text x C is a one-off cacheable cost and is listed separately.
"""
import argparse
import csv
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", type=int, default=336)
    p.add_argument("--clipseg-size", type=int, default=352)
    p.add_argument("--out", default="results/efficiency/flops.md")
    p.add_argument("--csv", default="results/efficiency/flops_units.csv")
    return p.parse_args()


def flops_of(fn, warmup=2):
    import torch
    from torch.profiler import ProfilerActivity, profile
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.flops for e in prof.key_averages() if e.flops > 0)


def format_report(g, tile=336):
    lines = ["# FLOPs accounting (torch.profiler, GFLOPs, %dx%d window)"
             % (tile, tile), "",
             "## Unit costs", "",
             "| unit | GFLOPs |", "|---|---:|"]
    for k, v in g.items():
        lines.append("| %s | %.1f |" % (k, v))

    lines += ["", "## Per-window totals", "",
              "- ProxyCLIP baseline = clip_visual + dino = **%.1f**"
              % (g["clip_visual"] + g["dino_vitb8"]),
              "- released pipeline (ours) = baseline + clipseg_vision + clipseg_decoder x C"
              " = %.1f + %.2f x C"
              % (g["clip_visual"] + g["dino_vitb8"]
                 + g["clipseg_vision_only"], g["clipseg_decoder_per_prompt"]),
              "- two-stage reference = 2 x clip_visual + clipseg_full x (C+C')"
              " = %.1f + %.1f x (C+C')"
              % (2 * g["clip_visual"], g["clipseg_full_per_prompt"]),
              "- text encoding (one-off, cacheable) = %.1f x C"
              % g["clip_text_per_prompt"], "",
              "## Representative vocabulary sizes (released pipeline)", "",
              "| Vocabulary size | baseline | released pipeline | relative cost |", "|---:|---:|---:|---:|"]
    base = g["clip_visual"] + g["dino_vitb8"]
    for c in (21, 59, 81, 150, 171, 303):
        ours = base + g["clipseg_vision_only"] \
            + g["clipseg_decoder_per_prompt"] * c
        lines.append("| %d | %.1f | %.1f | %.1fx |"
                     % (c, base, ours, ours / base))
    return "\n".join(lines)


def main():
    import torch
    from open_clip import create_model, tokenizer as clip_tokenizer

    torch.set_grad_enabled(False)
    args = parse_args()
    dev = "cuda"

    clip = create_model("ViT-B/16", pretrained="openai",
                        precision="fp16").eval().to(dev)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    seg = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").eval().to(dev)
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    dino = dino.half().eval().to(dev)

    tile = torch.randn(1, 3, args.tile, args.tile, device=dev).half()
    seg_px = torch.randn(1, 3, args.clipseg_size, args.clipseg_size,
                         device=dev)
    tok = clip_tokenizer.tokenize(["a photo of a dog"]).to(dev)
    seg_tok = proc.tokenizer(["dog"], return_tensors="pt", padding=True)
    seg_tok = {k: v.to(dev) for k, v in seg_tok.items()}

    units = {}
    units["clip_visual"] = flops_of(
        lambda: clip.visual(tile, None, 1.2, 3.0, return_cls=True))
    units["clip_text_per_prompt"] = flops_of(lambda: clip.encode_text(tok))
    units["clipseg_full_per_prompt"] = flops_of(
        lambda: seg(pixel_values=seg_px, **seg_tok))
    units["clipseg_vision_only"] = flops_of(
        lambda: seg.clip.vision_model(pixel_values=seg_px,
                                      output_hidden_states=True,
                                      return_dict=True))

    vision_outputs = seg.clip.vision_model(
        pixel_values=seg_px, output_hidden_states=True, return_dict=True)
    activations = [vision_outputs.hidden_states[layer + 1]
                   for layer in seg.extract_layers]
    cond = seg.clip.get_text_features(**seg_tok)
    units["clipseg_decoder_per_prompt"] = flops_of(
        lambda: seg.decoder(
            [a.expand(1, -1, -1).contiguous() for a in activations],
            cond, return_dict=True))

    dino_in = torch.randn(1, 3, args.tile, args.tile, device=dev).half()
    units["dino_vitb8"] = flops_of(
        lambda: dino.get_intermediate_layers(dino_in)[0])

    g = {k: v / 1e9 for k, v in units.items()}
    text = format_report(g, args.tile)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["unit", "gflops"])
        for k, v in g.items():
            writer.writerow([k, "%.3f" % v])


if __name__ == "__main__":
    main()
