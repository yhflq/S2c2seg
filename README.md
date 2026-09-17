# S2C2Seg

S2C2Seg / S2C2Seg++ is a training-free implementation for open-vocabulary
semantic segmentation. It uses frozen CLIP, DINO, and CLIPSeg models and
requires no additional training.

## Installation

```bash
conda create -n s2c2seg python=3.10 -y
conda activate s2c2seg
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Model weights are downloaded automatically on the first run.

## Data Preparation

Place or symlink datasets under `data/` using the directory structure expected
by mmsegmentation. Supported datasets include VOC, PASCAL Context, ADE20K,
Cityscapes, COCO-Object, and COCO-Stuff.

## Evaluation

| Script | Operating point | Paper entry |
|---|---|---|
| `scripts/run_conference.sh` | single-pass run of the released pipeline (T_max=0, patch-mean global score, no post-processing) | Conference row of Table I |
| `scripts/run_journal.sh` | released S2C2Seg++ configuration | Journal rows of Tables I and II |
| `scripts/run_efficiency_suite.sh` | vocabulary scaling, FLOPs, and baseline/R1 end-to-end timings; calls `scripts/bench_efficiency.py`, `scripts/bench_flops.py`, and `scripts/analyze_efficiency.py` | Tables X and XI |

```bash
bash scripts/run_conference.sh
bash scripts/run_journal.sh
bash scripts/run_efficiency_suite.sh
```

Outputs are written under `results/`. Every protocol is evaluated with
`eval.py --config configs/cfg_<protocol>.py --cfg-options ...`; the per-protocol
option strings are listed in the `opts_for` function of each script.

The efficiency suite writes its CSV files and `EFFICIENCY_REPORT.md` under
`results/efficiency/`; `python scripts/analyze_efficiency.py` rebuilds the report
and the vocabulary-scaling figure from those CSV files. The suite measures the
released method and the ProxyCLIP baseline.

## License

MIT License (see `LICENSE`). The `open_clip/` directory is vendored from
[open_clip](https://github.com/mlfoundations/open_clip) (MIT License).
