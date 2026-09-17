#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import custom_datasets
import models.segmentor
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='S2C2Seg evaluation with MMSeg')
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', default='./work_logs')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None,
        help='k=v overrides, e.g. '
             'model.x_options.X_SPATIAL_REFINE=3')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = 'none'
    cfg.work_dir = osp.join(
        args.work_dir,
        '%s_%d' % (osp.splitext(osp.basename(args.config))[0], os.getpid()))

    runner = Runner.from_cfg(cfg)
    results = runner.test()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    if hasattr(model, 'get_eval_metrics'):
        results.update(model.get_eval_metrics())

    metrics = ' '.join(
        '%s=%s' % (k.replace('/', '_'), v) for k, v in results.items())
    print('RESULT config=%s %s' % (osp.basename(args.config), metrics))

    import torch
    if torch.cuda.is_available():
        print('PEAKMEM allocated_mb=%.0f reserved_mb=%.0f' % (
            torch.cuda.max_memory_allocated() / 2 ** 20,
            torch.cuda.max_memory_reserved() / 2 ** 20))


if __name__ == '__main__':
    main()
