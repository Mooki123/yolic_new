"""Train one (config, seed) run under the fixed protocol v1 (plan 04, section 2.3).

    python -m yolic_bench.train configs/outdoor_b1_random.yaml --seed 0

Protocol: Adam lr 1e-3, MultiStepLR [100, 125] x0.1, 150 epochs, batch 32, AMP, ImageNet normalisation,
correct horizontal flip, ColorJitter hue 0.1. Cell models train with BCE-with-logits over every valid
(cell, bit); the segmenter (head: lraspp) trains with pixel cross-entropy on the cached group masks and
is read out into cells. A cache with aug.npy (Cityscapes) also draws the rescale-crop state per epoch.
The checkpoint with the best val macro-F1 at 0.5 over `primary_classes` (default: object_classes) is
kept; val is never augmented; train and test are never evaluated during training. Test is evaluated
once, on the selected checkpoint, at the end.

Run directory runs/<name>/seed<k>/: config.yaml, log.csv, last.pt (resume), best.pt, val_probs.npy and
test_probs.npy (float16, (N, n_cells, n_bits) in split-file order), metrics.json.
A run that finds last.pt resumes from it; one that finds metrics.json is skipped.
"""
import argparse
import csv
import json
import logging
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from yolic_bench import heads
from yolic_bench import metrics as M
from yolic_bench.data import CellCache, GpuAugment, batches
from yolic_bench.layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['cache'] = os.path.expanduser(cfg['cache'])
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def is_seg(model):
    return getattr(model, 'seg', False)


def cell_probs(model, logits):
    return model.cell_probs(logits) if is_seg(model) else torch.sigmoid(logits.float())


def masked_bce(probs, y, valid):
    """Mean BCE over valid cells (all cells when valid is None), computed from probabilities."""
    loss = nn.functional.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), y, reduction='none')
    if valid is None:
        return loss.sum(), loss.numel()
    w = valid.unsqueeze(-1).float()
    return (loss * w).sum(), w.sum() * y.shape[-1]


@torch.no_grad()
def predict(model, cache, idx, aug, device, batch_size=128):
    """Cell probabilities (N, n_cells, n_bits) float32 in idx order, plus the mean BCE over valid cells."""
    model.eval()
    out = np.zeros((len(idx),) + cache.labels.shape[1:], dtype=np.float32)
    loss_sum, n = 0.0, 0
    for imgs, lab, valid, _, pos in batches(cache, idx, batch_size, shuffle=False):
        xs = aug.to_input([v.to(device, non_blocking=True) for v in imgs])
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(*xs)
        probs = cell_probs(model, logits)
        s, k = masked_bce(probs, lab.to(device).float(), valid.to(device) if valid is not None else None)
        loss_sum, n = loss_sum + s.item(), n + float(k)
        out[pos] = probs.cpu().numpy()
    return out, loss_sum / max(n, 1)


def by_group(gt, pred, layout, object_classes, valid=None):
    """Object macro-F1 per cell group, over the classes with at least one positive in that group.

    A class with no positives in a group (e.g. only TrafficSign occurs in the outdoor top row) would
    otherwise count as F1 = 0 and swamp the average.
    """
    out = {}
    for g, cells in layout.groups.items():
        c = M.counts(gt, pred, valid, cells)
        present = [k for k in object_classes if c['tp'][k] + c['fn'][k] > 0]
        out[g] = {'object_macro_f1': float(M.macro_f1(c, present)) if present else None,
                  'classes_present': len(present)}
    return out


def report(gt, probs, layout, object_classes, class_names, valid=None, val_gt=None, val_probs=None,
           val_valid=None, primary_classes=None):
    """Every section 2.2 number that does not need other runs (bootstrap and seed stats come later)."""
    primary = object_classes if primary_classes is None else primary_classes
    pred = M.binarize(probs)
    c = M.counts(gt, pred, valid)
    p, r, f = M.prf(c)
    out = {
        'per_class': {name: {'precision': float(p[k]), 'recall': float(r[k]), 'f1': float(f[k]),
                             'support': int(c['tp'][k] + c['fn'][k])} for k, name in enumerate(class_names)},
        'object_macro_f1': float(M.macro_f1(c, object_classes)),
        'primary_classes': [class_names[k] for k in primary],
        'primary_macro_f1': float(M.macro_f1(c, primary)),
        'all_paper': dict(zip(('precision', 'recall', 'f1'), map(float, M.all_score(c, object_classes)))),
        'ap': dict(zip([class_names[k] for k in object_classes],
                       map(float, M.average_precision(gt, probs, object_classes, valid)))),
    }
    bp, br, bf, bs = M.binary_road(gt, pred, valid=valid)
    out['binary_road_all_cells'] = {lab: {'precision': float(bp[i]), 'recall': float(br[i]), 'f1': float(bf[i]),
                                          'support': int(bs[i])} for i, lab in enumerate(('Risk', 'Road'))}
    out['mAP'] = float(np.nanmean(list(out['ap'].values())))
    out['by_group'] = by_group(gt, pred, layout, object_classes, valid)
    out['by_group_primary'] = by_group(gt, pred, layout, primary, valid)
    if val_gt is not None:
        thr = M.tune_thresholds(val_gt, val_probs, object_classes, valid=val_valid)
        ct = M.counts(gt, M.binarize(probs, thr), valid)
        out['tuned'] = {'thresholds': thr.tolist(), 'object_macro_f1': float(M.macro_f1(ct, object_classes)),
                        'primary_macro_f1': float(M.macro_f1(ct, primary))}
    return out


def count_macs(model, cache, device):
    """fvcore MACs for one sample at the cached input size(s) (all views)."""
    from fvcore.nn import FlopCountAnalysis
    logging.getLogger('fvcore').setLevel(logging.ERROR)
    model.eval()
    inputs = tuple(torch.zeros(1, 3, v.shape[1], v.shape[2], device=device) for v in cache.views)
    fl = FlopCountAnalysis(model, inputs)
    fl.unsupported_ops_warnings(False)
    fl.uncalled_modules_warnings(False)
    return int(fl.total())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, help='override (smoke tests only)')
    ap.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg['epochs'] = args.epochs
        cfg['milestones'] = [m * args.epochs // 150 for m in cfg['milestones']]
    run_dir = os.path.join(args.runs, cfg['name'], f'seed{args.seed}')
    if os.path.exists(os.path.join(run_dir, 'metrics.json')):
        print(f'{run_dir} already finished; skipping')
        return
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg | {'seed': args.seed, 'git': git_hash()}, f, sort_keys=False)

    seed_all(args.seed)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    layout = Layout.load(os.path.join(ROOT, cfg['layout']))
    cache = CellCache(cfg['cache'], cfg.get('views'))
    split = os.path.join(ROOT, cfg['split'])
    tr, va, te = (cache.indices(os.path.join(split, p + '.txt')) for p in ('train', 'val', 'test'))
    n_bits = cache.labels.shape[-1]
    obj = cfg['object_classes']
    primary = cfg.get('primary_classes', obj)
    class_names = json.load(open(os.path.join(cfg['cache'], 'meta.json')))['classes']
    input_size = cfg.get('input_size', 224)

    model = heads.build(cfg['model'], layout, n_bits, input_size, n_views=len(cache.views)).to(device)
    model = model.to(memory_format=torch.channels_last)
    seg = is_seg(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg['milestones'], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', True))
    g = torch.Generator().manual_seed(args.seed)
    perm = layout.mirror_permutation() if cfg.get('flip', 'hflip') != 'none' else None
    aug = GpuAugment(perm, cfg.get('flip', 'hflip'), cfg.get('jitter'), cfg.get('normalize', True), g)

    start, best = 1, -1.0
    last = os.path.join(run_dir, 'last.pt')
    if os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        scaler.load_state_dict(st['scaler'])
        g.set_state(st['gen'])
        torch.set_rng_state(st['torch_rng'])
        start, best = st['epoch'] + 1, st['best']
        print(f'resuming from epoch {start}, best val macro-F1 {best:.4f}')

    log_path = os.path.join(run_dir, 'log.csv')
    new_log = not os.path.exists(log_path) or start == 1
    log = open(log_path, 'w' if new_log else 'a', newline='')
    w = csv.writer(log)
    if new_log:
        w.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_primary_macro_f1', 'best', 'img_per_s', 'sec'])
    val_gt = cache.labels[va]
    val_valid = cache.valid[va] if cache.valid is not None else None

    for epoch in range(start, cfg['epochs'] + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        rows = cache.sample_states(tr, g)
        for imgs, lab, valid, mk, _ in batches(cache, rows, cfg['batch'], shuffle=True, generator=g, masks=seg):
            imgs = [v.to(device, non_blocking=True) for v in imgs]
            lab = lab.to(device, non_blocking=True).float()
            valid = valid.to(device, non_blocking=True) if valid is not None else None
            mk = mk.to(device, non_blocking=True) if mk is not None else None
            xs, y, valid, mk = aug.train(imgs, lab, valid, mk)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', True)):
                out = model(*xs)
            if seg:
                loss = nn.functional.cross_entropy(out.float(), mk.long(), ignore_index=255)
            elif valid is None:
                loss = nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            else:
                per = nn.functional.binary_cross_entropy_with_logits(out.float(), y, reduction='none')
                wv = valid.unsqueeze(-1).float()
                loss = (per * wv).sum() / (wv.sum() * y.shape[-1]).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(lab)
            seen += len(lab)
        train_s = time.time() - t0
        probs, val_loss = predict(model, cache, va, aug, device)
        f1 = float(M.macro_f1(M.counts(val_gt, M.binarize(probs), val_valid), primary))
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))
            np.save(os.path.join(run_dir, 'val_probs.npy'), probs.astype(np.float16))
        lr = opt.param_groups[0]['lr']
        sched.step()
        w.writerow([epoch, lr, f'{loss_sum / seen:.6f}', f'{val_loss:.6f}', f'{f1:.5f}', f'{best:.5f}',
                    f'{seen / train_s:.1f}', f'{time.time() - t0:.1f}'])
        log.flush()
        print(f'[{cfg["name"]} s{args.seed}] ep {epoch}/{cfg["epochs"]} loss {loss_sum / seen:.4f} '
              f'val {val_loss:.4f} F1 {f1:.4f} (best {best:.4f}) {seen / train_s:.0f} img/s', flush=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'gen': g.get_state(), 'torch_rng': torch.get_rng_state(),
                    'epoch': epoch, 'best': best}, last + '.tmp')
        os.replace(last + '.tmp', last)
    log.close()

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt'), map_location=device))
    test_probs, test_loss = predict(model, cache, te, aug, device)
    np.save(os.path.join(run_dir, 'test_probs.npy'), test_probs.astype(np.float16))
    val_probs = np.load(os.path.join(run_dir, 'val_probs.npy')).astype(np.float32)
    test_valid = cache.valid[te] if cache.valid is not None else None
    res = report(cache.labels[te], test_probs, layout, obj, class_names, valid=test_valid, val_gt=val_gt,
                 val_probs=val_probs, val_valid=val_valid, primary_classes=primary)
    res |= {'name': cfg['name'], 'seed': args.seed, 'best_val_primary_macro_f1': best, 'test_loss': test_loss,
            'n_test': int(len(te)), 'params': int(sum(p.numel() for p in model.parameters())),
            'macs': count_macs(model, cache, device)}
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=1)
    os.remove(last)
    print(json.dumps({k: res[k] for k in ('primary_macro_f1', 'object_macro_f1', 'all_paper', 'mAP', 'macs')}, indent=1))


if __name__ == '__main__':
    main()
