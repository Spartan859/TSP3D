#!/usr/bin/env python3
"""
Clean .pth checkpoints in each folder under outputs/logs/scanrefer.
Rules:
 1. Parse log.txt for these metrics lines (example):
    "3dcnn Acc0.25: Top-1: 0.56077"
    "3dcnn Acc0.50: Top-1: 0.45774"
    "Acc_mask0.25 0.5817914213624895"
    "Acc_mask0.50 0.51419259882254"
 2. For each of the four metrics, keep the checkpoint (.pth) of the epoch that achieved the maximum value for that metric (if ties, keep the latest epoch).
 3. After collecting up to 4 epochs to keep, delete other .pth files in the folder.

By default runs in dry-run mode and prints what would be deleted. Use --apply to actually delete files.
"""

import re
import argparse
from pathlib import Path
import json
import sys
import shutil

METRIC_PATTERNS = {
    'acc0.25': re.compile(r"3dcnn Acc0\.25: Top-1:\s*([0-9]*\.?[0-9]+)"),
    'acc0.50': re.compile(r"3dcnn Acc0\.50: Top-1:\s*([0-9]*\.?[0-9]+)"),
    'mask0.25': re.compile(r"Acc_mask0\.25\s*([0-9]*\.?[0-9]+)"),
    'mask0.50': re.compile(r"Acc_mask0\.50\s*([0-9]*\.?[0-9]+)")
}
METRIC_TITLES = {
    'acc0.25': '3dcnn Acc0.25',
    'acc0.50': '3dcnn Acc0.50',
    'mask0.25': 'Acc_mask0.25',
    'mask0.50': 'Acc_mask0.50'
}

EPOCH_PATTERN = re.compile(r"\[?(?:[0-9]{2}/[0-9]{2})\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\]?\s+logs INFO: (?:Eval: \[(?P<epoch>\d+)\]|epoch\s+(?P<epoch2>\d+))")
# The log layout may vary; we'll also search for lines that say "Eval: [<epoch>]" and capture epoch.
EVAL_EPOCH_LINE = re.compile(r"Eval: \[(?P<epoch>\d+)\]")


def parse_log_for_metrics(log_path: Path):
    """Parse log.txt and return a dict mapping epoch -> dict(metric_name -> value)
    If no epoch can be determined for a metric line, that line will be ignored.
    """
    if not log_path.exists():
        return {}

    epoch_metrics = {}
    current_epoch = None

    with log_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # detect Eval epoch lines
            m_eval = EVAL_EPOCH_LINE.search(line)
            if m_eval:
                try:
                    current_epoch = int(m_eval.group('epoch'))
                except:
                    current_epoch = None
                continue

            # also detect explicit 'epoch <num>' lines
            m_epoch = re.search(r"epoch\s+(\d+)", line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))

            # try each metric
            for key, patt in METRIC_PATTERNS.items():
                m = patt.search(line)
                if m:
                    try:
                        val = float(m.group(1))
                    except:
                        continue
                    if current_epoch is None:
                        # If we don't know the epoch for this line, skip it
                        continue
                    epoch_metrics.setdefault(current_epoch, {})[key] = val
    return epoch_metrics


def select_best_epochs(epoch_metrics: dict):
    """Given epoch_metrics: {epoch: {metric: value}}, return set of epochs to keep (one per metric)
    If multiple epochs share the same metric value, keep the largest epoch (latest).
    """
    best_epochs = set()
    # build metric -> list of (epoch, value)
    per_metric = {k: [] for k in METRIC_PATTERNS.keys()}
    for epoch, metrics in epoch_metrics.items():
        for k in per_metric.keys():
            if k in metrics:
                per_metric[k].append((epoch, metrics[k]))
    # for each metric select best epoch
    for k, items in per_metric.items():
        if not items:
            continue
        # find max value, if tie pick max epoch
        items_sorted = sorted(items, key=lambda x: (x[1], x[0]))
        best_epoch = items_sorted[-1][0]
        best_epochs.add(best_epoch)
    return best_epochs


def select_best_epoch_per_metric(epoch_metrics: dict):
    """Return metric -> best epoch (tie-break by latest epoch)."""
    best_per_metric = {}
    for metric in METRIC_PATTERNS.keys():
        items = []
        for epoch, metrics in epoch_metrics.items():
            if metric in metrics:
                items.append((epoch, metrics[metric]))
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: (x[1], x[0]))
        best_per_metric[metric] = items_sorted[-1][0]
    return best_per_metric


def parse_eval_blocks(log_path: Path):
    """Return {epoch: {'eval': str|None, 'metrics': [str, ...]}} from log."""
    if not log_path.exists():
        return {}
    blocks = {}
    current_eval_epoch = None
    with log_path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            m_eval = EVAL_EPOCH_LINE.search(line)
            if m_eval:
                current_eval_epoch = int(m_eval.group('epoch'))
                blocks.setdefault(current_eval_epoch, {'eval': None, 'metrics': []})
                blocks[current_eval_epoch]['eval'] = line
                continue

            if current_eval_epoch is None:
                continue
            for patt in METRIC_PATTERNS.values():
                if patt.search(line):
                    blocks.setdefault(current_eval_epoch, {'eval': None, 'metrics': []})
                    blocks[current_eval_epoch]['metrics'].append(line)
                    break
    return blocks


def write_best_markdown(folder: Path, log_path: Path, epoch_metrics: dict):
    """Write best.md under folder with best epoch snippets for each metric."""
    best_md_path = folder / 'best.md'
    best_per_metric = select_best_epoch_per_metric(epoch_metrics)
    eval_blocks = parse_eval_blocks(log_path)

    if not best_per_metric:
        best_md_path.write_text('# No best metrics found\n', encoding='utf-8')
        return

    lines = []
    for metric in METRIC_PATTERNS.keys():
        if metric not in best_per_metric:
            continue
        epoch = best_per_metric[metric]
        title = METRIC_TITLES.get(metric, metric)
        lines.append(f'## {title}: {epoch}')
        lines.append('```')

        block = eval_blocks.get(epoch, {})
        eval_line = block.get('eval')
        metric_lines = block.get('metrics', [])
        if eval_line:
            lines.append(eval_line)
        if metric_lines:
            lines.extend(metric_lines)
        if not eval_line and not metric_lines:
            lines.append(f'(No matching Eval block lines found in log for epoch {epoch})')
        lines.append('```')
        lines.append('')

    best_md_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def find_pth_files(folder: Path):
    return sorted([p for p in folder.glob('*.pth')])


def epoch_from_pth(p: Path):
    # common naming patterns: epoch_123.pth, model_epoch_123.pth, ckpt_epoch_123.pth, 123.pth
    name = p.stem
    m = re.search(r"(\d{1,4})", name)
    if m:
        return int(m.group(1))
    return None


def process_folder(folder: Path, apply: bool = False, dry_run: bool = True):
    log_path = folder / 'log.txt'
    epoch_metrics = parse_log_for_metrics(log_path)
    if epoch_metrics:
        write_best_markdown(folder, log_path, epoch_metrics)
    if not epoch_metrics:
        # If no metrics parsed, delete entire folder (or dry-run report)
        if dry_run:
            print(f"No metrics parsed for {folder}; would REMOVE entire folder (dry-run)")
        else:
            try:
                shutil.rmtree(folder)
                print(f"Removed folder {folder}")
            except Exception as e:
                print(f"Failed to remove folder {folder}: {e}")
        return
    best_epochs = select_best_epochs(epoch_metrics)
    if not best_epochs:
        print(f"No best epochs found for {folder}; skipping")
        return
    pth_files = find_pth_files(folder)
    if not pth_files:
        print(f"No .pth files in {folder}")
        return

    # map pth -> epoch (if found)
    keep_files = set()
    for p in pth_files:
        e = epoch_from_pth(p)
        if e is not None and e in best_epochs:
            keep_files.add(p)
    # if some best_epoch not matched by filename, print warning
    unmatched = [e for e in best_epochs if not any(epoch_from_pth(p) == e for p in pth_files)]
    if unmatched:
        print(f"Warning: in {folder} best epochs {unmatched} not matched to any .pth filename")

    # decide deletions: all pth not in keep_files
    to_delete = [p for p in pth_files if p not in keep_files]

    if dry_run:
        print(f"Folder: {folder}")
        print(f"  Best epochs to keep: {sorted(best_epochs)}")
        print(f"  .pth files found: {len(pth_files)}; will keep {len(keep_files)}; would delete {len(to_delete)} files")
        # for p in to_delete:
        #     print(f"    DELETE (dry): {p.name}")
    else:
        print(f"Applying deletions in {folder}")
        for p in to_delete:
            try:
                p.unlink()
                print(f"    Deleted: {p.name}")
            except Exception as e:
                print(f"    Failed to delete {p.name}: {e}")


def find_scanrefer_dirs(rootp: Path):
    scanrefer_dirs = []
    if rootp.is_dir() and rootp.name == 'scanrefer':
        scanrefer_dirs.append(rootp)
    for p in rootp.rglob('scanrefer'):
        if p.is_dir():
            scanrefer_dirs.append(p)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in scanrefer_dirs:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def main(root: str, apply: bool = False):
    rootp = Path(root)
    if not rootp.exists():
        print(f"Root {root} does not exist")
        return
    scanrefer_dirs = find_scanrefer_dirs(rootp)
    if not scanrefer_dirs:
        print(f"No scanrefer directories found under {root}")
        return
    print("Found scanrefer directories:")
    for p in scanrefer_dirs:
        print(f"  {p}")
    for scanrefer_dir in scanrefer_dirs:
        for sub in sorted(scanrefer_dir.iterdir()):
            if not sub.is_dir():
                continue
            process_folder(sub, apply=apply, dry_run=not apply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='outputs/logs/scanrefer', help='Root path containing scanrefer run folders')
    parser.add_argument('--apply', action='store_true', help='Actually delete files; default is dry-run')
    args = parser.parse_args()

    main(args.root, apply=args.apply)
