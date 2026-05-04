"""
Summarize sweep results from a JSONL file produced by sweep.py.

Usage:
    python scripts/experiments/inference_optim/summarize.py \
        --input results.jsonl \
        [--sort_by acc0.25] \
        [--top_n 10] \
        [--extra_cols avg_latency_ms fps mem_peak_res_mib] \
        [--markdown]

Default columns shown: sweep params + acc0.25 + acc0.50.
Use --extra_cols to add any subset of the timing/memory fields recorded in the JSONL.

Available extra columns (all recorded by sweep.py):
    avg_latency_ms   elapsed_s        fps
    ext_bi0_ms       ext_bi1_ms       ext_bi2_ms    ext_total_ms
    mem_avg_alloc_mib  mem_avg_res_mib
    mem_peak_alloc_mib mem_peak_res_mib
"""

import argparse
import json


# Columns that are always treated as sweep params (not metrics)
_ALWAYS_METRIC = {
    'acc0.25', 'acc0.50', 'acc_mask0.25', 'acc_mask0.50',
    'elapsed_s', 'avg_latency_ms', 'fps',
    'ext_bi0_ms', 'ext_bi1_ms', 'ext_bi2_ms', 'ext_total_ms',
    'mem_avg_alloc_mib', 'mem_avg_res_mib',
    'mem_peak_alloc_mib', 'mem_peak_res_mib',
}

_DEFAULT_METRIC_COLS = ['acc0.25', 'acc0.50']


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _fmt(val):
    if isinstance(val, float):
        return f'{val:.5f}'
    return str(val) if val != '' else ''


def _build_table(results, param_keys, metric_cols):
    header = param_keys + metric_cols
    rows = []
    for r in results:
        row = [_fmt(r.get(k, '')) for k in param_keys]
        row += [_fmt(r.get(c, '')) for c in metric_cols]
        rows.append(row)
    return header, rows


def _print_plain(header, rows):
    col_w = max(14, max(len(h) for h in header))
    print('  ' + '  '.join(h.ljust(col_w) for h in header))
    print('  ' + '  '.join('-' * col_w for _ in header))
    for row in rows:
        print('  ' + '  '.join(v.ljust(col_w) for v in row))


def _print_markdown(header, rows):
    col_widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(header)
    ]

    def fmt_row(cells):
        return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, col_widths)) + ' |'

    print(fmt_row(header))
    print('| ' + ' | '.join('-' * w for w in col_widths) + ' |')
    for row in rows:
        print(fmt_row(row))


def print_table(results, sort_by, top_n, extra_cols, markdown):
    if not results:
        print('No results found.')
        return

    results = sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)
    if top_n > 0:
        results = results[:top_n]

    param_keys = [k for k in results[0].keys() if k not in _ALWAYS_METRIC]
    metric_cols = _DEFAULT_METRIC_COLS + [c for c in extra_cols if c not in _DEFAULT_METRIC_COLS]

    header, rows = _build_table(results, param_keys, metric_cols)
    if markdown:
        _print_markdown(header, rows)
    else:
        _print_plain(header, rows)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--input', required=True, help='Path to sweep_results.jsonl')
    p.add_argument('--sort_by', default='acc0.25',
                   help='Column to sort by descending (default: acc0.25)')
    p.add_argument('--top_n', type=int, default=0,
                   help='Show only top N results (0 = all)')
    p.add_argument('--extra_cols', nargs='*', default=[],
                   metavar='COL',
                   help='Extra metric columns to include in the table, e.g. '
                        'avg_latency_ms fps mem_peak_res_mib ext_total_ms')
    p.add_argument('--markdown', action='store_true',
                   help='Output as markdown table')
    args = p.parse_args()

    results = load_results(args.input)
    print(f'Loaded {len(results)} result(s) from {args.input}')
    print(f'Sorted by {args.sort_by} (descending):\n')
    print_table(results, args.sort_by, args.top_n, args.extra_cols, args.markdown)


if __name__ == '__main__':
    main()
