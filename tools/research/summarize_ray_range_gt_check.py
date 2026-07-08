#!/usr/bin/env python3
"""Summarize GT ray-range check CSV files by condition buckets."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="GT ray-range check 결과를 거리/view/class별로 요약"
    )
    parser.add_argument(
        "summary_csv",
        nargs="+",
        help="summary.csv path(s) from ray_range_gt_check.py",
    )
    parser.add_argument("--range-radius", type=float, default=2.0)
    return parser.parse_args()


def range_bucket(row):
    value = float(row["gt_range_m"])
    if value < 5.0:
        return "0-5m"
    if value < 10.0:
        return "5-10m"
    if value < 20.0:
        return "10-20m"
    return "20m+"


def visible_bucket(row):
    value = int(row["center_visible_views"])
    if value <= 2:
        return "2 views"
    if value == 3:
        return "3 views"
    return "4 views"


def label_bucket(row):
    label = int(row["label"])
    if label == 0:
        return "Car"
    if label == 1:
        return "Pedestrian"
    return str(label)


def summarize(rows, range_radius):
    count = len(rows)
    if count == 0:
        return None
    abs_offsets = sorted(abs(float(row["best_offset_m"])) for row in rows)
    near04 = sum(value <= 0.4 + 1e-9 for value in abs_offsets) / count
    near06 = sum(value <= 0.6 + 1e-9 for value in abs_offsets) / count
    edge = sum(value >= range_radius - 1e-9 for value in abs_offsets) / count
    median = abs_offsets[count // 2]
    gt_is_best = sum(
        float(row["gt_reliability"]) >= float(row["best_reliability"]) - 1e-12
        for row in rows
    ) / count
    return dict(
        n=count,
        near04=near04,
        near06=near06,
        edge=edge,
        median_abs=median,
        gt_is_best=gt_is_best,
    )


def print_summary(name, rows, range_radius):
    stats = summarize(rows, range_radius)
    if stats is None:
        return
    print(
        f"{name:16s} "
        f"n={stats['n']:4d} "
        f"near04={stats['near04']:.3f} "
        f"near06={stats['near06']:.3f} "
        f"edge={stats['edge']:.3f} "
        f"median_abs={stats['median_abs']:.3f} "
        f"gt_is_best={stats['gt_is_best']:.3f}"
    )


def print_grouped(title, rows, key_fn, range_radius):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    print(f"  {title}")
    for key in sorted(grouped):
        print_summary(f"    {key}", grouped[key], range_radius)


def main():
    args = parse_args()
    for summary_path in args.summary_csv:
        path = Path(summary_path)
        rows = list(csv.DictReader(path.open()))
        print(f"\n== {path.parent.name} ({len(rows)} objects) ==")
        print_summary("all", rows, args.range_radius)
        print_grouped("range", rows, range_bucket, args.range_radius)
        print_grouped("visible views", rows, visible_bucket, args.range_radius)
        print_grouped("label", rows, label_bucket, args.range_radius)


if __name__ == "__main__":
    main()
