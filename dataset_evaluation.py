"""Shared video-level pooling, bootstrap intervals and paired randomization.

Frames are pooled by each task first. Independent units here are videos; no
frame-level independence assumption is made. Requires only NumPy.
"""

import csv
import itertools
import json
from pathlib import Path

import numpy as np


def add_statistics_arguments(parser):
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--statistics-seed", type=int, default=42)


def validate_statistics(samples, confidence):
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Bootstrap samples must be positive and confidence must be between 0 and 1")


def finite(value):
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def mean_statistics(values, total, samples=2000, confidence=0.95, seed=42):
    validate_statistics(samples, confidence)
    values = np.asarray([v for v in values if finite(v)], dtype=np.float64)
    n = len(values)
    result = dict(n_videos=n, n_missing=total - n, mean=float(values.mean()) if n else None,
                  std=float(values.std(ddof=1)) if n > 1 else None, ci_low=None, ci_high=None)
    if n > 1:
        rng = np.random.default_rng(seed)
        # Bound memory for large video cohorts.
        means = np.concatenate([values[rng.integers(n, size=(min(256, samples - i), n))].mean(axis=1)
                                for i in range(0, samples, 256)])
        result["ci_low"], result["ci_high"] = map(float, np.quantile(means, [(1-confidence)/2, (1+confidence)/2]))
    return result


def paired_pvalue(differences, seed=42, draws=10000):
    """Two-sided sign-flip randomization test of mean paired difference = 0."""
    d = np.asarray(differences, dtype=np.float64)
    if len(d) < 2:
        return None
    observed = abs(d.mean())
    tolerance = 1e-12 * max(1.0, observed)
    if len(d) <= 16:
        count = sum(abs(np.dot(signs, d) / len(d)) >= observed - tolerance
                    for signs in itertools.product((-1, 1), repeat=len(d)))
        return count / (2 ** len(d))
    rng = np.random.default_rng(seed)
    count = 0
    for i in range(0, draws, 256):
        null = (rng.choice([-1, 1], size=(min(256, draws-i), len(d))) * d).mean(axis=1)
        count += int((np.abs(null) >= observed - tolerance).sum())
    return (count + 1) / (draws + 1)


def aggregate_records(records, group_fields, metrics, *, video_field="video", samples=2000,
                      confidence=0.95, seed=42, expected_videos=None):
    """One value per video/group/metric; reject duplicate units instead of double counting."""
    groups = {}
    cohort = set(expected_videos or [r[video_field] for r in records])
    for row in records:
        key = tuple(row[k] for k in group_fields)
        group = groups.setdefault(key, {})
        video = row[video_field]
        if video not in cohort:
            raise ValueError(f"Video outside the requested evaluation cohort: {video}")
        if video in group:
            raise ValueError(f"Duplicate video/group result: {video}, {key}")
        group[video] = row
    output = []
    for key, rows in sorted(groups.items()):
        for metric in metrics:
            output.append(dict(zip(group_fields, key), metric=metric,
                **mean_statistics([r.get(metric) for _, r in sorted(rows.items())], len(cohort), samples, confidence, seed)))
    return output


def paired_comparisons(records, metrics, *, group_fields=(), video_field="video", samples=2000,
                       confidence=0.95, seed=42, expected_videos=None):
    """Pair methods on identical videos and conditions. Positive delta = test minus anchor."""
    groups = {}
    cohort = set(expected_videos or [r[video_field] for r in records])
    for row in records:
        key = tuple(row[k] for k in group_fields)
        method = groups.setdefault(key, {}).setdefault(row["method"], {})
        if row[video_field] in method:
            raise ValueError("Duplicate paired video/method/condition")
        method[row[video_field]] = row
    output = []
    for key, methods in sorted(groups.items()):
        for anchor, test in itertools.combinations(sorted(methods), 2):
            common = sorted(set(methods[anchor]) & set(methods[test]))
            for metric in metrics:
                differences = [methods[test][v][metric] - methods[anchor][v][metric] for v in common
                               if finite(methods[test][v].get(metric)) and finite(methods[anchor][v].get(metric))]
                output.append(dict(zip(group_fields, key), anchor_method=anchor, test_method=test, metric=metric,
                    **mean_statistics(differences, len(cohort), samples, confidence, seed),
                    p_two_sided=paired_pvalue(differences, seed)))
    # Holm family-wise adjustment over all available tests in this export.
    ordered = sorted((r for r in output if r["p_two_sided"] is not None), key=lambda r: r["p_two_sided"])
    adjusted = 0.0
    for i, row in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(ordered)-i) * row["p_two_sided"]))
        row["p_holm"] = adjusted
    for row in output:
        row.setdefault("p_holm", None)
    return output


def statistics_protocol(samples=2000, confidence=0.95, seed=42):
    validate_statistics(samples, confidence)
    return dict(unit="video", pooling="equal weight per video after within-video pooling",
                uncertainty="percentile bootstrap of videos; undefined with fewer than two finite videos",
                bootstrap_samples=samples, confidence=confidence, seed=seed,
                missing="null/non-finite scores excluded with explicit counts; no imputation",
                paired="test minus anchor on common finite videos; two-sided sign-flip test, exact for n<=16 else 10000 seeded draws",
                multiplicity="Holm adjusted across all available paired tests in this export",
                caution="Videos must be independent sampling units. Small cohorts give unstable intervals. Equal-CRF deltas do not imply equal-rate gains; use per-video BD comparisons.")


def write_dataset_report(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for key in ("aggregate", "paired", "bdrate_aggregate", "per_video"):
        rows = payload.get(key, [])
        if not rows:
            continue
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / f"dataset_{key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Dataset evaluation", "", f"Videos requested: {len(payload['videos'])}.", "",
             "Videos receive equal weight after within-video pooling. Intervals resample videos, not frames.",
             "Missing/non-finite scores are excluded with explicit counts. With fewer than two videos, uncertainty is unavailable.",
             "Paired deltas are test minus anchor on common finite videos; p-values use two-sided sign flips and Holm correction.", ""]
    protocol = payload["protocol"]
    lines += [f"Confidence level: {100 * protocol['confidence']:g}%. Bootstrap samples: {protocol['bootstrap_samples']}. Seed: {protocol['seed']}.", ""]
    if "held_out_verified" in payload:
        lines += [f"Held-out provenance verified for every video: {payload['held_out_verified']}.", ""]
    failures = [row for row in payload.get("coverage", []) if row["status"] != "scored"]
    if failures:
        lines += [f"Incomplete coverage: {len(failures)} failed/skipped entries; see coverage in dataset_results.json.", ""]
    for key in ("aggregate", "bdrate_aggregate", "paired"):
        rows = payload.get(key, [])
        if not rows:
            continue
        identifiers = [k for k in rows[0] if k not in {"n_videos", "n_missing", "mean", "std", "ci_low", "ci_high", "p_two_sided", "p_holm"}]
        columns = identifiers + ["n_videos", "n_missing", "mean", "ci_low", "ci_high"]
        if key == "paired":
            columns += ["p_two_sided", "p_holm"]
        lines += [f"## {key.replace('_', ' ').title()}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in rows:
            lines.append("| " + " | ".join("N/A" if row.get(k) is None else f"{row[k]:.6g}" if isinstance(row[k], float)
                                             else str(row[k]).replace("|", "\\|") for k in columns) + " |")
        lines.append("")
    lines += [payload["protocol"]["caution"], "", "Full protocol, coverage and provenance: dataset_results.json."]
    (directory / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
