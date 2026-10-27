# SPDX-License-Identifier: Apache-2.0
"""Compare measured stage medians and paired-repeat marginal/cumulative gains."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

METRICS = {"throughput_tok_s": True, "ttft_ms": False, "tpot_ms": False, "p99_ms": False}


def summarize(rows: list[dict], min_repeats: int = 3) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(dict))
    fingerprints = {}
    for row in rows:
        key = row["experiment_id"], row["workload"]
        fingerprint = row["case_fingerprint"]
        if not fingerprint or fingerprints.setdefault(key, fingerprint) != fingerprint:
            raise ValueError(f"Mismatched/empty model, topology, workload or runtime fingerprint: {key}")
        if row["valid"].lower() != "true":
            raise ValueError("Results with failed correctness or feature-hit validation cannot measure gains")
        stage, repeat = int(row["stage"]), row["repeat"]
        if repeat in groups[key][stage]:
            raise ValueError(f"Duplicate stage/repeat: {key}, {stage}, {repeat}")
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{metric} must be positive and finite")
        groups[key][stage][repeat] = row
    result = []
    for (experiment, workload), stages in sorted(groups.items()):
        if sorted(stages) != list(range(max(stages) + 1)):
            raise ValueError("Include baseline and every preceding stage to compute incremental gains")
        baseline = stages[0]
        if len(baseline) < min_repeats or any(set(runs) != set(baseline) for runs in stages.values()):
            raise ValueError("Stages need the same repeat IDs and at least min_repeats measurements")
        for stage, runs in sorted(stages.items()):
            item = {"experiment_id": experiment, "workload": workload, "stage": stage, "metrics": {}}
            for metric, higher_is_better in METRICS.items():

                def gains(reference, runs=runs, metric=metric, higher_is_better=higher_is_better):
                    return median(
                        (
                            (float(runs[r][metric]) / float(reference[r][metric]) - 1)
                            if higher_is_better
                            else (1 - float(runs[r][metric]) / float(reference[r][metric]))
                        )
                        * 100
                        for r in runs
                    )

                item["metrics"][metric] = {
                    "median": median(float(row[metric]) for row in runs.values()),
                    "marginal_gain_percent": None if stage == 0 else gains(stages[stage - 1]),
                    "cumulative_gain_percent": gains(baseline),
                }
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--min-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.min_repeats < 1:
        parser.error("--min-repeats must be positive")
    with args.csv_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("No measured results; the template contains only column headers")
    print(json.dumps(summarize(rows, args.min_repeats), indent=2))


if __name__ == "__main__":
    main()
