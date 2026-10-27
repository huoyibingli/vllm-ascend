# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from benchmarks.qwen35_incremental.compare_results import summarize


def make_results():
    return [
        {
            "experiment_id": "fixed",
            "workload": "image",
            "case_fingerprint": "same-model-runtime-topology",
            "stage": str(stage),
            "repeat": str(repeat),
            "git_sha": str(stage),
            "valid": "true",
            "throughput_tok_s": str(throughput),
            "ttft_ms": str(latency),
            "tpot_ms": str(latency),
            "p99_ms": str(latency),
        }
        for stage, throughput, latency in [(0, 100, 10), (1, 125, 8), (2, 150, 10)]
        for repeat in range(3)
    ]


def test_paired_incremental_and_cumulative_gains():
    result = summarize(make_results())
    assert result[1]["metrics"]["throughput_tok_s"]["marginal_gain_percent"] == pytest.approx(25)
    assert result[2]["metrics"]["throughput_tok_s"]["marginal_gain_percent"] == pytest.approx(20)
    assert result[2]["metrics"]["throughput_tok_s"]["cumulative_gain_percent"] == pytest.approx(50)
    assert result[1]["metrics"]["tpot_ms"]["marginal_gain_percent"] == pytest.approx(20)
    assert result[2]["metrics"]["tpot_ms"]["marginal_gain_percent"] == pytest.approx(-25)
    assert result[2]["metrics"]["tpot_ms"]["cumulative_gain_percent"] == pytest.approx(0)


@pytest.mark.parametrize("invalid", ["fingerprint", "correctness", "unpaired", "missing_stage", "duplicate", "nan"])
def test_invalid_experiments_are_rejected(invalid):
    rows = copy.deepcopy(make_results())
    if invalid == "fingerprint":
        rows[-1]["case_fingerprint"] = "different-topology"
    elif invalid == "correctness":
        rows[-1]["valid"] = "false"
    elif invalid == "unpaired":
        rows.pop()
    elif invalid == "missing_stage":
        rows = [row for row in rows if row["stage"] != "1"]
    elif invalid == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1]["tpot_ms"] = "nan"
    with pytest.raises(ValueError):
        summarize(rows)
