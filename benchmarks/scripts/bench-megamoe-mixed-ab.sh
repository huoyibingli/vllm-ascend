#!/usr/bin/env bash
set -euo pipefail

scenario=${1:?Usage: bash bench-megamoe-mixed-ab.sh baseline|megamoe}
case "$scenario" in
  baseline|megamoe) ;;
  *) printf 'Unknown scenario: %s\n' "$scenario" >&2; exit 2 ;;
esac
model_path=${MODEL_PATH:-/data/Qwen3.5-35B-A3B}
case_dir=${LOG_ROOT:-$PWD/perf_runs/megamoe-mixed}/$scenario
curl --fail --silent http://localhost:8899/health >/dev/null
mkdir -p "$case_dir"
git rev-parse HEAD > "$case_dir/commit.txt"
for round in 1 2 3 4 5; do
  result_dir="$case_dir/round_$round"
  mkdir "$result_dir"
  vllm bench serve --backend vllm --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 64 --max-concurrency 32 --ignore-eos \
    --model qwen3.5-35b --tokenizer "$model_path" \
    --endpoint /v1/completions --request-rate inf --seed 1000 \
    --host localhost --port 8899 \
    --save-result --save-detailed --result-dir "$result_dir" \
    --result-filename result.json 2>&1 | tee "$result_dir/bench.log"
done
