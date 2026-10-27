# SPDX-License-Identifier: Apache-2.0
"""Start the current cumulative stage with a recorded, fixed server profile."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def build_command(model: str, profile: dict, port: int) -> list[str]:
    command = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size",
        str(profile["tensor_parallel_size"]),
        "--dtype",
        profile["dtype"],
        "--seed",
        str(profile["seed"]),
        "--max-model-len",
        str(profile["max_model_len"]),
        "--max-num-batched-tokens",
        str(profile["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(profile["max_num_seqs"]),
        "--mm-encoder-tp-mode",
        "weights",
        "--port",
        str(port),
        "--additional-config",
        json.dumps(profile["additional_config"], separators=(",", ":")),
    ]
    if profile["enable_expert_parallel"]:
        command.append("--enable-expert-parallel")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Same local Qwen3.5-35B-A3B checkpoint for every stage")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = json.loads(Path(__file__).with_name("stage.json").read_text())
    command = build_command(args.model, profile, args.port)
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"stage": profile["stage"], "branch": profile["branch"], "profile_sha256": digest}), flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env.update(profile["environment"])
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
