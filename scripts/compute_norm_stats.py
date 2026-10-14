#!/usr/bin/env python3
"""Compute LeRobot normalization stats (mean/std/q01/q99) for SO-ARM101.

Self-contained drop-in for wall-x's scripts/compute_norm_stats.py, scoped to the
SO-ARM101 joint-space dataset. Reads only the `observation.state` and `action`
parquet columns (no video decode), so it works on any local LeRobot v3 dataset
layout (data/chunk-*/file-*.parquet) regardless of metadata version.

Output is written in the exact format wall-x's loader expects:

    {"norm_stats": {
        "observation.state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
        "action":            {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
    }}

The vectors are the REAL 6-dim joint vector (shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper). wall-x's normalizer tail-pads the
6-dim stats to the model's 26-dim action/state space (min=0, delta=1 on the pad),
driven by `action_padding: 20` in dof_config / agent_pos_config.

Usage:
    python compute_norm_stats.py \
        --data_root /path/to/so101_table_pick_place/20260926_195903 \
        --output_path /path/to/norm_stats.json
"""

import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq


def load_vectors(data_root):
    # Match both v3.0 layout (data/chunk-*/file-*.parquet) and v2.1 layout
    # (data/chunk-*/episode_*.parquet) so the script works on converted data too.
    paths = sorted(
        glob.glob(f"{data_root}/data/chunk-*/file-*.parquet")
        + glob.glob(f"{data_root}/data/chunk-*/episode_*.parquet")
    )
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_root}/data")
    state_parts, action_parts = [], []
    for p in paths:
        t = pq.read_table(p, columns=["observation.state", "action"]).to_pandas()
        state_parts.append(np.stack(t["observation.state"].to_numpy()))
        action_parts.append(np.stack(t["action"].to_numpy()))
    state = np.concatenate(state_parts, axis=0).astype(np.float64)
    action = np.concatenate(action_parts, axis=0).astype(np.float64)
    return state, action


def vector_stats(x):
    return {
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    state, action = load_vectors(args.data_root)
    print(f"frames={state.shape[0]} state_dim={state.shape[1]} action_dim={action.shape[1]}")
    assert state.shape[1] == 6 and action.shape[1] == 6, "expected 6-dim joint vector"

    norm_stats = {
        "observation.state": vector_stats(state),
        "action": vector_stats(action),
    }
    out = {"norm_stats": norm_stats}
    with open(args.output_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {args.output_path}")
    for k, s in norm_stats.items():
        print(f"  {k}: q01={[round(v,4) for v in s['q01']]}")
        print(f"  {k}: q99={[round(v,4) for v in s['q99']]}")


if __name__ == "__main__":
    main()
