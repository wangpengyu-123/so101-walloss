#!/usr/bin/env python
"""merge_v2.py — Plan A merge for wall-oss SO-ARM101 LoRA fine-tune.

Round 1 used ``PeftModel.from_pretrained(base, adapter).merge_and_unload()``.
That only merges the LoRA *adapter* weights and silently drops every
full-parameter update that was trained alongside the adapters — most notably
the action head (ActionProcessor: w1/w2/w3/action_proj_back) and the
random-init state projection ``propri_proj``.

Plan A (round 2) fixes this at both ends:

  * training (see patches/): the action head is LoRA'd too, ``propri_proj``
    stays full-param trainable, and the checkpoint now saves *all* trainable
    weights (``requires_grad`` based) instead of only keys containing
    "lora_";
  * this script: rebuilds the base model exactly like training does, injects
    the same LoRA wrappers, loads the training checkpoint (LoRA adapters +
    full-param weights), then ``merge_and_unload``s each LoRA wrapper and
    saves the complete model.

Usage (run inside the wallx conda/venv that has wall_x + peft + torch):

    python merge_v2.py \
        --train-config config_local_v2.yml \
        --checkpoint /path/to/output_full_v2/5000 \
        --output /path/to/wall_oss_so101_lora_v2 \
        [--wallx-src /home/ubuntu22/so101_walloss/wall-x] \
        [--base-model /path/to/Qwen2.5-VL-3B-Instruct] \
        [--processor-path /path/to/Qwen2.5-VL-3B-Instruct] \
        [--config-path /path/to/wall-oss-0.5/config.json] \
        [--resume-from /path/to/wall-oss-0.5/model.safetensors] \
        [--norm-stats /path/to/norm_stats.json]

``--checkpoint`` may be the training checkpoint directory (containing
``model.safetensors``) or the file itself.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("merge_v2")


def _resolve_checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        p = os.path.join(path, "model.safetensors")
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"no model.safetensors under {path}")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"checkpoint not found: {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan A wall-oss LoRA merge")
    p.add_argument("--train-config", required=True, help="wall-x train config yaml")
    p.add_argument("--checkpoint", required=True, help="training ckpt dir or model.safetensors")
    p.add_argument("--output", required=True, help="output dir for the merged model")
    p.add_argument(
        "--wallx-src",
        default="/home/ubuntu22/so101_walloss/wall-x",
        help="patched wall-x source directory (added to sys.path[0])",
    )
    p.add_argument("--base-model", default=None, help="override model.pretrained_path (Qwen2.5-VL-3B)")
    p.add_argument("--processor-path", default=None, help="override model.processor_path")
    p.add_argument("--config-path", default=None, help="override model.config_path (wall-oss-0.5 config.json)")
    p.add_argument("--resume-from", default=None, help="override checkpoint.resume_from (wall-oss-0.5 model.safetensors)")
    p.add_argument("--norm-stats", default=None, help="override data.norm_stats_path")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="merged output dtype")
    return p.parse_args()


def _inject_lora(model, cfg_model) -> None:
    """Replicate FSDPTrainer._inject_lora_if_needed (without the FSDP/log bits)."""
    use_lora = getattr(cfg_model, "use_lora", False)
    if not use_lora:
        return

    if cfg_model.lora_target_modules:
        targets = list(cfg_model.lora_target_modules)
    else:
        targets = []
        for n, mod in model.model.named_modules():
            if n.endswith(".qkv_proj_experts"):
                targets.extend(f"{n}.{i}" for i in range(len(mod)))
        targets = targets or ["qkv_proj"]

    action_head_targets = getattr(cfg_model, "lora_action_head_target_modules", None)
    if action_head_targets is None:
        action_head_targets = ["w1", "w2", "w3", "action_proj_back"]
    full_param_keywords = getattr(cfg_model, "full_param_train_keywords", None)
    if full_param_keywords is None:
        full_param_keywords = ["propri_proj"]

    logger.info(
        "Injecting LoRA: r=%s alpha=%s targets=%s action_head=%s full_param=%s dropout=%s",
        cfg_model.lora_r,
        cfg_model.lora_alpha,
        targets,
        action_head_targets,
        full_param_keywords,
        cfg_model.lora_dropout,
    )
    model.add_lora(
        r=cfg_model.lora_r,
        lora_alpha=cfg_model.lora_alpha,
        target_modules=targets,
        lora_dropout=cfg_model.lora_dropout,
        action_head_target_modules=action_head_targets,
        full_param_train_keywords=full_param_keywords,
    )
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
        logger.info("Vision tower frozen")


def main() -> None:
    args = _parse_args()

    # Import the patched wall_x source, not whatever happens to be installed.
    if args.wallx_src and args.wallx_src not in sys.path:
        sys.path.insert(0, args.wallx_src)
        logger.info("using wall-x source: %s", args.wallx_src)

    from wall_x.config.loader import load_config
    from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
    from wall_x.trainer.adapters import resolve_adapter
    from wall_x.trainer.fsdp_trainer import checkpoint_io

    cfg = load_config(args.train_config)

    # Path overrides for local runs.
    if args.base_model:
        cfg.model.pretrained_path = args.base_model
    if args.processor_path:
        cfg.model.processor_path = args.processor_path
    if args.config_path:
        cfg.model.config_path = args.config_path
    if args.resume_from:
        cfg.checkpoint.resume_from = args.resume_from
    if args.norm_stats:
        cfg.data.norm_stats_path = args.norm_stats

    ckpt_file = _resolve_checkpoint_file(args.checkpoint)
    os.makedirs(args.output, exist_ok=True)

    # --- normalizers (fp32, saved separately; mirrors training) ---
    loaded = load_lerobot_normalizers(cfg)
    if loaded is None:
        raise RuntimeError("could not load normalizers; set data.norm_stats_path")
    normalizer_action, normalizer_propri = loaded[0], loaded[1]

    # --- adapter + processor ---
    adapter_cls = resolve_adapter(cfg.model_type)
    adapter = adapter_cls(cfg=cfg, logger=logger, model_type=cfg.model_type)
    adapter.normalizer_action = normalizer_action
    adapter.normalizer_propri = normalizer_propri
    processors_dict = adapter.load_processor(None)
    processor = processors_dict["processor"]
    tokenizer_mixin = processors_dict["tokenizer_mixin"]

    # --- model + pretrained weights (Qwen backbone -> resize -> normalizers) ---
    model_config = adapter.build_model_config()
    model = adapter.create_model(processor, tokenizer_mixin, model_config)
    model = adapter.load_weights(
        model, normalizer_action, normalizer_propri, processor=processor
    )
    logger.info("loaded Qwen2.5-VL backbone + resized embeddings + normalizers")

    # --- wall-oss-0.5 ACT head (single-file, fixed path) ---
    # NOTE: must NOT use cfg.checkpoint.resume_from — after a training resume
    # that field points at a checkpoint DIRECTORY, which breaks the single-file
    # load here. Derive wall-oss from model.config_path's sibling instead.
    model_class = type(adapter).model_class()
    _walloss_ckpt = os.path.join(
        os.path.dirname(cfg.model.config_path), "model.safetensors"
    )
    checkpoint_io.load_weights(
        model=model,
        resume_config={"ckpt": _walloss_ckpt},
        model_class=model_class,
        log_fn=logger.info,
    )
    logger.info("loaded wall-oss-0.5 ACT head from %s", _walloss_ckpt)

    # Snapshot the random-init propri_proj BEFORE the training checkpoint load
    # so we can assert the checkpoint actually overwrote it.
    prop_before = model.action_preprocessor.propri_proj.weight.detach().clone()

    # --- inject LoRA (backbone + action head) ---
    _inject_lora(model, cfg.model)

    # --- load training checkpoint (LoRA adapters + full-param trainable) ---
    ckpt_state = load_file(ckpt_file)
    lora_keys = [k for k in ckpt_state if "lora_" in k]
    full_param_keys = [k for k in ckpt_state if "lora_" not in k]
    logger.info(
        "checkpoint contains %d LoRA keys + %d full-param keys",
        len(lora_keys),
        len(full_param_keys),
    )
    if not lora_keys:
        raise RuntimeError("checkpoint has no lora_* keys; is this a LoRA run?")
    if not any("propri_proj" in k for k in full_param_keys):
        raise RuntimeError(
            "checkpoint has no propri_proj full-param weight. This is the round-1 "
            "bug (trainable_only saved only lora_*). Re-train with the patched "
            "checkpoint_io.py before merging."
        )

    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        # Expected: frozen base weights are absent from a trainable-only ckpt.
        logger.info("state_dict missing keys (frozen base, expected): %d", len(missing))
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:10]}")

    prop_after = model.action_preprocessor.propri_proj.weight.detach().clone()
    if torch.equal(prop_before, prop_after):
        logger.warning(
            "propri_proj did not change after checkpoint load; the checkpoint "
            "may not contain a meaningful update."
        )
    else:
        logger.info("propri_proj full-param update loaded and applied")

    # --- merge LoRA wrappers back into the base weights ---
    model.model = model.model.merge_and_unload()
    model.action_preprocessor = model.action_preprocessor.merge_and_unload()
    logger.info("merged backbone + action-head LoRA adapters")

    # --- save normalizers (fp32) then cast the model for the final file ---
    torch.save(normalizer_action.state_dict(), os.path.join(args.output, "normalizer_action.pth"))
    torch.save(normalizer_propri.state_dict(), os.path.join(args.output, "normalizer_propri.pth"))

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = model.to(dtype)

    state = model.state_dict()
    assert not any("lora_" in k for k in state), "lora_* keys leaked into merged model"
    # Unshare tied tensors (e.g. lm_head.weight == model.embed_tokens.weight):
    # safetensors rejects tensors that share memory.
    _seen_ptr = {}
    for _k in list(state.keys()):
        _ptr = state[_k].data_ptr()
        if _ptr in _seen_ptr:
            state[_k] = state[_k].clone()
            logger.info("unshared tied tensor %s (shared with %s)", _k, _seen_ptr[_ptr])
        else:
            _seen_ptr[_ptr] = _k
    save_file(state, os.path.join(args.output, "model.safetensors"))
    logger.info("saved merged model.safetensors (%d tensors, %s)", len(state), args.dtype)

    # --- config + processor artifacts (mirrors checkpoint_io metadata copy) ---
    if os.path.isfile(cfg.model.config_path):
        shutil.copy(cfg.model.config_path, os.path.join(args.output, "config.json"))
    if os.path.isfile(cfg.data.norm_stats_path):
        shutil.copy(cfg.data.norm_stats_path, os.path.join(args.output, "norm_stats.json"))
    processor_dir = cfg.model.processor_path
    if processor_dir and os.path.isdir(processor_dir):
        for fn in (
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            src = os.path.join(processor_dir, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.output, fn))

    # --- adapter backup (trainable weights, for reference/re-merge) ---
    adapter_dir = os.path.join(args.output, "lora_adapter_backup")
    os.makedirs(adapter_dir, exist_ok=True)
    save_file(ckpt_state, os.path.join(adapter_dir, "adapter_model.safetensors"))
    import json

    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": cfg.model.lora_r,
                "lora_alpha": cfg.model.lora_alpha,
                "lora_dropout": cfg.model.lora_dropout,
                "target_modules": list(cfg.model.lora_target_modules or []),
                "bias": "none",
                "base_model_name_or_path": "X-Square-Robot/wall-oss-0.5",
            },
            f,
            indent=2,
        )

    logger.info("merge complete -> %s", args.output)


if __name__ == "__main__":
    main()
