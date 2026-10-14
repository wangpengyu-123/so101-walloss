"""严肃验证修复后的 resume 流程：动作头 base / LoRA / propri_proj 逐位比对。
全部差异=0 才证明模型权重正确恢复（不只 loss 表面正常）。
复现 fsdp_trainer 修复后的加载顺序：create_model → Qwen base → wall-oss(动作头base) → LoRA → checkpoint13。
"""
import os, sys, logging

sys.path.insert(0, "/root/autodl-tmp/wallx_ft/wall-x")
import torch
from safetensors import safe_open
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_resume")

CLOUD = "/root/autodl-tmp/wallx_ft"

from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.trainer.adapters import resolve_adapter
from wall_x.trainer.fsdp_trainer import checkpoint_io

cfg = load_config(f"{CLOUD}/config_full_v2_round2.yml")

# --- normalizers（照抄 merge_v2）---
loaded = load_lerobot_normalizers(cfg)
assert loaded is not None, "normalizers 加载失败"
normalizer_action, normalizer_propri = loaded[0], loaded[1]

# --- adapter + processor（照抄 merge_v2）---
adapter_cls = resolve_adapter(cfg.model_type)
adapter = adapter_cls(cfg=cfg, logger=logger, model_type=cfg.model_type)
adapter.normalizer_action = normalizer_action
adapter.normalizer_propri = normalizer_propri
processors_dict = adapter.load_processor(None)
processor = processors_dict["processor"]
tokenizer_mixin = processors_dict["tokenizer_mixin"]

# --- model + Qwen base（照抄 merge_v2）---
model_config = adapter.build_model_config()
model = adapter.create_model(processor, tokenizer_mixin, model_config)
model = adapter.load_weights(model, normalizer_action, normalizer_propri, processor=processor)
logger.info("Qwen base 加载完成")

# --- 修复的关键步：resume 目录时预加载 wall-oss（动作头 base）---
walloss_path = f"{CLOUD}/weights/wall-oss-0.5/model.safetensors"
checkpoint_io.load_weights(
    model=model,
    resume_config={"ckpt": walloss_path},
    model_class=adapter._checkpoint_model_class() if hasattr(adapter, "_checkpoint_model_class") else None,
    log_fn=logger.info,
)
logger.info("wall-oss 预加载完成")

# ---- 比对 1：动作头 base == wall-oss 预训练权重 ----
w = safe_open(walloss_path, framework="pt")
print("\n=== 比对 1：动作头 base == wall-oss（差异必须 0；若动作头 base 是随机初始化，差异会巨大）===")
checks = [
    ("action_preprocessor.w1.weight", "w1"),
    ("action_preprocessor.w2.weight", "w2"),
    ("action_preprocessor.w3.weight", "w3"),
    ("action_preprocessor.action_proj_back.weight", "action_proj_back"),
]
all_ok = True
for ckpt_key, attr in checks:
    expected = w.get_tensor(ckpt_key).float()
    actual = getattr(model.action_preprocessor, attr).weight.detach().float()
    diff = (expected - actual).abs().max().item()
    ok = "OK" if diff == 0.0 else "FAIL"
    if diff != 0.0:
        all_ok = False
    print(f"  action_preprocessor.{attr}.weight: 逐位最大差异={diff} [{ok}]")

# --- 注入 LoRA（照抄 merge_v2 的 _inject_lora）---
import importlib.util
spec = importlib.util.spec_from_file_location("merge_v2", f"{CLOUD}/merge_v2.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)
m2._inject_lora(model, cfg.model)
logger.info("LoRA 注入完成")

# --- 加载 checkpoint 13（同训练 resume）---
ckpt13_path = f"{CLOUD}/output_full_v2/13/model.safetensors"
ckpt_state = load_file(ckpt13_path)
missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
logger.info(f"checkpoint 13 加载完成（missing {len(missing)} 个冻结 base，预期）")

# ---- 比对 2：LoRA B == checkpoint 13 ----
c13 = safe_open(ckpt13_path, framework="pt")
print("\n=== 比对 2：LoRA == checkpoint 13（差异必须 0）===")
lora_pairs = [
    ("model.model.layers.0.self_attn.qkv_proj_experts.0.lora_B.default.weight",
     "model.model.layers.0.self_attn.qkv_proj_experts.0.lora_B.default.weight"),
    ("action_preprocessor.model.w1.lora_B.default.weight",
     "action_preprocessor.model.w1.lora_B.default.weight"),
]
named = dict(model.named_parameters())
for ckpt_key, model_key in lora_pairs:
    expected = c13.get_tensor(ckpt_key).float()
    actual = named[model_key].detach().float()
    diff = (expected - actual).abs().max().item()
    ok = "OK" if diff == 0.0 else "FAIL"
    if diff != 0.0:
        all_ok = False
    print(f"  {ckpt_key}: 逐位最大差异={diff} [{ok}]")

# ---- 比对 3：propri_proj == checkpoint 13 ----
print("\n=== 比对 3：propri_proj == checkpoint 13（差异必须 0）===")
expected = c13.get_tensor("action_preprocessor.model.propri_proj.weight").float()
actual = model.action_preprocessor.propri_proj.weight.detach().float()
diff = (expected - actual).abs().max().item()
ok = "OK" if diff == 0.0 else "FAIL"
if diff != 0.0:
    all_ok = False
print(f"  propri_proj.weight: 逐位最大差异={diff} [{ok}]")

print("\n" + "=" * 60)
print("总判定:", "全部逐位一致，模型权重正确恢复 ✅" if all_ok else "存在不一致 ❌")
print("=" * 60)
