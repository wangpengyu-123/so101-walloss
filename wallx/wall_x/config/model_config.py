"""Base model config dataclasses.

Optional model variants register their config dataclasses from their own
packages via ``wall_x.config.registry.register_model_config``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .registry import register_model_config


@dataclass
class ModelConfig:
    """Base model architecture fields."""

    use_ema: bool = False
    attn_implementation: Optional[str] = None
    attn_deterministic: Optional[bool] = None
    ar_loss_weight: float = 1.0


@register_model_config("qwen2_5")
@dataclass
class QActModelConfig(ModelConfig):
    """QAct model config for the Qwen2.5 VLA path."""

    config_path: str = ""
    processor_path: str = ""
    pretrained_path: Optional[str] = None
    backbone: str = "qwen2_5"
    action_tokenizer_type: Optional[str] = None
    action_tokenizer_path: Optional[str] = None
    action_tokenizer_checkpoint_path: Optional[str] = None
    action_tokenizer_config_dir: Optional[str] = None
    new_special_tokens: Optional[List[str]] = None
    flow_loss_weight: float = 1.0
    # LoRA fine-tuning switches. Injection is wired in
    # FSDPTrainer._inject_lora_if_needed and happens AFTER all pretrained
    # weights are loaded (Qwen2.5-VL base + wall-oss-0.5 single-file ACT
    # weights), so base weights load cleanly and adapters are added
    # zero-init afterwards.
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: Optional[List[str]] = None  # None -> ["qkv_proj"] (fused QKV)
    lora_dropout: float = 0.05
    # Plan A (round 2): also LoRA the action head (ActionProcessor) Linear
    # layers that have pretrained weights (w1/w2/w3/action_proj_back), so the
    # flow-matching head update survives merge. propri_proj is random-init
    # (created only when use_state_string_representation=false) and has no
    # pretrained weight to LoRA onto, so it trains full-param instead.
    lora_action_head_target_modules: Optional[List[str]] = None  # None -> ["w1","w2","w3","action_proj_back"]
    full_param_train_keywords: Optional[List[str]] = None  # None -> ["propri_proj"]
    enable_customized_robot_config: bool = False
    customized_robot_config: Optional[Dict[str, Any]] = None


__all__ = ["ModelConfig", "QActModelConfig"]
