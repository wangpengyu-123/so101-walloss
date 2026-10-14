"""Fix wall-x resume bug: preload wall-oss base weights when resuming from a directory."""
p = "/root/autodl-tmp/wallx_ft/wall-x/wall_x/trainer/fsdp_trainer/fsdp_trainer.py"
s = open(p).read()

old = """        if self._resume_from_single_file():
            self.load_state_dict(
                self.model,
                {"ckpt": self.cfg.checkpoint.resume_from},
            )
"""

new = """        if self._resume_from_single_file():
            self.load_state_dict(
                self.model,
                {"ckpt": self.cfg.checkpoint.resume_from},
            )
        elif self._resume_from_training_checkpoint():
            # FIX (round2): resuming from a training-checkpoint directory skips
            # the single-file wall-oss load above, leaving the action-head base
            # weights (w1/w2/w3/action_proj_back) randomly initialized. The
            # trainable-only checkpoint holds only LoRA + propri_proj, so the
            # pretrained action head base must be loaded from wall-oss explicitly.
            import os
            _walloss_ckpt = os.path.join(
                os.path.dirname(self.cfg.model.config_path), "model.safetensors"
            )
            if os.path.isfile(_walloss_ckpt):
                self.log(
                    "Resuming from directory: preloading wall-oss base from "
                    + _walloss_ckpt
                )
                self.load_state_dict(
                    self.model,
                    {"ckpt": _walloss_ckpt},
                )
"""

assert old in s, "resume pattern not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("fsdp_trainer.py resume fix applied")
