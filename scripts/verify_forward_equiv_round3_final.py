import os, sys, logging, gc, importlib.util
sys.path.insert(0, '/root/autodl-tmp/wallx_ft/wall-x')
import torch
from safetensors.torch import load_file
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('fwd_equiv')

CLOUD = '/root/autodl-tmp/wallx_ft'
from wall_x.config.loader import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.trainer.adapters import resolve_adapter
from wall_x.trainer.fsdp_trainer import checkpoint_io
from wall_x.trainer.utils import move_batch_to_device

import torch.distributed as dist
if not dist.is_available() or not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='env://')

# mirror the trainer: seed BEFORE data loading
from wall_x._vendor.harrix.utils.seed import set_seed_everywhere
set_seed_everywhere(10222)
logger.info('seeded 10222 (same as trainer)')

spec = importlib.util.spec_from_file_location('merge_v2', f'{CLOUD}/merge_v2.py')
m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)

cfg = load_config(f'{CLOUD}/config_full_v2_round3.yml')

# ---- adapter/processor/normalizers (no model yet) ----
loaded = load_lerobot_normalizers(cfg)
assert loaded is not None
na, npr = loaded[0], loaded[1]
adapter_cls = resolve_adapter(cfg.model_type)
adapter = adapter_cls(cfg=cfg, logger=logger, model_type=cfg.model_type)
adapter.normalizer_action = na; adapter.normalizer_propri = npr
pd = adapter.load_processor(None)
processor, tkm = pd['processor'], pd['tokenizer_mixin']
mc = adapter.build_model_config()

# ---- one REAL batch from the exact trainer data path ----
dataset, train_loader, train_iters = adapter.load_dataset(
    cfg.data, processor, 0, 1,
    model_config=mc, tokenizer_mixin=tkm,
    normalizer_action=na, normalizer_propri=npr)
batch = None
for attempt in range(20):
    b = next(iter(train_loader))
    g = b['image_grid_thw']
    pv = b['pixel_values']
    toks = (g[:, 0] * g[:, 1] * g[:, 2]).tolist()
    logger.info(f'pull {attempt}: grid={tuple(g.shape)} tokens/row={toks} pixel_values={tuple(pv.shape)}')
    if pv.numel() > 0 and all(t > 0 for t in toks):
        batch = b
        break
assert batch is not None, 'no healthy batch in 20 pulls'
# BatchFeature is a UserDict; move_batch_to_device only recurses dict/list, so
# move each value explicitly (the trainer's DDP pre-hook does this at forward).
batch = {k: move_batch_to_device(v, torch.device('cuda')) for k, v in batch.items()}
logger.info('real batch loaded and moved to cuda')

def collect_outputs(out):
    res = {}
    for f in dir(out):
        if f.startswith('_'): continue
        v = getattr(out, f)
        if isinstance(v, torch.Tensor):
            res[f] = v.detach().float().cpu()
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    res[f'{f}.{k2}'] = v2.detach().float().cpu()
    return res

def build_base():
    m = adapter.create_model(processor, tkm, mc)
    m = adapter.load_weights(m, na, npr, processor=processor)
    return m

# clone the batch for B so any in-place mutation by A's forward cannot leak
batch_for_b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
batch_for_a = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
del batch; gc.collect(); torch.cuda.empty_cache()

# ---- model A: base(wall-oss) + LoRA + ckpt (merge_v2 recipe) ----
model_a = build_base()
checkpoint_io.load_weights(model=model_a,
    resume_config={'ckpt': f'{CLOUD}/weights/wall-oss-0.5/model.safetensors'},
    model_class=type(adapter).model_class(), log_fn=logger.info)
m2._inject_lora(model_a, cfg.model)
ckpt_state = load_file(f'{CLOUD}/output_full_v3/53_30000/model.safetensors')
missing_a, unexp_a = model_a.load_state_dict(ckpt_state, strict=False)
assert not unexp_a, unexp_a
logger.info(f'model A ready (LoRA+ckpt), missing {len(missing_a)} frozen-base keys expected')
# mirror trainer DDP placement: bf16 params on cuda
model_a.convert_to_mix_precision()
model_a = model_a.to(torch.device('cuda'))
model_a.eval()
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
with torch.autocast('cuda', dtype=torch.bfloat16):
    out_a = model_a(**batch_for_a, mode='train')
torch.cuda.synchronize()
# determinism check: second forward on a fresh clone
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
batch_for_a2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch_for_b.items()}
with torch.autocast('cuda', dtype=torch.bfloat16):
    out_a2 = model_a(**batch_for_a2, mode='train')
torch.cuda.synchronize()
d_aa = (out_a.logits.float() - out_a2.logits.float()).abs().max().item()
print(f'A-vs-A determinism: logits max diff = {d_aa:.3e}, loss {out_a.loss.item():.8f} vs {out_a2.loss.item():.8f}')
fields_a = collect_outputs(out_a)
loss_a = out_a.loss.item()
logger.info(f'model A forward done, loss={loss_a}')
del model_a, out_a, out_a2, batch_for_a2, ckpt_state
gc.collect(); torch.cuda.empty_cache()

# ---- model B: same build + merged file ----
model_b = build_base()
merged_state = load_file(f'{CLOUD}/merged_round3_final/model.safetensors')
checkpoint_io.load_weights(model=model_b,
    resume_config={'ckpt': f'{CLOUD}/merged_round3_final/model.safetensors'},
    model_class=type(adapter).model_class(), log_fn=logger.info)
missing_b, unexp_b = model_b.load_state_dict(merged_state, strict=False)
assert not unexp_b, f'unexpected: {list(unexp_b)[:5]}'
# the only expected misses: 224 normalizer entries (checkpoint_io keeps
# current-run normalizers, which hold identical values)
missing_b = [k for k in missing_b if 'normalizer' not in k]
assert not missing_b, f'missing: {list(missing_b)[:5]}'
for nk in ('action_preprocessor.normalizer_action.min.so101_table_pick_place',
           'action_preprocessor.normalizer_propri.delta.so101_table_pick_place'):
    sv = model_b.state_dict()[nk]
    cv = merged_state[nk]
    assert torch.equal(sv, cv), f'normalizer mismatch: {nk}'
logger.info('model B ready (merged weights, no LoRA)')
model_b.convert_to_mix_precision()
model_b = model_b.to(torch.device('cuda'))
model_b.eval()
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
with torch.autocast('cuda', dtype=torch.bfloat16):
    out_b = model_b(**batch_for_b, mode='train')
torch.cuda.synchronize()
fields_b = collect_outputs(out_b)
loss_b = out_b.loss.item()
logger.info(f'model B forward done, loss={loss_b}')

print()
print('=== LAYER 5: forward equivalence (base+LoRA vs merged) ===')
print(f'loss A = {loss_a:.8f}')
print(f'loss B = {loss_b:.8f}')
print(f'loss abs diff = {abs(loss_a-loss_b):.6e}')
print(f'output fields compared: {sorted(fields_a.keys())}')
worst = ('', 0.0, '')
n_exact = 0
for f, va in fields_a.items():
    vb = fields_b.get(f)
    if vb is None: continue
    d = (va - vb).abs().max().item()
    rel = d / max(va.abs().max().item(), 1e-12)
    if d == 0.0: n_exact += 1
    if d > worst[1]: worst = (f, d, f'rel={rel:.2e}')
    print(f'  {f}: maxabs={d:.6e} rel={rel:.2e} shape={tuple(va.shape)}')
print(f'fields exactly equal: {n_exact}/{len(fields_a)}')

# diagnostics: where do the logits differ?
la = fields_a['logits']
lb = fields_b['logits']
d = (la - lb).abs()
big = (d > 0.5)
print(f'logits: positions with |diff|>0.5: {big.sum().item()} / {big.numel()}')
idx = d.flatten().argsort(descending=True)[:10]
for i in idx:
    flat = i.item()
    b, s, v = flat // (la.shape[1]*la.shape[2]), (flat // la.shape[2]) % la.shape[1], flat % la.shape[2]
    print(f'  top diff pos: batch={b} seq={s} vocab={v} | A={la[b,s,v].item():.4f} B={lb[b,s,v].item():.4f} diff={d[b,s,v].item():.4f}')
# how many positions per batch have large diffs
for b in range(la.shape[0]):
    print(f'  batch {b}: big-diff positions: {big[b].sum().item()}')
print(f'  logits median |diff|: {d.median().item():.3e} | mean |diff|: {d.mean().item():.3e}')
print(f'WORST field: {worst[0]} maxabs={worst[1]:.6e} {worst[2]}')
print('VERDICT:', 'PASS (near-bitwise/equivalent)' if worst[1] < 1e-3 else 'FAIL')
dist.destroy_process_group()
