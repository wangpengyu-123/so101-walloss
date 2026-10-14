import asyncio, base64, io, json
import numpy as np
import websockets, msgpack
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

INSTRUCTION = "Grab the red block and place it on the target"

JOINT_NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']

def img_to_b64(t):
    a = (t.numpy() * 255).astype(np.uint8)
    a = np.transpose(a, (1, 2, 0))
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format='JPEG', quality=95)
    return base64.b64encode(buf.getvalue()).decode()

def decode_numpy_dict(d):
    buf = d[b'data'] if b'data' in d else d['data']
    shape = tuple(int(x) for x in (d[b'shape'] if b'shape' in d else d['shape']))
    dt = d[b'type'] if b'type' in d else d['type']
    if isinstance(dt, bytes):
        dt = dt.decode()
    return np.frombuffer(buf, dtype=np.dtype(dt)).reshape(shape)

async def main():
    # delta = action q99 - q01 from round3 norm stats (raw physical units)
    ns = json.load(open('/root/autodl-tmp/wallx_ft/data/norm_stats_round4.json'))
    aq = ns['norm_stats']['action']
    q01 = np.asarray(aq['q01'], dtype=np.float64)[:6]
    q99 = np.asarray(aq['q99'], dtype=np.float64)[:6]
    delta = q99 - q01
    print('action delta (q99-q01) per joint:', np.round(delta, 3))

    ds = LeRobotDataset(repo_id='so101_table_pick_place', root='/root/autodl-tmp/wallx_ft/data/so101_real_round4')
    N = 60
    total = len(ds)
    indices = np.linspace(0, total - 1, N).astype(int)

    async with websockets.connect('ws://localhost:32195', open_timeout=30) as ws:
        meta = msgpack.unpackb(await ws.recv(), raw=False)
        print('serving meta:', str(meta)[:120])

        preds = []
        targets = []
        l1s = []
        per_joint = np.zeros(6)
        for i, idx in enumerate(indices):
            s = ds[int(idx)]
            state = s['observation.state'].numpy()[:6]
            action = s['action'].numpy()[:6]
            g = img_to_b64(s['observation.images.global'])
            w = img_to_b64(s['observation.images.wrist'])
            req = {
                'state': {'so101_joint_pos': state.tolist()},
                'views': {'global_view': g, 'wrist_view': w},
                'instruction': INSTRUCTION,
            }
            await ws.send(msgpack.packb(req))
            r = await ws.recv()
            if isinstance(r, str):
                print(f'帧 {i} 服务端报错:', r[:300])
                continue
            resp = msgpack.unpackb(r, raw=False)
            if 'predict_action' not in resp:
                print(f'帧 {i} 无 predict_action:', str(resp)[:200])
                continue
            pa = resp['predict_action']
            if isinstance(pa, dict):
                pa = decode_numpy_dict(pa)
            pa = np.asarray(pa, dtype=np.float64)
            if pa.ndim == 3:
                pa = pa[0]
            pred = pa[0, :6]
            diff = np.abs(pred - action)
            l1s.append(diff.mean())
            per_joint += diff
            preds.append(pred)
            targets.append(action)
            if (i + 1) % 20 == 0:
                print(f'  进度 {i+1}/{N}, 最近 L1={diff.mean():.3f}')

        if l1s:
            preds = np.array(preds)
            targets = np.array(targets)
            n = len(l1s)
            per_joint_l1 = per_joint / n
            bias = (preds - targets).mean(axis=0)
            pred_mean = preds.mean(axis=0)
            tgt_mean = targets.mean(axis=0)
            norm_l1 = per_joint_l1 / delta  # fraction of full range
            print('=' * 70)
            print(f'样本数: {n}')
            print(f'mean L1 (deg): {np.mean(l1s):.4f}')
            print()
            print(f'{"joint":<15} {"L1(deg)":>9} {"L1/delta":>9} {"bias(deg)":>10} {"pred_mean":>10} {"tgt_mean":>10}')
            for j in range(6):
                print(f'{JOINT_NAMES[j]:<15} {per_joint_l1[j]:>9.3f} {norm_l1[j]:>9.4f} {bias[j]:>10.3f} {pred_mean[j]:>10.3f} {tgt_mean[j]:>10.3f}')
            print('=' * 70)
            print('per-joint L1 (deg):', np.round(per_joint_l1, 3))
            print('per-joint normalized L1 (L1/delta):', np.round(norm_l1, 4))
            print('per-joint bias (pred-target, deg):', np.round(bias, 3))
        else:
            print('没有成功样本')

asyncio.run(main())
