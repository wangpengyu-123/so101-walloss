#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_smoother.py — 平滑执行（ActionEnsembler/接缝/限幅/等待）纯数学自测。

不含任何硬件/网络。验证：
  1. n=1 时 ensemble == 最新预测原样（退化到旧行为，无加权扭曲）
  2. n=3 时帧计数时间对齐 + 指数加权平均的精确数学（push 快照帧数 0/8/16）
  3. horizon 越界裁剪（start_idx=31 无 NaN/除零；负 age 截断到帧 0 而非末尾）
  4. 接缝缓动 + max_delta 限幅链：chunk 间轨迹跳变时每帧步进 ≤ max_delta
  5. 接缝缓动分支真正执行（last_target 非 None）且数学精确、不引入额外扭曲
  6. wait_new_prediction：无新预测持续阻塞、通知/停止即唤醒
  7. 空缓冲保护（raise）+ 空档重置「只留最新」（exec_frame 快照保留）
  8. 帧计数对齐：启动瞬态（前两条预测同条件）相位一致；空档期间计数器不前进 →
     旧预测不被空档误推进（相对锚定 k=0 恒 age=0）
  9. decay 边界：0=只用最新，1=新旧等权
  10. wsum==0 防御（decay=-1 两条目时 raise，而不是产出 NaN）
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallx_realtime_client as wrc

N, H = wrc.N_JOINTS, wrc.ACTION_HORIZON
PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  PASS  {name}  {detail}")


def make_traj(delta_per_frame, offset=0.0):
    """构造一条内部连续的 32 帧轨迹：frame i = offset + i*delta（6 关节同值）。"""
    t = np.zeros((H, N))
    for i in range(H):
        t[i, :] = offset + i * delta_per_frame
    return t


print("== 1. n=1：ensemble 应等于最新预测原样 ==")
ens = wrc.ActionEnsembler(history=3, decay=0.5)
traj = make_traj(2.0, offset=10.0)
ens.push(traj, exec_frame=0)
chunk = ens.ensemble_chunk(start_idx=0, chunk_size=8)
for j in range(8):
    check("n=1 frame", np.allclose(chunk[j], traj[j]), f"j={j} chunk={chunk[j,0]:.3f} traj={traj[j,0]:.3f}")

print("== 2. n=3：帧计数时间对齐 + 权重精确数学 ==")
ens.clear()
p0, p1, p2 = make_traj(1.0, 40.0), make_traj(1.0, 32.0), make_traj(1.0, 24.0)
ens.push(p0, exec_frame=0); ens.push(p1, exec_frame=8); ens.push(p2, exec_frame=16)
chunk = ens.ensemble_chunk(start_idx=0, chunk_size=8)
w0, w1, w2 = 0.25, 0.5, 1.0
ws = w0 + w1 + w2
# 基准 = 最新快照 16：p0 age=16 → [16+j]；p1 age=8 → [8+j]；p2 age=0 → [j]
for j in range(8):
    expect = (w0 * p0[16 + j] + w1 * p1[8 + j] + w2 * p2[j]) / ws
    check("n=3 align+weight", np.allclose(chunk[j], expect), f"j={j} got={chunk[j,0]:.4f} expect={expect[0]:.4f}")

print("== 3. horizon 越界裁剪：start_idx=31 无 NaN；负 age 截断到帧 0 ==")
chunk = ens.ensemble_chunk(start_idx=31, chunk_size=8)
check("clamp no-nan", np.isfinite(chunk).all())
expect_hold = (w0 * p0[31] + w1 * p1[31] + w2 * p2[31]) / ws
check("clamp = weighted last frames", np.allclose(chunk, expect_hold),
      f"chunk0={chunk[0,0]:.3f} expect={expect_hold[0]:.3f}")
# 负 age（注入：旧预测快照 > 基准，时钟倒拨等异常）→ idx 截断到 0，读起始帧而非末尾帧
ens3 = wrc.ActionEnsembler(history=3, decay=0.5)
ens3.push(make_traj(1.0, 100.0), exec_frame=99)   # 旧预测快照异常地大
ens3.push(make_traj(1.0, 0.0), exec_frame=0)      # 基准 = 0 → 旧预测 age = -99 → 截断到帧 0
chunk3 = ens3.ensemble_chunk(start_idx=5, chunk_size=8)
expect3 = (0.5 * 100.0 + 1.0 * 5.0) / 1.5   # 旧预测读其帧 0（=100），最新读 [5+j]
check("neg-age clamps to frame0", np.allclose(chunk3[0], expect3),
      f"got={chunk3[0,0]:.4f} expect={expect3:.4f}（读起始帧而非末尾帧）")

print("== 4. 接缝缓动 + max_delta 限幅链（轨迹跳变 20 单位）==")
last_target = np.full(N, 50.0)
cur = np.full(N, 50.0)
new_chunk = np.array([np.full(N, 70.0)] * 8)   # 新 chunk 目标恒 70
max_delta = 5.0
executed = []
for j in range(8):
    t = new_chunk[j].copy()
    if j < min(3, len(wrc.SEAM_ALPHAS)):
        a = wrc.SEAM_ALPHAS[j]
        t = last_target + a * (t - last_target)
    d = np.clip(t - cur, -max_delta, max_delta)
    t = cur + d
    executed.append(t.copy())
    cur = t.copy()
for j in range(1, 8):
    step = np.max(np.abs(executed[j] - executed[j - 1]))
    check("max-delta step", step <= max_delta + 1e-9, f"j={j} step={step:.3f}")
check("seam j=0 math", np.allclose(executed[0], 55.0), f"got={executed[0][0]:.3f}")
check("seam j=1 + clamp", np.allclose(executed[1], 60.0), f"got={executed[1][0]:.3f}")
check("j=2 clamp", np.allclose(executed[2], 65.0), f"got={executed[2][0]:.3f}")

print("== 5. 接缝缓动分支真正执行（last_target 非 None）：数学精确 ==")
last_target = np.full(N, -2.0)
cur = np.full(N, 0.0)
traj = make_traj(3.0, 0.0)
executed = []
for j in range(8):
    t = traj[j].copy()
    if j < 3:
        a = wrc.SEAM_ALPHAS[j]
        t = last_target + a * (t - last_target)
    if max_delta > 0:
        d = np.clip(t - cur, -max_delta, max_delta)
        t = cur + d
    executed.append(t.copy())
    cur = t.copy()
check("seam active j=0", np.allclose(executed[0], -1.5), f"got={executed[0][0]:.3f}")
check("seam active j=1", np.allclose(executed[1], 1.0), f"got={executed[1][0]:.3f}")
check("seam active j=2", np.allclose(executed[2], 6.0), f"got={executed[2][0]:.3f}")
for j in range(3, 8):
    check("follow after seam", np.allclose(executed[j], traj[j]), f"j={j} got={executed[j][0]:.3f} expect={traj[j][0]:.3f}")

print("== 6. wait_new_prediction：无新预测时持续阻塞，有通知/停止即唤醒 ==")
import threading, time
cond = threading.Condition()
seq = {"n": 0}
stopped = threading.Event()
should_stop = lambda: stopped.is_set()
wake = {"t": None}
def waiter():
    wake["t"] = wrc.wait_new_prediction(cond, seq, 0, should_stop, poll=0.1)
th = threading.Thread(target=waiter)
th.start()
time.sleep(0.35)
check("still waiting", th.is_alive(), "0.35s 无新预测仍在阻塞")
with cond:
    seq["n"] = 1
    cond.notify_all()
th.join(timeout=2)
check("notify wakeup", not th.is_alive() and wake["t"] is not None, f"waited={wake['t']:.3f}s")
wake2 = {"t": None}
def waiter2():
    wake2["t"] = wrc.wait_new_prediction(cond, seq, 1, should_stop, poll=0.1)
th2 = threading.Thread(target=waiter2)
th2.start()
time.sleep(0.15)
stopped.set()
th2.join(timeout=2)
check("stop wakeup", not th2.is_alive(), f"stopped 后立即返回 waited={wake2['t']:.3f}s")

print("== 7. 空缓冲保护 + 空档重置「只留最新」（快照帧数保留）==")
ens7 = wrc.ActionEnsembler(history=3, decay=0.5)
try:
    ens7.ensemble_chunk(0, 8)
    check("empty raises", False, "空缓冲没有抛错！")
except ValueError as e:
    check("empty raises", "缓冲为空" in str(e), f"msg={e}")
ens7.push(make_traj(1.0, 40.0), exec_frame=0)
ens7.push(make_traj(1.0, 32.0), exec_frame=8)
ens7.push(make_traj(1.0, 24.0), exec_frame=16)
newest = ens7.newest()
ens7.clear()
ens7.push(newest[1], exec_frame=newest[0])
check("keep-newest depth", ens7.depth == 1, f"depth={ens7.depth}")
check("keep-newest snapshot", ens7.newest()[0] == 16, f"snapshot={ens7.newest()[0]}")
chunk = ens7.ensemble_chunk(0, 8)
check("keep-newest ensemble", np.allclose(chunk, newest[1][:8]), f"chunk0={chunk[0,0]:.3f} newest0={newest[1][0,0]:.3f}")

print("== 8. 帧计数对齐：启动瞬态相位一致 + 空档不推进旧预测 ==")
# 启动瞬态（复现审核场景）：P_0/P_1 同条件（同一轨迹 from 0），P_2 条件于 8 帧后（轨迹 from 8）。
# 相对锚定：基准 = P_2 快照 8 → P_2 age=0、P_1 age=8、P_0 age=8
ens8 = wrc.ActionEnsembler(history=3, decay=0.5)
ens8.push(make_traj(1.0, 0.0), exec_frame=0)     # P_0，条件化于 S0
ens8.push(make_traj(1.0, 0.0), exec_frame=0)     # P_1，同样条件化于 S0（启动瞬态）
ens8.push(make_traj(1.0, 8.0), exec_frame=8)     # P_2，条件化于 8 帧后
chunk8 = ens8.ensemble_chunk(start_idx=0, chunk_size=8)
# 三条相位一致：P_2[j]=8+j、P_1[8+j]=8+j、P_0[8+j]=8+j → ensemble = 8+j（零扭曲）
for j in range(8):
    check("startup phase-consistent", np.allclose(chunk8[j], 8.0 + j),
          f"j={j} got={chunk8[j,0]:.4f} expect={8.0+j:.4f}")
# 空档冻结：墙钟过去 2s 但 0 帧执行（手臂静止）→ ensemble 结果必须与上完全一致
ens8b = wrc.ActionEnsembler(history=3, decay=0.5)
ens8b.push(make_traj(1.0, 0.0), exec_frame=0)
ens8b.push(make_traj(1.0, 0.0), exec_frame=0)
ens8b.push(make_traj(1.0, 8.0), exec_frame=8)
chunk8b = ens8b.ensemble_chunk(start_idx=0, chunk_size=8)
check("gap-freeze", np.allclose(chunk8b, chunk8), "空档不推进轨迹年龄")

print("== 9. decay 边界：0=只用最新，1=新旧等权 ==")
ens9 = wrc.ActionEnsembler(history=3, decay=0.0)
ens9.push(make_traj(1.0, 40.0), exec_frame=0)
ens9.push(make_traj(1.0, 32.0), exec_frame=8)
ens9.push(make_traj(1.0, 24.0), exec_frame=16)
chunk9 = ens9.ensemble_chunk(start_idx=0, chunk_size=8)
check("decay=0 newest only", np.allclose(chunk9, make_traj(1.0, 24.0)[:8]),
      f"chunk0={chunk9[0,0]:.3f} expect=24.0")
ens9b = wrc.ActionEnsembler(history=3, decay=1.0)
ens9b.push(make_traj(1.0, 40.0), exec_frame=0)
ens9b.push(make_traj(1.0, 32.0), exec_frame=8)
ens9b.push(make_traj(1.0, 24.0), exec_frame=16)
chunk9b = ens9b.ensemble_chunk(start_idx=0, chunk_size=8)
expect9b = (p0[16:24] + p1[8:16] + p2[0:8]) / 3.0
check("decay=1 equal weights", np.allclose(chunk9b, expect9b),
      f"chunk0={chunk9b[0,0]:.4f} expect={expect9b[0,0]:.4f}")

print("== 10. wsum==0 防御：decay=-1 两条目时 raise（不产 NaN）==")
ens10 = wrc.ActionEnsembler(history=3, decay=-1.0)
ens10.push(make_traj(1.0, 10.0), exec_frame=0)
ens10.push(make_traj(1.0, 10.0), exec_frame=8)
try:
    out = ens10.ensemble_chunk(start_idx=0, chunk_size=8)
    check("wsum=0 raises", False, f"没有抛错，输出={out[0,0]}")
except ValueError as e:
    check("wsum=0 raises", "权重和为 0" in str(e), f"msg={e}")

print("== 11. smooth_plan 滑动平均：抑制尖峰、保底斜坡 ==")
sp = np.zeros((5, 6)); sp[2, :] = 10.0
out_sp = wrc.smooth_plan(sp, 3)
expect_sp = np.zeros((5, 6)); expect_sp[1:4, :] = 10.0 / 3.0
check("spike damped", np.allclose(out_sp, expect_sp), f"mid={out_sp[1,0]:.3f} expect={10/3:.3f}")
ramp = make_traj(3.0, 0.0)
out_ramp = wrc.smooth_plan(ramp, 3)
check("ramp preserved (interior)", np.allclose(out_ramp[1:-1], ramp[1:-1]),
      "斜坡内部不被滑动平均改变")
check("window<=1 passthrough", wrc.smooth_plan(ramp, 0) is ramp and wrc.smooth_plan(ramp, 1) is ramp)
check("bias correction dims", wrc.JOINT_BIAS.shape == (6,), f"shape={wrc.JOINT_BIAS.shape}")

print("== 12. GraspReflex 夹取判定状态机 ==")
def L(pos, gripper):
    s = np.zeros(N); s[:5] = pos; s[5] = gripper
    return s
def T(pos, gripper):
    s = np.zeros(N); s[:5] = pos; s[5] = gripper
    return s
r = wrc.GraspReflex()
# 1. 夹爪张开（>28）→ tracking 原样返回
t_out, ph = r.step(L(10.0, 50.0), T(11.0, 45.0), 5.0, 0.06)
check("open gripper tracking", ph == "tracking" and np.allclose(t_out, T(11.0, 45.0)), f"ph={ph}")
# 1b. 接近途中：位置持续移动 + 夹爪已 ≤28 → 不触发（到位结算未满足）
for i in range(20):
    t_out, ph = r.step(L(10.0 + i * 3.0, 24.0), T(10.0 + i * 3.0 + 1.0, 30.0), 5.0, 0.06)
check("moving no trigger", ph == "tracking", f"ph={ph}（接近途中不触发）")
# 1c. 到位结算：位置稳定 12 帧（夹爪保持张开 >28，不提前触发）
for _ in range(15):
    t_out, ph = r.step(L(70.0, 50.0), T(70.0, 45.0), 5.0, 0.06)
check("settled but gripper open", ph == "tracking", f"ph={ph}（位置稳定但夹爪张开不触发）")
# 2. 触发：位置稳定 + 夹爪 25 ≤ 28 且位置接近 → closing，位置冻结、夹爪按限幅夹紧
for _ in range(13):
    t_out, ph = r.step(L(70.0, 25.0), T(71.0, 30.0), 5.0, 0.06)
check("trigger closing", ph == "closing", f"ph={ph}")
check("pos frozen", np.allclose(t_out[:5], 70.0), f"pos={t_out[:5]}")
check("gripper ramp", abs(t_out[5] - 20.0) < 1e-9, f"gripper={t_out[5]:.1f} expect=20（25-5）")
# 3. 夹紧到位（实况夹爪 ≤ 16）→ carry
t_out, ph = r.step(L(70.0, 15.5), T(71.0, 30.0), 5.0, 0.06)
check("closing→carry", ph == "carry", f"ph={ph}")
# 4. carry：位置跟随计划、夹爪保持（从夹取点 70 附近开始）
t_out, ph = r.step(L(72.0, 15.0), T(73.0, 45.0), 5.0, 0.06)
check("carry pos follows", np.allclose(t_out[:5], 73.0) and ph == "carry", f"pos={t_out[:5]} ph={ph}")
check("carry gripper held", abs(t_out[5] - 15.0) < 1e-9, f"gripper={t_out[5]:.1f} expect=15（保持夹紧）")
# 5. 位移释放：EMA 中心位移 > 25 → tracking（释放即停：真实夹爪随后会物理张开，测试假实况不跟踪命令）
ph = "carry"
for i in range(60):
    t_out, ph = r.step(L(72.0 + i * 1.2, 15.0), T(72.0 + i * 1.2, 45.0), 5.0, 0.06)
    if ph == "tracking":
        break
check("release by displacement", ph == "tracking", f"ph={ph}（EMA 位移超阈值后释放，第 {i} 步）")
# 5b. 冷却期：释放后 2s 内不重新触发（夹爪物理张开前的防重复夹取）
t_out, ph = r.step(L(72.0, 15.0), T(72.0, 45.0), 5.0, 0.06)
check("cooldown suppress", ph == "tracking", f"ph={ph}（冷却期内不触发）")
for _ in range(40):
    t_out, ph = r.step(L(72.0, 15.0), T(72.0, 45.0), 5.0, 0.06)
check("cooldown expires → retrigger", ph == "closing" or ph == "carry",
      f"ph={ph}（冷却 2s 后重新触发）")
# 6. 超时释放：位移不动但 8s 超时 → tracking（释放即停，冷却后重触发属正常循环）
r2 = wrc.GraspReflex()
for _ in range(13):
    r2.step(L(10.0, 50.0), T(11.0, 45.0), 5.0, 0.06)   # 先张开结算
for _ in range(13):
    r2.step(L(10.0, 25.0), T(11.0, 30.0), 5.0, 0.06)   # → closing
r2.step(L(10.0, 15.0), T(11.0, 30.0), 5.0, 0.06)      # → carry
ph = "carry"
for i in range(200):
    t_out, ph = r2.step(L(10.0, 15.0), T(11.0, 45.0), 5.0, 0.06)
    if ph == "tracking":
        break
check("release by timeout", ph == "tracking", f"ph={ph}（8s 超时兜底，第 {i} 步 ≈ {i*0.06:.1f}s）")

print("== 13. parse_args 参数校验：NaN/inf/越界必须拒绝 ==")
import argparse as _ap
_orig_argv = sys.argv[:]
def _try_args(*extra):
    try:
        sys.argv = ["wallx_realtime_client.py"] + list(extra)
        wrc.parse_args()
        return None
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = _orig_argv
check("reject --smooth-decay -1", _try_args("--smooth-decay", "-1") is not None)
check("reject --smooth-decay nan", _try_args("--smooth-decay", "nan") is not None)
check("reject --ensemble-history 0", _try_args("--ensemble-history", "0") is not None)
check("reject --chunk-size 0", _try_args("--chunk-size", "0") is not None)
check("reject --execute-interval nan", _try_args("--execute-interval", "nan") is not None)
check("reject --execute-interval inf", _try_args("--execute-interval", "inf") is not None)
check("reject --interval inf", _try_args("--interval", "inf") is not None)
check("reject --interval -1", _try_args("--interval", "-1") is not None)
check("reject --max-delta nan", _try_args("--max-delta", "nan") is not None)
check("reject --timeout 0", _try_args("--timeout", "0") is not None)
check("reject --max-seconds -5", _try_args("--max-seconds", "-5") is not None)
check("accept 默认参数", _try_args() is None)
check("accept 合法组合", _try_args("--execute", "--max-seconds", "30", "--smooth-decay", "0.3") is None)

print(f"\n全部通过：{PASS} 项断言")
