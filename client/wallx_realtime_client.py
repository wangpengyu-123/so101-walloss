#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wallx_realtime_client.py — SO-ARM101 真机 ←→ 云端 wall-x serving 实时推理客户端

链路：/dev/video10 相机帧 + /dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82111073-if00 从臂 6 关节 → ws://localhost:32195 云端推理
     → predict_action[32,26] 取前 6 维 → （--execute 时）写回从臂舵机。

单位约定（全部经代码+数据实测确认，见项目汇报）：
  - 与云端交互的状态/动作单位 = lerobot 的归一化单位：
      关节 1-5（shoulder_pan..wrist_roll）: RANGE_M100_100（标定范围百分比，-100..100）
      关节 6（gripper）:              RANGE_0_100（0..100）
    这正好是训练数据集 so101_table_pick_place 的原始单位（parquet 实测 + norm_stats 印证）。
  - 舵机原始编码 0..4095 与归一化单位的换算由 lerobot FeetechMotorsBus 用同一份
    标定文件（my_awesome_follower_arm.json）自动完成：
      norm = ((raw - range_min)/(range_max - range_min)) * 200 - 100   (RANGE_M100_100)
      norm = ((raw - range_min)/(range_max - range_min)) * 100         (RANGE_0_100)
    （drive_mode=0，无符号翻转；bus.sync_read/sync_write 默认 normalize=True 即走这套换算）
  - 云端 serving 侧：状态用 norm_stats.json 的 q01/q99 做 min-max 归一化到 [-1,1]，
    模型输出动作再按 q01/q99 反归一化回上述单位后返回（--no-serialize-actions 原样返回
    [32,26]，前 6 维=6 关节，后 20 维=action_padding 恒 0）。

安全设计：
  - 默认 dry-run：只打印动作，绝不碰舵机。
  - 只有显式传 --execute 才写舵机；每个 chunk 执行后打印首/末帧目标（归一化）+ 末帧原始值。
  - --max-delta 限制单次执行每关节相对当前值的最大步进（默认 5.0 归一化单位，
    ≈ 关节量程的 2.5%；传 0 关闭）。
  - 执行帧选择：32 帧里挑与当前关节状态最接近的一帧（ACT 惯例），再叠加 max-delta 限幅。
  - Ctrl-C 立即停止发送（急停）；意外退出不再写 Goal_Position，舵机保持当前位姿。
  - 云端返回 NaN/Inf 或形状不对 → 丢弃并报警，不执行。

平滑执行（--execute 时默认开启，参考 lerobot ACT 真机做法）：
  - 流水线：推理线程持续取帧推理入缓冲，主线程按固定节拍连续执行（不再推理-执行串行卡顿）
  - 计划复用：一次对齐后顺序走完同一个 32 帧计划（约 2s），消除每轮重对齐的相位跳变
    （旧行为在位置静止的相位里来回挑帧 → 手臂来回挪/夹爪开合反复）
  - 关节偏差补偿：round3 开环 60 样本实测 bias 反向修正（腕 +26.9°/肘 -12.5°/夹爪 +8.2° 等），
    --no-bias-correct 关闭
  - 计划平滑：3 帧滑动平均抑制计划内逐帧预测噪声（--plan-smooth，0=关闭）
  - 夹取判定（grasp reflex，默认开）：判定模型「意图闭合 + 位置静止」→ 冻结位置 + 夹紧
    → 搬运保持（位置跟计划、夹爪无视模型重张开）→ 位移/超时后释放（--no-grasp-reflex 关闭）
  - temporal ensembling：默认关闭（--ensemble-history 默认 1 = 只用最新预测、纯平滑；
    实验时可设 2~3 开启多条预测按已执行帧计数对齐的加权平均）
  - 接缝插值：chunk 前几帧（--seam-frames）从本 chunk 起始实际位置缓动进入新轨迹
  - 固定节拍：每帧按 --execute-interval（默认 0.083s=12fps 训练帧率）wall-clock 匀速执行

用法（conda env lerobot）：
  # 真机 dry-run 联调（只读不写，打印动作与延迟）：
  python wallx_realtime_client.py --frames 5
  # 相机缺失时用离线帧测试推理链路：
  python wallx_realtime_client.py --frames 3 --image /tmp/frame.jpg
  # 真机执行（用户在场时）：
  python wallx_realtime_client.py --execute
  # 完整参数见 --help
"""

import argparse
import base64
import json
import logging
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import websockets
import websockets.sync.client as ws_sync

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import MotorCalibration

log = logging.getLogger("wallx_client")

# ---------------------------------------------------------------------------
# 常量（与 lerobot so101_follower / 训练配置一致，顺序不能乱）
# ---------------------------------------------------------------------------
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
N_JOINTS = 6
ACTION_HORIZON = 32
ACTION_DIM = 26          # 6 关节 + 20 padding（config_full.yml: so101_joint_pos 6 + action_padding 20）
DEFAULT_CALIB = "/home/ubuntu22/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json"
DEFAULT_INSTRUCTION = "Grab the red block and place it on the target"
JOINT_RANGE = (-100.0, 100.0)   # 关节 1-5 归一化量程
GRIPPER_RANGE = (0.0, 100.0)    # gripper 归一化量程


# ---------------------------------------------------------------------------
# 从臂读写（复用 lerobot FeetechMotorsBus 与录制时同一份标定）
# ---------------------------------------------------------------------------
class FollowerArm:
    def __init__(self, port: str, calib_path: str = DEFAULT_CALIB):
        cal = json.load(open(calib_path))
        self.calib = {
            name: MotorCalibration(
                id=v["id"], drive_mode=v["drive_mode"], homing_offset=v["homing_offset"],
                range_min=v["range_min"], range_max=v["range_max"],
            )
            for name, v in cal.items()
        }
        motors = {
            name: Motor(i, "sts3215",
                        MotorNormMode.RANGE_0_100 if i == 6 else MotorNormMode.RANGE_M100_100)
            for i, name in enumerate(JOINT_ORDER, start=1)
        }
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=self.calib)

    def connect(self):
        self.bus.connect()
        log.info("从臂已连接 %s, EEPROM 标定与文件一致: %s",
                 self.bus.port, self.bus.is_calibrated)
        return self

    def read_state(self) -> np.ndarray:
        """读 6 关节当前值（归一化单位，与训练数据同单位）。"""
        pos = self.bus.sync_read("Present_Position")  # normalize=True
        return np.array([pos[n] for n in JOINT_ORDER], dtype=np.float64)

    def write_state(self, target: np.ndarray) -> np.ndarray:
        """把 6 维归一化目标写成 Goal_Position（bus 内部换算到 0..4095）。返回入参目标（总线内部量化不回读）。"""
        goal = {n: float(target[i]) for i, n in enumerate(JOINT_ORDER)}
        self.bus.sync_write("Goal_Position", goal)  # normalize=True
        return target

    def disconnect(self):
        try:
            self.bus.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------
class Camera:
    """读 /dev/video10（v4l2loopback，640x480 RGB3）。

    注意：若 ASC60C 相机本体掉线，桥接进程不再写入新帧，read 会阻塞；
    用 CAP_PROP_READ_TIMEOUT_MSEC 限时避免卡死。
    """

    def __init__(self, path: str, read_timeout_ms: int = 1500, jpeg_quality: int = 95):
        self.path = path
        self.jpeg_quality = jpeg_quality
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开相机 {path}（相机链是否在跑？）")
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        if read_timeout_ms > 0:
            if not self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms):
                log.warning("相机 %s 不支持 READ_TIMEOUT 设置（读帧卡死保护不可用）", path)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("相机已打开 %s: %dx%d", path, w, h)

    def grab_b64_jpeg(self) -> tuple[str, float]:
        """读一帧 → JPEG → base64 字符串。返回 (b64, 耗时ms)。"""
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"读帧失败/超时（{self.path} 无新帧）")
        # 统一缩放到训练尺寸 640x480（ASC60C 原生 UVC 是 1280x1040）
        if frame.shape[0] != 480 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        dt_ms = (time.time() - t0) * 1000
        return base64.b64encode(buf.tobytes()).decode("ascii"), dt_ms

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def load_image_override(path: str) -> str:
    """--image 离线帧：读文件（jpg/png）→ 按训练同尺寸 640x480 缩放 → base64。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图像文件 {path}")
    if img.shape[0] != 480 or img.shape[1] != 640:
        img = cv2.resize(img, (640, 480))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# 云端推理
# ---------------------------------------------------------------------------
class WallxCloud:
    def __init__(self, ws_url: str, timeout: float = 60.0):
        self.ws_url = ws_url
        self.timeout = timeout
        self.ws = None

    def connect(self):
        self.ws = ws_sync.connect(self.ws_url, open_timeout=15, close_timeout=5)
        meta = msgpack.unpackb(self.ws.recv(self.timeout), raw=False)
        log.info("已连接 %s, metadata=%s", self.ws_url, meta)
        return meta

    def infer(self, state: np.ndarray, frame_b64: str, wrist_b64: str, instruction: str,
              recv_timeout: float = None) -> dict:
        """发一帧观测（global+wrist 双相机），收 predict_action[32,26]。返回 {predict_action, server_timing, rtt_ms}。

        recv_timeout：本次收包超时秒数（默认 self.timeout；dry-run 传短超时让 Ctrl-C 更快生效）。
        """
        req = {
            "state": {"so101_joint_pos": state.tolist()},
            "views": {"global_view": frame_b64, "wrist_view": wrist_b64},
            "instruction": instruction,
        }
        t0 = time.time()
        self.ws.send(msgpack.packb(req))
        resp = msgpack.unpackb(self.ws.recv(recv_timeout if recv_timeout is not None else self.timeout), raw=False)
        rtt_ms = (time.time() - t0) * 1000
        if not isinstance(resp, dict) or "predict_action" not in resp:
            raise RuntimeError(f"云端响应异常: {type(resp)} {str(resp)[:300]}")
        pa = np.asarray(resp["predict_action"], dtype=np.float64)
        if pa.ndim == 3 and pa.shape[0] == 1:      # 服务端带 batch 维 (1,32,26)
            pa = pa[0]
        if pa.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"predict_action 形状异常: {pa.shape}，期望 ({ACTION_HORIZON},{ACTION_DIM})")
        if not np.isfinite(pa[:, :N_JOINTS]).all():
            raise RuntimeError("predict_action 含 NaN/Inf，丢弃")
        out = {
            "predict_action": pa,
            "server_timing": resp.get("server_timing", {}),
            "rtt_ms": rtt_ms,
        }
        if "subtask" in resp:
            out["subtask"] = resp["subtask"]
        return out

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 动作处理（安全裁剪 + 执行帧选择）
# ---------------------------------------------------------------------------
def clamp_joints(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    x[:5] = np.clip(x[:5], *JOINT_RANGE)
    x[5] = np.clip(x[5], *GRIPPER_RANGE)
    return x


def select_and_clamp(action: np.ndarray, current: np.ndarray, max_delta: float):
    """选 32 帧中与当前状态最接近的一帧，并做相对步进限幅。

    返回 (target, chosen_frame_index) 二元组。
    """
    act = np.asarray(action, dtype=np.float64)[:, :N_JOINTS]
    act = clamp_joints(act)
    dist = np.linalg.norm(act - current[None, :], axis=1)
    idx = int(np.argmin(dist))
    target = act[idx].copy()
    if max_delta > 0:
        delta = np.clip(target - current, -max_delta, max_delta)
        target = current + delta
        target = clamp_joints(target)
    return target, idx


# ---------------------------------------------------------------------------
# 平滑执行（temporal ensembling + 接缝插值 + 固定节拍，参考 lerobot ACT 真机做法）
# ---------------------------------------------------------------------------
SEAM_ALPHAS = (0.25, 0.6, 1.0)  # 接缝缓动系数：chunk 前 seam_frames 帧从本 chunk 起始实际位置缓动进入新轨迹

# 关节级系统偏差补偿（实测值，来源：round4 10000步开环评估 60 样本，bias = mean(预测-真值)，
# 单位 = lerobot 归一化单位：关节 ±100、夹爪 0..100）。
# shoulder_pan -2.18 / shoulder_lift +6.18 / elbow_flex -4.72 /
# wrist_flex +8.08 / wrist_roll -1.27 / gripper -6.04
# 补偿 = 预测 - bias；--no-bias-correct 关闭。
JOINT_BIAS = np.array([-2.18, 6.18, -4.72, 8.08, -1.27, -6.04])


def smooth_plan(plan: np.ndarray, window: int) -> np.ndarray:
    """计划轨迹滑动平均（居中窗口，越界裁剪；window<=1 原样返回）。

    抑制计划内的高频帧间抖动（模型预测逐帧噪声），不改相位、不拖慢整体轨迹。
    """
    if window <= 1:
        return plan
    out = np.empty_like(plan)
    half = window // 2
    for i in range(plan.shape[0]):
        lo = max(0, i - half)
        hi = min(plan.shape[0], i + half + 1)
        out[i] = plan[lo:hi].mean(axis=0)
    return out


class GraspReflex:
    """夹取判定状态机（grasp reflex）。

    问题：模型在夹取相位反复开合、位置小幅振荡，夹取动作永远无法完成。
    判定：当模型已驱动夹爪进入闭合区（实况夹爪 <= trigger_gripper）且计划位置目标
    与实况接近（相位静止区）时，判定为「夹取时刻」：
      tracking（跟随计划）
        → closing：位置冻结在判定点，夹爪按 max_delta 限幅夹紧到 hold_value
        → carry：位置恢复跟随计划（抬升/搬运），夹爪保持夹紧（无视模型重张开）
        → tracking：手臂位移（EMA 平滑，滤掉原地抖动）超过 release_dist 或超时
          后恢复完全跟随计划（让模型在目标处执行释放）
    纯状态机，无 IO/日志，可单测；--no-grasp-reflex 关闭。
    """

    def __init__(self, trigger_gripper: float = 28.0, pos_radius: float = 10.0,
                 hold_value: float = 15.0, release_dist: float = 25.0,
                 timeout_s: float = 8.0, ema_alpha: float = 0.2, cooldown_s: float = 2.0,
                 settle_delta: float = 2.5, settle_frames: int = 12):
        self.trigger_gripper = trigger_gripper
        self.pos_radius = pos_radius
        self.hold_value = hold_value
        self.release_dist = release_dist
        self.timeout_s = timeout_s
        self.ema_alpha = ema_alpha
        self.cooldown_s = cooldown_s
        self.settle_delta = settle_delta
        self.settle_frames = settle_frames
        self.phase = None            # None(tracking) | "closing" | "carry"
        self.grasp_pos = None        # 判定时的位置（冻结基准/位移基准）
        self.t_elapsed = 0.0
        self.carry_center = None     # 搬运阶段位置的 EMA（滤掉原地抖动）
        self.cooldown_left = 0.0     # 释放后冷却：夹爪物理张开前防重复触发
        self.prev_pos = None         # 到位结算检测：上一帧实况位置
        self.settle_count = 0        # 连续「几乎不动」帧数

    def step(self, live: np.ndarray, plan_target: np.ndarray,
             max_delta: float, dt: float):
        """输入实况 6 维与（接缝+偏差修正后的）计划目标，返回 (修正目标, 相位)。"""
        if self.cooldown_left > 0:
            self.cooldown_left = max(0.0, self.cooldown_left - dt)
        # 到位结算：实况位置连续 settle_frames 帧几乎不动 = 到达夹取点。
        # 关键作用：区分「接近途中」（位置持续移动，即使模型已提前发出闭合指令也不触发）
        # 与「到达夹取点」（位置稳定，含小幅摇摆）。
        if self.prev_pos is not None:
            if np.linalg.norm(live[:5] - self.prev_pos) < self.settle_delta:
                self.settle_count += 1
            else:
                self.settle_count = 0
        self.prev_pos = live[:5].copy()
        if self.phase is None:
            if (self.cooldown_left <= 0
                    and live[5] <= self.trigger_gripper
                    and self.settle_count >= self.settle_frames
                    and np.linalg.norm(plan_target[:5] - live[:5]) < self.pos_radius):
                self.phase = "closing"
                self.grasp_pos = live[:5].copy()
                self.t_elapsed = 0.0
                self.carry_center = live[:5].copy()
        if self.phase == "closing":
            t = plan_target.copy()
            t[:5] = self.grasp_pos                      # 位置冻结
            step = max_delta if max_delta > 0 else float("inf")
            t[5] = max(self.hold_value, live[5] - step)  # 夹紧（限幅）
            if live[5] <= self.hold_value + 1.0:
                self.phase = "carry"
            return t, self.phase   # 返回实际状态（可能已转入 carry）
        if self.phase == "carry":
            self.t_elapsed += dt
            self.carry_center = ((1.0 - self.ema_alpha) * self.carry_center
                                 + self.ema_alpha * live[:5])
            if (np.linalg.norm(self.carry_center - self.grasp_pos) > self.release_dist
                    or self.t_elapsed > self.timeout_s):
                self.phase = None
                self.cooldown_left = self.cooldown_s
                return plan_target, "tracking"
            t = plan_target.copy()
            step = max_delta if max_delta > 0 else float("inf")
            t[5] = max(self.hold_value, live[5] - step)  # 保持夹紧
            return t, "carry"
        return plan_target, "tracking"


class ActionEnsembler:
    """ACT 式 temporal ensembling 动作缓冲。

    - 保存最近 history 条预测（最新在尾部），每条 = (push 时已执行帧数, [ACTION_HORIZON, N_JOINTS] 轨迹)
    - 时间对齐用「已执行帧计数器」而非墙钟：age = 最新预测的快照帧数 - 该预测的快照帧数。
      关键性质：推理空档期间主线程不写舵机、计数器不前进 → 旧预测年龄不包含空档，
      空档不会被误算成轨迹推进；启动瞬态（前两条预测同条件）也能精确对齐。
    - 以最新预测为相位基准：k=0 恒 age=0（start_idx 已把最新预测锚到当前实况位置）。
    - 加权平均权重 w_k = decay**k（最新最重）。
      注意：与 lerobot ACT 默认相反（ACT 的 exp(-i) 微偏最旧）；本客户端真机执行
      刻意强偏最新——最新预测条件化状态最新鲜，旧预测只用于抑制抖动，安全优先。
    - 越界帧双向截断：上界取轨迹末帧（保持末端位姿）、下界取帧 0（防御性，正常不会出现）

    前提（帧计数对齐成立的条件）：推理延迟基本恒定（约 1 个 chunk），所有预测共享
    相同的「条件化→push」滞后，start_idx 重锚把共享滞后整体抵消；推理延迟抖动只
    产生 ≤1-2 帧的相位误差，被 decay 加权稀释。

    线程安全：所有方法都必须在同一个 threading.Condition 保护下调用。
    """

    def __init__(self, history: int = 3, decay: float = 0.5):
        self.history = history
        self.decay = decay
        self._buf = deque(maxlen=history)  # 每条 = (exec_frame_at_push, act)

    @property
    def depth(self) -> int:
        return len(self._buf)

    def push(self, act: np.ndarray, exec_frame: int = None) -> None:
        """act: [ACTION_HORIZON, N_JOINTS] 已 clamp 的预测轨迹。

        exec_frame：push 时主线程已执行的帧数快照（调用方传入；测试可注入）。
        """
        self._buf.append((0 if exec_frame is None else int(exec_frame),
                          np.asarray(act, dtype=np.float64)[:, :N_JOINTS].copy()))

    def newest(self) -> tuple:
        """返回 (exec_frame_at_push, act)。"""
        return self._buf[-1]

    def clear(self) -> None:
        self._buf.clear()

    def ensemble_chunk(self, start_idx: int, chunk_size: int) -> np.ndarray:
        """帧计数时间对齐 + 指数加权平均 → [chunk_size, N_JOINTS] 目标轨迹。

        start_idx：最新预测中与当前实际状态最近的帧（由调用方算好）。
        以最新预测的快照帧数为基准：age_k = 基准 - 第 k 条的快照帧数（k=0 恒 0）。
        """
        n = len(self._buf)
        if n == 0:
            raise ValueError("ensemble 缓冲为空（调用方必须保证先 push 再取）")
        base_frame, _ = self._buf[-1]
        out = np.empty((chunk_size, N_JOINTS), dtype=np.float64)
        for j in range(chunk_size):
            acc = np.zeros(N_JOINTS)
            wsum = 0.0
            for k in range(n):  # k=0 最新（_buf 尾部）
                f_push, pred = self._buf[n - 1 - k]
                age_frames = base_frame - f_push  # k=0 → 0；计数单调递增保证 >= 0
                idx = min(max(start_idx + j + age_frames, 0), ACTION_HORIZON - 1)
                w = self.decay ** k
                acc += w * pred[idx]
                wsum += w
            if wsum == 0.0:
                raise ValueError("ensemble 权重和为 0（smooth_decay 非法值）")
            out[j] = acc / wsum
        # 按列裁剪（out 是 [chunk_size, N_JOINTS] 矩阵，clamp_joints 只适用于单帧向量）
        out[:, :5] = np.clip(out[:, :5], *JOINT_RANGE)
        out[:, 5] = np.clip(out[:, 5], *GRIPPER_RANGE)
        return out


def wait_new_prediction(cond: threading.Condition, seq: dict, consumed: int,
                        should_stop, poll: float = 0.3) -> float:
    """阻塞直到推理线程 push 一条新预测（seq["n"] > consumed）或 should_stop() 为真。

    should_stop：可调用（包含 Ctrl-C 标志与停机 Event 的检查）。
    返回空档等待秒数。
    """
    t0 = time.time()
    with cond:
        while seq["n"] <= consumed and not should_stop():
            cond.wait(timeout=poll)
    return time.time() - t0


def run_execute_pipelined(args, arm: FollowerArm, cam, wrist_cam, cloud: WallxCloud,
                          image_override, stopped: threading.Event, stop_requested: list) -> None:
    """流水线真机执行：推理线程持续取帧→推理→入缓冲；主线程按固定节拍连续执行。

    线程安全职责划分：
    - 串口总线：只有主线程读/写（sync_read/sync_write），推理线程绝不碰
    - 云端 websocket：只有推理线程使用；主线程等推理线程退出后才 close
    - shared["state"]：主线程维护（chunk 开头=实况位置，每帧写后=命令位置），
      推理线程只读（dict 引用替换在 GIL 下原子，读到的永远是完整快照）
    - stop_requested[0]：信号处理器只置这个普通标志（不在信号上下文碰 Event 锁），
      主线程统一转成 stopped.set()
    """
    ens = ActionEnsembler(history=args.ensemble_history, decay=args.smooth_decay)
    cond = threading.Condition()
    seq = {"n": 0}
    exec_frames = {"n": 0}     # 主线程已执行帧计数（推理线程 push 时快照做时间对齐）
    shared = {"state": arm.read_state()}
    infer_fatal = []
    CAM_ERROR_LIMIT = 5        # 连续读帧失败超过此数 = 相机永久掉线 → 停机

    def infer_loop():
        cam_errors = 0
        wrist_errors = 0
        frame_b64 = image_override
        wrist_b64 = image_override
        try:
            while not stopped.is_set():
                if cam is not None:
                    try:
                        frame_b64, frame_ms = cam.grab_b64_jpeg()
                        cam_errors = 0
                    except RuntimeError as e:
                        cam_errors += 1
                        if frame_b64 is None or cam_errors > CAM_ERROR_LIMIT:
                            log.error("读帧失败（%s，连续 %d 次），停机", e, cam_errors)
                            infer_fatal.append(f"读帧失败（{e}）连续 {cam_errors} 次")
                            stopped.set()
                            break
                        log.warning("读帧失败（%s），复用上一帧（连续 %d 次）", e, cam_errors)
                if wrist_cam is not None:
                    try:
                        wrist_b64, _ = wrist_cam.grab_b64_jpeg()
                        wrist_errors = 0
                    except RuntimeError as e:
                        wrist_errors += 1
                        if wrist_errors > CAM_ERROR_LIMIT:
                            log.error("wrist 读帧失败（%s，连续 %d 次），停机", e, wrist_errors)
                            infer_fatal.append(f"wrist 读帧失败连续 {wrist_errors} 次")
                            stopped.set()
                            break
                        log.warning("wrist 读帧失败（%s），复用上一帧（连续 %d 次）", e, wrist_errors)
                if frame_b64 is None or wrist_b64 is None:
                    log.error("没有可用帧，停机")
                    infer_fatal.append("没有可用帧")
                    stopped.set()
                    break
                try:
                    resp = cloud.infer(shared["state"], frame_b64, wrist_b64, args.instruction)
                except Exception as e:
                    if not stopped.is_set():
                        log.error("云端推理失败: %s", e)
                        infer_fatal.append(str(e))
                        stopped.set()
                    break
                act = clamp_joints(np.asarray(resp["predict_action"], dtype=np.float64)[:, :N_JOINTS])
                with cond:
                    ens.push(act, exec_frame=exec_frames["n"])
                    seq["n"] += 1
                    depth_now = ens.depth
                    cond.notify_all()
                st = resp.get("server_timing", {})
                log.info("推理 #%d ok  infer_ms=%.0f  rtt=%.0fms  buffer=%d",
                         seq["n"], st.get("infer_ms", float("nan")),
                         resp.get("rtt_ms", float("nan")), depth_now)
        except Exception as e:
            # 兜底：推理线程任何未预期异常（如 cv2.error）都不能静默杀死线程，
            # 否则主线程会在 wait_new_prediction 永久活锁。
            # 关机期间被 cam.close() 打断属预期，不误报错误。
            if not stopped.is_set():
                log.error("推理线程未预期异常: %r", e)
                infer_fatal.append(f"推理线程未预期异常: {e!r}")
                stopped.set()

    t_infer = threading.Thread(target=infer_loop, name="wallx-infer", daemon=True)
    t_infer.start()

    def seam_alpha(j: int) -> float:
        """接缝缓动：chunk 前 seam_frames 帧从本 chunk 起始实际位置缓动进入新轨迹，之后 1.0（不缓动）。"""
        if j < min(args.seam_frames, len(SEAM_ALPHAS)):
            return SEAM_ALPHAS[j]
        return 1.0

    reflex = GraspReflex()
    last_reflex_phase = None

    consumed = 0
    cur = shared["state"].copy()
    last_target = None
    round_no = 0
    chunk_dur = args.execute_interval * args.chunk_size
    t_start = time.time()   # --max-seconds 墙钟上限基准

    # ---- 计划复用（ACT 式顺序执行）----
    # 每轮对最新预测 argmin 重对齐的病灶：在「位置几乎静止」的相位（接近→夹取）里，
    # 很多帧离当前状态都近，噪声决定挑哪帧 → 每轮换相位 → 位置来回挪、夹爪开合反复。
    # 修法：一次对齐后顺着同一个 32 帧计划连续走完（约 2s），计划内部相位严格递增；
    # 只有跟踪误差过大（丢跟踪）或计划耗尽时才取新预测重新对齐。
    active_plan = None      # 当前执行的计划 [32, N_JOINTS] 或 None
    plan_pos = 0            # 计划内下一帧索引
    plan_start_pos = 0      # 本计划起始索引（首轮跳过跟踪恢复检查，允许从远处逐步跟上计划）
    REALIGN_ERR = 20.0      # 跟踪误差（6 关节 L2，归一化单位）超过此值且非首轮 → 放弃计划重新对齐

    def should_stop():
        if args.max_seconds > 0 and time.time() - t_start >= args.max_seconds:
            return True
        return stop_requested[0] or stopped.is_set()

    try:
        while not stopped.is_set():
            if args.frames > 0 and round_no >= args.frames:
                break
            if stop_requested[0]:
                stopped.set()
                break
            # ---- 计划起点：无活动计划时才等待新预测（计划内各 chunk 之间连续执行不等待）----
            if active_plan is None:
                waited = wait_new_prediction(cond, seq, consumed, should_stop)
                if args.max_seconds > 0 and time.time() - t_start >= args.max_seconds:
                    log.info("达到 --max-seconds %.0fs 上限，停止", args.max_seconds)
                    break
                if stop_requested[0]:
                    stopped.set()
                    break
                if stopped.is_set():
                    break
            state = arm.read_state()          # 实况位置：对齐与条件化共用
            shared["state"] = state
            # 每 chunk 从实际位置重锚（与旧版语义一致）：max-delta 相对实际位置限幅，
            # 防止命令链因舵机滞后越漂越超前；接缝也从实际位置缓动进入新轨迹
            cur = state.copy()
            last_target = state.copy()
            with cond:
                if active_plan is None:
                    if ens.depth == 0:
                        # 缓冲空 = 不变量破坏（正常由 wait_new_prediction 保证新预测已 push）
                        log.error("推理缓冲为空且无新预测，停机")
                        infer_fatal.append("推理缓冲为空")
                        stopped.set()
                        break
                    _, newest_act = ens.newest()
                    if waited > max(chunk_dur, 0.12) * 1.5:
                        # 推理空档超过 1.5 个 chunk（下限 0.18s）：空档期间手臂静止，旧预测基于过时观测，
                        # 只保留最新（帧计数对齐保证旧预测不被空档误推进，但它们的「计划」已过期）。
                        newest = ens.newest()
                        ens.clear()
                        ens.push(newest[1], exec_frame=newest[0])
                        log.info("推理空档 %.1fs，重置平滑缓冲（只用最新预测）", waited)
                    active_plan = np.asarray(newest_act, dtype=np.float64).copy()
                    # 计划轨迹滑动平均：抑制模型逐帧预测噪声（不改相位/不拖慢整体轨迹）
                    active_plan = smooth_plan(active_plan, args.plan_smooth)
                    plan_pos = int(np.argmin(np.linalg.norm(active_plan - state[None, :], axis=1)))
                    plan_start_pos = plan_pos
                    consumed = seq["n"]
                    depth_now = ens.depth
                else:
                    depth_now = ens.depth
                    # 跟踪恢复：计划与实况偏离过大且已执行过至少一个 chunk → 放弃本计划，下轮重对齐
                    if plan_pos > plan_start_pos and \
                            np.linalg.norm(active_plan[plan_pos] - state) > REALIGN_ERR:
                        log.info("跟踪误差过大（计划帧 %d/31 vs 实况），放弃计划重新对齐", plan_pos)
                        active_plan = None
                        continue
                # 取本 chunk 的 8 帧（越界补齐末帧 = 保持末端位姿）
                chunk = active_plan[plan_pos:plan_pos + args.chunk_size]
                if len(chunk) < args.chunk_size:
                    pad = np.repeat(chunk[-1:], args.chunk_size - len(chunk), axis=0)
                    chunk = np.vstack([chunk, pad])
                chunk_start_pos = plan_pos

            round_no += 1
            chunk_targets = []
            chunk_raw = []
            t_chunk = time.time()
            for j in range(args.chunk_size):
                # 急停/停机：chunk 内逐帧响应（写帧前检查），延迟 ≤1 帧节拍（--execute-interval）。
                # stopped 可能被推理线程置位（云端失败/相机掉线/线程死亡）——同样立即冻结
                if stop_requested[0] or stopped.is_set():
                    if stop_requested[0]:
                        stopped.set()
                    break
                t = chunk[j].copy()
                if args.bias_correct:
                    # 关节级偏差补偿（实测 bias 反向修正）：腕 +8.08 / 肘 -4.72 / 夹爪 -6.04 等
                    t = t - JOINT_BIAS
                if seam_alpha(j) < 1.0:
                    t = last_target + seam_alpha(j) * (t - last_target)
                if args.grasp_reflex:
                    live = arm.read_state()   # 每帧实况：夹取判定用真实位姿/夹爪
                    t, reflex_phase = reflex.step(live, t, args.max_delta, args.execute_interval)
                    if reflex_phase != last_reflex_phase:
                        if reflex_phase == "closing":
                            log.info("夹取判定触发（实况夹爪=%.1f ≤ %.0f 且位置接近）→ 冻结位置+夹紧",
                                     live[5], reflex.trigger_gripper)
                        elif reflex_phase == "carry":
                            log.info("夹紧到位（实况夹爪=%.1f）→ 搬运保持（位置跟随计划、夹爪保持夹紧）", live[5])
                        elif last_reflex_phase is not None:
                            log.info("夹取判定结束 → 恢复完全跟随计划（释放）")
                        last_reflex_phase = reflex_phase
                # NaN/Inf 检查必须在 max_delta clip 与 clamp_joints 之前：
                # 两者都会把 Inf 吸收成有限值、raw 转换会让 NaN 先抛 ValueError，
                # 只有在此处拦截才能兑现「NaN/Inf → 不执行」的停机语义
                if not np.isfinite(t).all():
                    log.error("目标含 NaN/Inf，停机保护（不写舵机）")
                    stopped.set()
                    break
                if args.max_delta > 0:
                    d = np.clip(t - cur, -args.max_delta, args.max_delta)
                    t = cur + d
                t = clamp_joints(t)
                raw = {
                    n: int(((tv + 100) / 200) * (arm.calib[n].range_max - arm.calib[n].range_min) + arm.calib[n].range_min)
                    if i < 5 else int((tv / 100) * (arm.calib[n].range_max - arm.calib[n].range_min) + arm.calib[n].range_min)
                    for i, (n, tv) in enumerate(zip(JOINT_ORDER, t))
                }
                arm.write_state(t)
                cur = t
                exec_frames["n"] += 1
                # 每帧把命令位置写回共享状态（引用替换原子、零总线读）：
                # 推理条件化滞后从 ~1 chunk 降到 ~1 帧，图像与关节状态失配最小化
                shared["state"] = cur.copy()
                chunk_targets.append(t)
                chunk_raw.append(raw)
                # 固定节拍（wall-clock 补时，自动扣除写入耗时）
                next_t = t_chunk + (j + 1) * args.execute_interval
                sleep_s = next_t - time.time()
                if sleep_s > 0 and not should_stop():
                    time.sleep(sleep_s)
            if not chunk_targets:
                log.warning("第 %d 轮未执行任何帧（急停/保护中断）", round_no)
                break
            # 计划指针只按实际执行的帧数推进；计划走完 → 下轮取新预测重新对齐
            plan_pos += len(chunk_targets)
            if plan_pos >= ACTION_HORIZON:
                active_plan = None
            log.info("─" * 78)
            log.info("第 %d 轮 连续执行 %d 帧 (计划帧 %d/31, buffer=%d, 接缝=%d帧, 节拍=%.0fms)",
                     round_no, len(chunk_targets), chunk_start_pos, depth_now,
                     min(args.seam_frames, len(SEAM_ALPHAS)), args.execute_interval * 1000)
            log.info("  起始状态: %s", np.array2string(state, precision=1, floatmode="fixed"))
            log.info("  首帧目标=%s", np.array2string(chunk_targets[0], precision=1, floatmode="fixed"))
            log.info("  末帧目标=%s", np.array2string(chunk_targets[-1], precision=1, floatmode="fixed"))
            log.info("  夹爪帧序列=%s", np.array2string(np.array([c[5] for c in chunk_targets]), precision=0))
            log.info("  末帧原始=%s", str(chunk_raw[-1]))
    finally:
        # 只有真正收到 Ctrl-C 才打印急停提示（其他停机原因不冒充急停）
        was_interrupted = stop_requested[0]
        stopped.set()
        t_infer.join(timeout=1)
        if t_infer.is_alive():
            log.warning("推理线程仍在退出（最长阻塞 %ss）；websocket/相机由进程退出兜底关闭", args.timeout)
        if infer_fatal:
            log.error("推理线程致命错误: %s", infer_fatal[0])
        if not t_infer.is_alive():
            # 推理线程已退出：安全释放 websocket 与相机（相机读超时 1.5s > join 1s，
            # 线程仍活时不关相机，避免释放与读帧并发）
            cloud.close()
            if cam is not None:
                cam.close()
            if wrist_cam is not None:
                wrist_cam.close()
        arm.disconnect()
        log.info("已断开。%s", "（Ctrl-C 急停后舵机保持当前位姿，未断电）" if was_interrupted else "")
    # 非零退出码：推理线程致命错误/停机保护触发过（脚本可据此感知失败）
    return 2 if infer_fatal else 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SO-ARM101 真机 ↔ 云端 wall-x serving 实时客户端")
    p.add_argument("--ws-url", default="ws://localhost:32195", help="云端 serving 地址（走 SSH 隧道）")
    p.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82111073-if00",
                   help="从臂串口（by-id 稳定路径，ttyACM 号会随插拔漂移）")
    p.add_argument("--calib", default=DEFAULT_CALIB, help="从臂标定文件（与录制时同一份）")
    p.add_argument("--camera", default="/dev/video10", help="global 相机（v4l2loopback 虚拟相机）")
    p.add_argument("--wrist-camera", default="/dev/v4l/by-id/usb-DHZJ-250122-ZW_W19_HD_Webcam-video-index0",
                   help="wrist 相机（W19 UVC）")
    p.add_argument("--image", default=None, help="离线帧文件（相机缺失时替代真机相机测试链路）")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="任务指令文本")
    p.add_argument("--frames", type=int, default=0, help="推理轮数（0=持续循环直到 Ctrl-C）")
    p.add_argument("--interval", type=float, default=0.0, help="每轮间隔秒数（仅 dry-run 生效；0=不等）")
    p.add_argument("--execute", action="store_true", help="【危险】真发舵机（默认 dry-run 只打印）")
    p.add_argument("--max-delta", type=float, default=5.0,
                   help="--execute 时每帧每关节相对上一帧的最大步进（归一化单位；0=不限幅；"
                        "默认 5.0，单轮累计上限 = max-delta × chunk-size）")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="--execute 总墙钟时间上限秒数（0=不限；云端停摆时的兜底，--frames 只计数已执行轮）")
    p.add_argument("--bias-correct", dest="bias_correct", action="store_true", default=True,
                   help="关节级系统偏差补偿（round4 10000步开环 60 样本实测 bias 反向修正；--no-bias-correct 关闭）")
    p.add_argument("--no-bias-correct", dest="bias_correct", action="store_false",
                   help=argparse.SUPPRESS)
    p.add_argument("--plan-smooth", type=int, default=3,
                   help="--execute 计划轨迹滑动平均窗口（帧数；默认 3 抑制计划内高频抖；0=关闭）")
    p.add_argument("--grasp-reflex", dest="grasp_reflex", action="store_true", default=True,
                   help="夹取判定状态机（判定夹取时刻→冻结位置+夹紧→搬运保持→位移后释放；"
                        "--no-grasp-reflex 关闭）")
    p.add_argument("--no-grasp-reflex", dest="grasp_reflex", action="store_false",
                   help=argparse.SUPPRESS)
    p.add_argument("--execute-interval", type=float, default=0.083,
                   help="--execute 固定节拍：每帧执行间隔秒数（wall-clock 补时）。"
                        "默认 0.083 = 12fps，即数据集/训练帧率（开源时序语义：wall-oss chunk 32 帧"
                        "= 32×(1/12fps)=2.67s 轨迹；wall-x serving 不插值 multiplier=1，"
                        "客户端按数据集 fps 执行）。轻柔慢动作可设 0.12~0.15（满幅度不变、只拉长时间）")
    p.add_argument("--chunk-size", type=int, default=8,
                   help="--execute 每 chunk 执行帧数")
    p.add_argument("--ensemble-history", type=int, default=1,
                   help="temporal ensembling 保留的预测条数（默认 1=只用最新预测、纯平滑；"
                        "2~3=多条预测按已执行帧计数对齐加权平均，实验用）")
    p.add_argument("--smooth-decay", type=float, default=0.5,
                   help="temporal ensembling 旧预测权重衰减系数（0~1；0=只用最新；"
                        "默认 0.5 强偏最新=安全优先。代价：场景突变时新方向稀释 ~1 chunk；"
                        "高交互/有人场景可用 --ensemble-history 1 或 --smooth-decay 0 退化）")
    p.add_argument("--seam-frames", type=int, default=3,
                   help="chunk 接缝缓动帧数（前 N 帧从本 chunk 起始实际位置缓动进入新轨迹；0=关闭接缝插值）")
    p.add_argument("--timeout", type=float, default=60.0, help="云端响应超时秒数")
    p.add_argument("--verbose", "-v", action="store_true", help="（仅 dry-run）打印动作前 3 帧")
    args = p.parse_args()
    # 参数边界校验（非法值会造成 NaN 崩溃/死循环/权重方向翻转，必须显式拒绝；
    # 浮点统一用 isfinite 检查，NaN/inf 能穿透普通的 < 0 / <= 0 比较）
    if args.ensemble_history < 1:
        p.error("--ensemble-history 必须 >= 1（1=只用最新预测，即旧行为）")
    if not (math.isfinite(args.smooth_decay) and 0.0 <= args.smooth_decay <= 1.0):
        p.error("--smooth-decay 必须在 0~1 之间（0=只用最新，1=新旧等权）")
    if args.chunk_size < 1:
        p.error("--chunk-size 必须 >= 1")
    if not (math.isfinite(args.execute_interval) and args.execute_interval >= 0):
        p.error("--execute-interval 必须 >= 0 的有限数")
    if args.seam_frames < 0:
        p.error("--seam-frames 必须 >= 0")
    if not (math.isfinite(args.max_delta) and args.max_delta >= 0):
        p.error("--max-delta 必须 >= 0 的有限数（0=不限幅）")
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        p.error("--timeout 必须 > 0 的有限数")
    if not (math.isfinite(args.interval) and args.interval >= 0):
        p.error("--interval 必须 >= 0 的有限数")
    if not (math.isfinite(args.max_seconds) and args.max_seconds >= 0):
        p.error("--max-seconds 必须 >= 0 的有限数（0=不限）")
    if args.plan_smooth < 0:
        p.error("--plan-smooth 必须 >= 0（0=关闭）")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.execute:
        log.warning("=" * 70)
        log.warning("  --execute 真机模式：将向从臂舵机写 Goal_Position！")
        log.warning("  请确认：手臂活动范围内无人/无物，随时可 Ctrl-C 急停。")
        log.warning("=" * 70)
    else:
        log.info("dry-run 模式：只打印动作，不碰舵机。")

    # 1. 从臂
    arm = FollowerArm(args.port, args.calib).connect()

    # 2. 相机（真机相机缺失时可用 --image 兜底）
    cam = None
    wrist_cam = None
    image_override = None
    try:
        if args.image:
            image_override = load_image_override(args.image)
            log.info("使用离线帧 %s（%d 字符 base64）", args.image, len(image_override))
        else:
            cam = Camera(args.camera)
            log.info("global 相机已就绪")
            wrist_cam = Camera(args.wrist_camera)
            log.info("wrist 相机已就绪")

        # 3. 云端
        cloud = WallxCloud(args.ws_url, args.timeout)
        meta = cloud.connect()
    except Exception:
        # 前置初始化失败也要归还从臂总线，不能跳过 disconnect
        arm.disconnect()
        raise

    stopped = threading.Event()
    # 信号处理器里不碰 threading.Event（其内部锁与主线程存在理论自死锁窗口），
    # 只置普通标志，主线程/执行循环统一转成 stopped.set()
    stop_requested = [False]

    def stop_handler(signum, frame):
        stop_requested[0] = True
        log.warning("收到 Ctrl-C，立即停止发送动作（舵机保持当前位姿）")

    import signal
    signal.signal(signal.SIGINT, stop_handler)

    if args.execute:
        # 流水线平滑执行（temporal ensembling + 接缝插值 + 固定节拍）
        return run_execute_pipelined(args, arm, cam, wrist_cam, cloud, image_override,
                                     stopped, stop_requested)

    # ---- dry-run：以下逻辑与原来完全一致 ----
    last_frame_b64 = image_override
    last_wrist_b64 = image_override
    cam_errors = 0
    round_no = 0
    dry_failed = [False]

    try:
        while not stopped.is_set() and not stop_requested[0]:
            if args.frames > 0 and round_no >= args.frames:
                break
            round_no += 1
            t_round = time.time()

            # --- 读状态 ---
            state = arm.read_state()
            t_state = (time.time() - t_round) * 1000

            # --- 读帧（global + wrist 双相机）---
            frame_ms = -1.0
            wrist_ms = -1.0
            if cam is not None:
                try:
                    last_frame_b64, frame_ms = cam.grab_b64_jpeg()
                    cam_errors = 0
                except RuntimeError as e:
                    cam_errors += 1
                    if last_frame_b64 is None:
                        log.error("没有可用帧且读帧失败（%s），退出。相机链需要恢复。", e)
                        return 2
                    log.warning("读帧失败（%s），复用上一帧（连续 %d 次）", e, cam_errors)
            if wrist_cam is not None:
                try:
                    last_wrist_b64, wrist_ms = wrist_cam.grab_b64_jpeg()
                except RuntimeError as e:
                    log.warning("wrist 读帧失败（%s），复用上一帧", e)
            if last_frame_b64 is None or last_wrist_b64 is None:
                log.error("没有可用帧。")
                return 2

            # --- 云端推理（dry-run 用 15s 收包超时：覆盖 serving 首次预热 ~5-10s，
            # 同时 Ctrl-C 最多 ~15s 内生效，不用等 60s）---
            try:
                resp = cloud.infer(state, last_frame_b64, last_wrist_b64, args.instruction,
                                   recv_timeout=15.0)
            except Exception as e:
                log.error("云端推理失败: %s", e)
                dry_failed[0] = True
                break

            action = resp["predict_action"]
            st = resp["server_timing"]
            infer_ms = st.get("infer_ms", float("nan"))

            # --- 动作处理：dry-run 单帧打印（--execute 已走流水线 run_execute_pipelined）---
            target, chosen = select_and_clamp(action, state, 0.0)
            delta_norm = target - state
            log.info("─" * 78)
            log.info("第 %d 轮  状态: %s", round_no, np.array2string(state, precision=1, floatmode="fixed"))
            log.info("帧: %s  base64=%d字符  (读帧 %.1fms | 状态读取 %.1fms)",
                     "真机相机" if cam is not None else "离线帧", len(last_frame_b64), frame_ms, t_state)
            log.info("云端: infer_ms=%.0f  客户端RTT=%.0fms%s", infer_ms, resp["rtt_ms"],
                     f"  prev_total_ms={st['prev_total_ms']:.0f}" if "prev_total_ms" in st else "")
            log.info("动作[0,:6]=%s", np.array2string(action[0, :N_JOINTS], precision=1, floatmode="fixed"))
            if args.verbose:
                log.info("动作前3帧=\n%s", np.array2string(
                    np.asarray(action, dtype=np.float64)[:3, :N_JOINTS], precision=1, floatmode="fixed"))
            log.info("执行帧 index=%d/31  目标=%s", chosen, np.array2string(target, precision=1, floatmode="fixed"))
            log.info("相对步进=%s", np.array2string(delta_norm, precision=2, floatmode="fixed"))
            log.info("  [dry-run] 不写舵机。加 --execute 才真机执行。")

            if args.interval > 0 and not stopped.is_set() and not stop_requested[0]:
                time.sleep(args.interval)
    finally:
        cloud.close()
        if cam is not None:
            cam.close()
        if wrist_cam is not None:
            wrist_cam.close()
        arm.disconnect()
        log.info("已断开。%s", "（Ctrl-C 急停后舵机保持当前位姿，未断电）" if stop_requested[0] else "")
    return 2 if dry_failed[0] else 0


if __name__ == "__main__":
    sys.exit(main())
