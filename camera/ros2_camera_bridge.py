#!/usr/bin/env python3
"""
ros2_camera_bridge.py — 把 ROS2 图像话题桥接到 v4l2loopback 虚拟相机 (/dev/video10)

用途:
    安思疆 ASC60C 是 ROS2 SDK 相机（话题输出），而 lerobot 的 opencv 相机只能
    读 /dev/video*。本脚本订阅 RGB 话题 (bgr8)，转换为 RGB24 后直接通过
    V4L2 read/write 模式写入 v4l2loopback 设备，供 lerobot / ffplay / OpenCV 读取。

写设备方式:
    直接 ioctl(VIDIOC_S_FMT) + write()（V4L2 READWRITE 模式）。
    不用 cv2.VideoWriter —— 本机 OpenCV 5.0 的 wheel 不支持 V4L2 写设备；
    不用 pyfakewebcam / ffmpeg —— 少一层依赖。实测最稳。

前置:
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="ASC60C_RGB" exclusive_caps=1
    （本机已配置开机自动加载 /etc/modules-load.d/v4l2loopback.conf + /etc/modprobe.d/v4l2loopback.conf）

用法:
    # 真机（先 ros2 launch ascamera ascamera.launch.py）
    python3 ros2_camera_bridge.py
    # 无相机自检（合成测试图）
    python3 ros2_camera_bridge.py --synthetic
    # 自定义
    python3 ros2_camera_bridge.py --topic /ascamera/camera_publisher/rgb0/image \
                                  --device /dev/video10 --width 640 --height 480

注意:
    - exclusive_caps=1 时，v4l2loopback 只有在有写入者打开后才向读取端
      (lerobot/OpenCV) 报告 Video Capture 能力。本脚本启动即打开设备并
      STREAMON，所以启动顺序必须是: 先本脚本，后 lerobot record。
    - 未收到 ROS 帧时默认写"WAITING"占位帧，避免读取端阻塞在 read() 上。
"""

import argparse
import fcntl
import logging
import os
import signal
import struct
import sys
import time

import cv2
import numpy as np

log = logging.getLogger("ros2_camera_bridge")

# ---- V4L2 常量（kernel 6.8: v4l2_format 中 union 8 字节对齐，pix 偏移 8）----
VIDIOC_S_FMT = 0xC0D05605          # _IOWR('V', 5, struct v4l2_format)
VIDIOC_STREAMON = 0x40045612       # _IOW('V', 18, int)
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 1
RGB3 = ord("R") | (ord("G") << 8) | (ord("B") << 16) | (ord("3") << 24)  # V4L2_PIX_FMT_RGB24


class V4L2LoopbackWriter:
    """以 V4L2 READWRITE 模式向 v4l2loopback 设备写 RGB24 帧。"""

    def __init__(self, device: str, width: int, height: int):
        self.device = device
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.fd = None
        self._open()

    def _open(self):
        # O_NONBLOCK: 无消费者（或消费慢）时 write() 返回 EAGAIN 丢帧，而不是卡住桥接
        self.fd = os.open(self.device, os.O_WRONLY | os.O_NONBLOCK)
        # struct v4l2_format = u32 type + [4 字节对齐 padding] + v4l2_pix_format
        # v4l2_pix_format: width,height,pixelformat,field,bytesperline,sizeimage,
        #                  colorspace,priv,flags,ycbcr_enc,quantization,xfer_func
        fmt = struct.pack(
            "I4x" + "I" * 12,
            V4L2_BUF_TYPE_VIDEO_OUTPUT,
            self.width, self.height, RGB3, V4L2_FIELD_NONE,
            self.width * 3, self.frame_size, V4L2_COLORSPACE_SRGB,
            0, 0, 0, 0, 0,
        )
        # 补齐到 sizeof(struct v4l2_format)=208 字节，防止内核回写越界
        fmt = fmt.ljust(208, b"\x00")
        fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
        # STREAMON 之后 exclusive_caps 设备才对读取端报告 Video Capture
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("I", V4L2_BUF_TYPE_VIDEO_OUTPUT))
        log.info("已打开 %s: %dx%d RGB24 (fd=%d)", self.device, self.width, self.height, self.fd)

    def write_rgb(self, rgb: np.ndarray) -> bool:
        """写入一帧 HxWx3 uint8 RGB 图像。返回 False 表示帧被丢弃（无消费者/消费慢）。"""
        rgb = np.ascontiguousarray(rgb)
        assert rgb.shape == (self.height, self.width, 3) and rgb.dtype == np.uint8
        try:
            os.write(self.fd, rgb.tobytes())
            return True
        except BlockingIOError:
            return False
        except OSError:
            raise

    def reopen(self):
        """设备异常（如被 rmmod 后重建）时重开。"""
        try:
            self.close()
        except OSError:
            pass
        self._open()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def make_placeholder(width: int, height: int) -> np.ndarray:
    """等待相机数据时的占位帧（RGB）。"""
    img = np.zeros((height, width, 3), np.uint8)
    img[10:40, 10:width - 10] = (60, 60, 60)  # RGB 灰条
    cv2.putText(img, "WAITING FOR CAMERA...", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return img


def bgr8_msg_to_rgb(msg, width: int, height: int) -> np.ndarray:
    """sensor_msgs/Image(bgr8) -> HxWx3 RGB（照抄 save_camera_frame.py 的解析写法）。"""
    bgr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return rgb


class Bridge:
    """ROS2 订阅 + v4l2loopback 写入的桥接核心（与 ROS 解耦，便于 --synthetic 自检）。"""

    def __init__(self, args):
        self.args = args
        self.writer = V4L2LoopbackWriter(args.device, args.width, args.height)
        self.latest = None           # 最新一帧 RGB
        self.latest_seq = 0          # 最新帧序号（只有新帧到达才重写设备，保证设备帧率=相机帧率）
        self.last_written_seq = -1
        self.placeholder = make_placeholder(args.width, args.height)
        self.got_first_frame = False
        self.stats = {"recv": 0, "written": 0, "dropped": 0}
        self.t_last_stats = time.time()
        self.t_last_placeholder = 0.0

    def set_latest(self, msg):
        try:
            self.latest = bgr8_msg_to_rgb(msg, self.args.width, self.args.height)
            self.latest_seq += 1
            self.stats["recv"] += 1
            if not self.got_first_frame:
                self.got_first_frame = True
                log.info("收到第一帧 ROS 图像 (%dx%d, encoding=%s)", msg.width, msg.height, msg.encoding)
        except Exception:
            log.exception("解析 ROS 图像失败，跳过该帧")

    def step(self) -> None:
        """写一帧到 v4l2loopback。新 ROS 帧到达才写；无数据时按低频写占位帧。"""
        now = time.time()
        if self.latest is not None and self.latest_seq != self.last_written_seq:
            frame, source = self.latest, "ros"
        elif self.latest is None and not self.args.no_placeholder \
                and now - self.t_last_placeholder >= 1.0 / self.args.placeholder_fps:
            frame, source = self.placeholder, "placeholder"
        else:
            return

        try:
            ok = self.writer.write_rgb(frame)
        except OSError:
            log.error("写 %s 失败，尝试重开设备...", self.args.device)
            self.writer.reopen()
            return

        if ok:
            self.stats["written"] += 1
        else:
            self.stats["dropped"] += 1

        if source == "ros":
            self.last_written_seq = self.latest_seq
        else:
            self.t_last_placeholder = now

        if now - self.t_last_stats >= 5.0:
            self.t_last_stats = now
            log.info(
                "统计: 收到=%d 写入=%d 丢帧=%d (丢帧>0 说明没有消费者在读 %s)",
                self.stats["recv"], self.stats["written"], self.stats["dropped"],
                self.args.device,
            )


def run_ros(args):
    """ROS2 模式: 订阅话题 -> 桥接。"""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    rclpy.init(args=["ros2_camera_bridge"])
    node = Node("ros2_camera_bridge")
    bridge = Bridge(args)

    def cb(msg: Image):
        bridge.set_latest(msg)

    node.create_subscription(Image, args.topic, cb, 5)
    log.info("订阅话题 %s 中（相机未出图时会持续写占位帧）...", args.topic)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            bridge.step()
    finally:
        bridge.writer.close()
        node.destroy_node()
        rclpy.shutdown()
        log.info("已退出")


def run_synthetic(args):
    """合成帧自检模式: 不依赖 ROS/相机，验证 /dev/videoN -> 读取端整条链路。"""
    bridge = Bridge(args)
    log.info("合成帧模式: 向 %s 写红色底+移动白条测试图 @%dfps，Ctrl+C 退出", args.device, args.fps)
    buf = np.zeros((args.height, args.width, 3), np.uint8)
    i = 0
    t0 = time.time()
    try:
        while True:
            buf[:, :, :] = 0
            buf[:, :, 0] = 255                      # RGB 红底
            x = (i * 8) % args.width
            buf[:, x:x + 60, :] = 255               # 移动白条
            cv2.putText(buf, f"SYNTH {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            bridge.writer.write_rgb(buf)
            i += 1
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.writer.close()
        log.info("合成帧模式退出（共写入 %d 帧）", i)


def main():
    ap = argparse.ArgumentParser(description="ROS2 图像话题 -> v4l2loopback 虚拟相机桥接")
    ap.add_argument("--topic", default="/ascamera/camera_publisher/rgb0/image",
                    help="ROS2 RGB 图像话题 (默认 /ascamera/camera_publisher/rgb0/image)")
    ap.add_argument("--device", default="/dev/video10", help="v4l2loopback 设备路径 (默认 /dev/video10)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0, help="合成帧模式的输出帧率（ROS 模式由话题频率决定）")
    ap.add_argument("--placeholder-fps", type=float, default=2.0, help="无 ROS 数据时占位帧频率")
    ap.add_argument("--no-placeholder", action="store_true", help="不写占位帧（读取端在无数据时会阻塞）")
    ap.add_argument("--synthetic", action="store_true", help="自检模式：写合成测试图，不订阅 ROS")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.synthetic:
        run_synthetic(args)
    else:
        try:
            run_ros(args)
        except ImportError:
            log.error("缺少 rclpy：请先 source /opt/ros/humble/setup.bash")


if __name__ == "__main__":
    main()
