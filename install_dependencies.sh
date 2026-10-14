#!/usr/bin/env bash
# install_dependencies.sh — SO-101 wall-oss 真机部署一键依赖安装
#
# 幂等、可重跑。装的是「本地真机部署」依赖（客户端 + 相机链 + 机械臂驱动），
# 不含 torch（客户端只用 lerobot.motors 串口模块，不涉及训练）。
# 云端训练环境已在 AutoDL 实例就绪，见 docs/训练流程.md。
#
# 用法：bash install_dependencies.sh
set -euo pipefail

GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="lerobot"

echo "=== SO-101 wall-oss 真机部署依赖安装（仓库：${REPO_DIR}）==="
echo

# ── 1. conda（非交互 shell 可能不在 PATH，按常见路径找）────
CONDA_CMD=""
if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [ -x "$c" ]; then CONDA_CMD="$c"; break; fi
  done
fi
if [ -z "$CONDA_CMD" ]; then
  fail "未找到 conda。请先安装 Miniconda：https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
ok "conda 已找到：${CONDA_CMD}"
CONDA_BASE="$("${CONDA_CMD}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── 2. env lerobot（Python 3.10）────────────────────────────
if conda env list | grep -qE "(^|\s)${ENV_NAME}($|\s|\*)"; then
  ok "conda env '${ENV_NAME}' 已存在"
else
  echo "  创建 conda env '${ENV_NAME}' (python 3.10)..."
  conda create -n "${ENV_NAME}" python=3.10 -y
  ok "env '${ENV_NAME}' 创建完成"
fi
conda activate "${ENV_NAME}"

# ── 3. Python 运行依赖 ──────────────────────────────────────
echo "  安装 Python 运行依赖（opencv/numpy/msgpack/websockets/serial...）"
pip install -q opencv-python numpy msgpack msgpack-numpy websockets pyyaml pyserial deepdiff tqdm
ok "Python 运行依赖安装完成"

# ── 4. lerobot fork（只装 motors/robots 模块，不装 torch）──
if python -c "from lerobot.motors.feetech import FeetechMotorsBus" >/dev/null 2>&1; then
  ok "lerobot fork（SO-101 电机驱动）已安装"
else
  echo "  安装 lerobot fork（--no-deps，避免拖入 torch）..."
  pip install -q -e "${REPO_DIR}/wallx/lerobot" --no-deps
  ok "lerobot fork 安装完成"
fi

# ── 5. ROS2 humble（相机链需要）─────────────────────────────
if [ -d /opt/ros/humble ]; then
  ok "ROS2 humble 已安装"
else
  warn "ROS2 humble 未安装（跑相机链才需要）。安装："
  warn "  sudo apt update && sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions"
  warn "  （Ubuntu 22.04 + ROS2 apt 源，见 https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html）"
fi

# ── 6. v4l2loopback /dev/video10 ────────────────────────────
if [ -e /dev/video10 ]; then
  ok "/dev/video10 已存在"
else
  warn "/dev/video10 不存在，尝试创建（需 sudo）..."
  if sudo modprobe v4l2loopback video_nr=10 card_label="so101_global" exclusive_caps=1 2>/dev/null; then
    ok "/dev/video10 创建成功"
  else
    warn "modprobe 失败，手动：sudo apt install -y v4l2loopback-utils && sudo modprobe v4l2loopback video_nr=10 card_label=so101_global exclusive_caps=1"
  fi
fi

# ── 7. ascamera SDK 编译 ────────────────────────────────────
if [ -d /opt/ros/humble ]; then
  ASC_DIR="${REPO_DIR}/camera"
  if [ -d "${ASC_DIR}/install/ascamera" ]; then
    ok "ascamera 已编译（install/ascamera 已存在）"
  else
    echo "  编译 ascamera ROS2 包（colcon）..."
    if ( cd "${ASC_DIR}" && set +u && source /opt/ros/humble/setup.bash && set -u && colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF ); then
      ok "ascamera 编译完成"
    else
      warn "ascamera 编译失败，检查 ROS2 环境后重跑本脚本"
    fi
  fi
fi

# ── 8. 标定文件（放到 lerobot 缓存目录）─────────────────────
CAL_BASE="$HOME/.cache/huggingface/lerobot/calibration"
mkdir -p "${CAL_BASE}/robots/so101_follower" "${CAL_BASE}/teleoperators/so101_leader"
cp -n "${REPO_DIR}/robot/my_awesome_follower_arm.json" "${CAL_BASE}/robots/so101_follower/" \
  && ok "从臂标定已就位（robots/so101_follower/）"
cp -n "${REPO_DIR}/robot/my_awesome_leader_arm.json" "${CAL_BASE}/teleoperators/so101_leader/" \
  && ok "主臂标定已就位（teleoperators/so101_leader/）"

# ── 9. SSH 隧道提示 ─────────────────────────────────────────
warn "SSH 隧道：从队友处拿到云端 <host>:<port> 后执行："
warn "  ssh -N -f -L 32195:localhost:32195 -o ServerAliveInterval=30 root@<host> -p <port>"

# ── 10. 自检 ────────────────────────────────────────────────
echo
echo "=== 自检 ==="
python -c "import lerobot.motors" >/dev/null 2>&1 \
  && ok "lerobot.motors 可导入" || fail "lerobot.motors 导入失败"
echo "  相机设备（应含 /dev/video10 和 W19 的 by-id）："
ls /dev/video* 2>/dev/null | sed 's/^/    /' || echo "    （无）"
echo "  串口设备（应含 ...72 主臂 / ...73 从臂）："
ls /dev/serial/by-id/ 2>/dev/null | sed 's/^/    /' || echo "    （无，机械臂未接）"

echo
echo "=== 完成 ==="
echo "接好机械臂 USB + ASC60C + W19 后，先 dry-run 验证："
echo "  python ${REPO_DIR}/client/wallx_realtime_client.py"
echo "真机执行："
echo "  python ${REPO_DIR}/client/wallx_realtime_client.py --execute"
