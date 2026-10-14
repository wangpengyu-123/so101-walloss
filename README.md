# SO-101 wall-oss 真机抓取

用 **X-Square-Robot wall-oss-0.5** 基础模型，在自采 SO-ARM101 数据上做 LoRA 微调，实现「**抓取红色方块并放到目标盒**」的真机实时推理。

- 模型：Qwen2.5-VL-3B + MoE attention + flow-matching 动作头（26 维 = 6 维关节 + 20 padding）
- 训练：71 集自采数据，30001 步 LoRA 微调（实际 ~5000 步已收敛）
- 推理：真机客户端 → SSH 隧道 → 云端 serving（WebSocket + msgpack）

## 架构

```
[本地真机]                                            [云端 AutoDL]
 ASC60C 相机 ──ROS2──► /dev/video10 ──┐
 W19 腕相机 (by-id) ─────────────────┤
 SO-ARM101 从臂 (serial, ...73) ─────┤
                                     ├─► client/wallx_realtime_client.py
                                     │        │ 相机帧 + 6关节状态 (msgpack)
                                     │        ▼
                                     │   SSH 隧道 :32195 ──► wall-x serving
                                     │                         │ round4 模型推理
                                     │                         ▼
                                     │◄──── 动作[32,26] 取前6维 ──┘
                                     └─► 写回从臂舵机
```

数据流（训练侧）：

```
录制(v2.1) → compute_norm_stats(6维) → train_fsdp(LoRA, 153key) → merge_v2(防丢断言) → serving → 开环评估
```

## 仓库结构

```
├── README.md                  本文件
├── install_dependencies.sh    ⭐ 一键装依赖（队友跑这一句就绪）
├── 指导性.md                  从零到跑通的主文档
├── client/                    真机客户端 + 平滑/夹取判定逻辑
├── camera/                    相机链（ascamera SDK + ros2 桥接）
├── robot/                     机械臂标定文件（从臂+主臂）
├── wallx/                     wall-x 补丁源码（含 lerobot 0.3.4 fork）
├── configs/                   各轮训练配置 + norm_stats
├── scripts/                   训练/合并/评估脚本
├── patches/                   LoRA + serving 补丁
└── docs/                      衔接文档（架构/环境/云端/真机/训练/录制/FAQ）
```

## 快速上手（队友三行）

```bash
# 1. 下载仓库
git clone <本仓库地址>
# 2. 一键装依赖
bash so101-walloss/install_dependencies.sh
# 3. 从队友处拿到云端 SSH，建隧道后跑客户端
ssh -N -f -L 32195:localhost:32195 root@<host> -p <port>
python so101-walloss/client/wallx_realtime_client.py   # dry-run 验证
python so101-walloss/client/wallx_realtime_client.py --execute   # 真机执行
```

详细步骤见 **[指导性.md](指导性.md)**。

## 文档导航

| 想做什么 | 看哪篇 |
|---|---|
| 从零跑通真机模型 | [指导性.md](指导性.md) |
| 理解各组件怎么衔接 | [docs/架构总览.md](docs/架构总览.md) |
| 装依赖 | [install_dependencies.sh](install_dependencies.sh) + [docs/环境搭建.md](docs/环境搭建.md) |
| 云端 serving / SSH 隧道 | [docs/云端部署.md](docs/云端部署.md) |
| 相机链 / 机械臂 / 客户端参数 | [docs/真机部署.md](docs/真机部署.md) |
| 复现训练 | [docs/训练流程.md](docs/训练流程.md) |
| 录数据 | [docs/数据录制协议.md](docs/数据录制协议.md) |
| 踩坑 | [docs/常见问题.md](docs/常见问题.md) |

## 版权说明

- **wall-x**：本仓库的 `wallx/` 是 [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x) 的派生（含 wall-oss-0.5 训练/推理代码 + HuggingFace lerobot 0.3.4 fork），遵循其开源许可。
- **ascamera SDK**（`camera/ascamera/`）：来自 Angstrong（ASC60C 相机厂商），版权归 Angstrong 所有，仅用于本项目内部协作。
- **模型权重**：`wall-oss-0.5` + `Qwen2.5-VL-3B-Instruct` 从 HuggingFace 下载（训练时用，本仓库不含权重文件）。
- 本仓库只含工程内容，**不含任何 SSH 密钥、密码、API token**。云端 SSH 地址由队友单独提供。
