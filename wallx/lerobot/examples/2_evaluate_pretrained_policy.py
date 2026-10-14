# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
本脚本演示如何评测一个预训练策略：可以从 HuggingFace Hub 加载，也可以从你本地的
训练输出目录加载。若是后者，你可能需要先运行 examples/3_train_policy.py。

它需要安装 'gym_pusht' 仿真环境。安装方法：
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# 创建一个目录，用来存放评测的视频
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# 选择你的设备
device = "cuda"

# 提供 [Hugging Face 仓库 id](https://huggingface.co/lerobot/diffusion_pusht)：
pretrained_policy_path = "lerobot/diffusion_pusht"
# 或者一个指向本地 outputs/train 文件夹的路径。
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

# 初始化评测环境，让它渲染两种观测：
# 场景图像，以及 agent 的状态/位置。环境还会在 300 次交互/步之后自动停止。
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# 我们可以验证：策略期望的 feature 形状，和环境产生的观测形状是否匹配
print(policy.config.input_features)
print(env.observation_space)

# 类似地，我们可以检查：策略产生的动作，和环境期望的动作是否匹配
print(policy.config.output_features)
print(env.action_space)

# 重置策略和环境，为 rollout（一次完整的执行过程）做准备
policy.reset()
numpy_observation, info = env.reset(seed=42)

# 准备收集这一集的所有奖励和所有帧，
# 从初始状态到最终状态。
rewards = []
frames = []

# 渲染初始状态的那一帧
frames.append(env.render())

step = 0
done = False
while not done:
    # 为运行在 PyTorch 里的策略准备观测
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # 转成 float32，并把图像从 [0,255] 的通道优先（channel first）
    # 变成 [0,1] 的通道靠后（channel last）
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # 把数据张量从 CPU 搬到 GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # 加一个（空的）batch 维度，这是策略前向传播所必需的
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # 创建策略的输入字典
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # 根据当前观测，预测下一步动作
    with torch.inference_mode():
        action = policy.select_action(observation)

    # 为环境准备动作
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # 在环境里执行一步，得到新的观测
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # 记录所有奖励和帧
    rewards.append(reward)
    frames.append(env.render())

    # 当到达成功状态（即 terminated 为 True），或达到最大迭代次数
    # （即 truncated 为 True）时，认为这次 rollout 结束
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# 获取环境的速度（即它的每秒帧数）。
fps = env.metadata["render_fps"]

# 把所有帧编码成一个 mp4 视频。
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
