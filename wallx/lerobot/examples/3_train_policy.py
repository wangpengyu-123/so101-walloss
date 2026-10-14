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

"""本脚本演示如何在 PushT 环境上训练 Diffusion Policy。

用这个脚本训练出模型后，你可以试着用 examples/2_evaluate_pretrained_policy.py 评测它。
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    # 创建一个目录，用来存放训练 checkpoint。
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # 选择你的设备
    device = torch.device("cuda")

    # 离线训练步数（这个例子我们只做离线训练）。
    # 按你的喜好调整。要得到值得评测的结果，需要 5000 步。
    training_steps = 5000
    log_freq = 1

    # 从零开始训练（即不是从预训练策略开始）时，在创建策略前需要指定两样东西：
    #   - 输入/输出形状：用于给策略正确设定尺寸
    #   - 数据集统计量：用于输入/输出的归一化和反归一化
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 策略用一个配置类初始化，这里是 `DiffusionConfig`。这个例子里我们只用默认值，
    # 所以除了输入/输出特征外，不需要传其他参数。
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # 现在我们可以用这个配置和数据集统计量来实例化策略。
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # 策略和数据集之间还有一个交互，就是 delta_timestamps。每个策略期望一定数量的帧，
    # 输入、输出和奖励（如果有的话）的帧数可能各不相同。
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # 在 Diffusion Policy 的标准配置下，它等价于下面这样：
    delta_timestamps = {
        # 加载当前帧前 0.1 秒的上一张图像和状态，
        # 再加载当前帧对应的图像和状态（0.0 秒）。
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # 加载上一个动作（-0.1）、即将执行的下一个动作（0.0），
        # 以及未来 14 个动作，间隔 0.1 秒。这些动作都会被用来监督策略。
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # 然后我们可以用这些 delta_timestamps 配置来实例化数据集。
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

    # 然后为离线训练创建优化器和 dataloader。
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 运行训练循环。
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # 保存一个策略 checkpoint。
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
