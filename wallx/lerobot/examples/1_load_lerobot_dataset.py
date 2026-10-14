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
本脚本演示如何使用 `LeRobotDataset` 类来处理 Hugging Face 上的机器人数据集。
展示如何加载数据集、操作它，并应用适合 PyTorch 机器学习任务的变换。

脚本包含以下功能：
- 查看数据集的元数据（metadata）并探索其属性
- 从 Hub 加载已有数据集，或它的一个子集
- 按 episode 编号访问帧
- 使用高级特性，比如基于时间戳的帧选择
- 演示与 PyTorch DataLoader 的兼容性，用于批量处理

脚本最后用 PyTorch 的 DataLoader 演示了如何批量处理数据。
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# 我们已经自己迁移了一批数据集，用这行查看列表：
print("List of available datasets:")
pprint(lerobot.available_datasets)

# 你也可以用 hub api 浏览社区在 Hub 上创建/迁移的数据集：
hub_api = HfApi()
repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
pprint(repo_ids)

# 或者直接在浏览器里浏览：
# https://huggingface.co/datasets?other=LeRobot

# 这个例子用这个数据集
repo_id = "lerobot/aloha_mobile_cabinet"
# 我们可以看一下并获取它的元数据，了解更多信息：
ds_meta = LeRobotDatasetMetadata(repo_id)

# 只实例化这个类，你就能快速获取数据集的内容和结构信息，
# 而不必下载实际数据（只下载轻量的元数据文件）。
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
pprint(ds_meta.features)

# 你也可以直接打印对象，得到一份简短摘要：
print(ds_meta)

# 然后你可以从 Hub 加载实际数据集。
# 可以加载任意一个 episode 子集：
dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])

# 看看你有多少帧：
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# 或者直接加载整个数据集：
dataset = LeRobotDataset(repo_id)
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# 前面的元数据类就包含在数据集的 'meta' 属性里：
print(dataset.meta)

# LeRobotDataset 实际上封装了一个底层的 Hugging Face 数据集
# （更多信息见 https://huggingface.co/docs/datasets）。
print(dataset.hf_dataset)

# LeRobot 数据集也是 PyTorch 数据集的子类，所以你熟悉 PyTorch 数据集的那些操作
# 都能用，比如遍历数据集。
# __getitem__ 遍历数据集的帧。因为我们的数据集还按 episode 组织，
# 你可以用 episode_data_index 访问任意 episode 的帧索引。这里访问第一集的帧索引：
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# 然后我们从第一个相机抓取所有图像帧：
camera_key = dataset.meta.camera_keys[0]
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# 数据集返回的对象都是 torch.Tensor
print(type(frames[0]))
print(frames[0].shape)

# 因为我们用的是 pytorch，shape 是 pytorch 的通道优先约定 (c, h, w)。
# 我们可以把这个 shape 和该 feature 的信息对比：
pprint(dataset.features[camera_key])
# 具体来说：
print(dataset.features[camera_key]["shape"])
# shape 是 (h, w, c)，这是一种更通用的格式。

# 很多机器学习应用需要加载过去观测的历史，或未来动作的轨迹。
# 我们的数据集可以针对每个键/模态，用与当前帧的时间戳差值来加载过去和未来的帧。比如：
delta_timestamps = {
    # 加载 4 张图：当前帧前 1 秒、前 500 毫秒、前 200 毫秒，以及当前帧
    camera_key: [-1, -0.5, -0.20, 0],
    # 加载 6 个状态向量：前 1.5 秒、前 1 秒……前 200 毫秒、前 100 毫秒，以及当前帧
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    # 加载 64 个动作向量：当前帧、未来 1 帧、未来 2 帧……未来 63 帧
    "action": [t / dataset.fps for t in range(64)],
}
# 注意：无论如何，这些 delta_timestamps 值需要是 (1/fps) 的倍数，
# 这样加到任意时间戳上，仍能得到合法的时间戳。

dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
print(f"\n{dataset[0][camera_key].shape=}")  # (4, c, h, w)
print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

# 最后，我们的数据集完全兼容 PyTorch 的 dataloader 和 sampler，
# 因为它们本来就是 PyTorch 数据集。
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
    print(f"{batch['action'].shape=}")  # (32, 64, c)
    break
