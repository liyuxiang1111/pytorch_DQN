'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
modified from (batch, h, w, ch) to (batch, ch, h, w)
'''

import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """辅助函数。给定一个返回可比较对象的函数 `sampling_f`，从中采样 n 个唯一对象。"""
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """
        初始化 ReplayBuffer，使用内存高效的方式存储经验。

        参数:
        size: int
        缓冲区的最大容量。当缓冲区满时，旧数据将被覆盖。
        frame_history_len: int
        每个观察包含的历史帧数。
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0 # 下一个要存储的索引
        self.num_in_buffer = 0 # 当前缓冲区中的经验数

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """检查是否可以从缓冲区中采样 `batch_size` 个不同的经验。"""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        """
        根据索引列表 `idxes` 编码样本。

        返回:
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
        """
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        # 该函数通过索引 idxes 获取所需的经验（状态、动作、奖励、下一个状态和完成标志），并返回这些值组成的批次。
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):

        """
        从缓冲区中采样 `batch_size` 个不同的经验。
        返回:
        - `obs_batch`: 当前状态
        - `act_batch`: 动作
        - `rew_batch`: 奖励
        - `next_obs_batch`: 下一个状态
        - `done_mask`: 完成标志
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """
        返回最近的 `frame_history_len` 帧，用于构造当前观察。
        返回:
        observation: 最近观察（np.array）
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        """
        根据索引 `idx` 编码观察。
        """
        end_idx   = idx + 1 # 设置为非包含
        start_idx = end_idx - self.frame_history_len

        # 检查是否使用低维度观察（例如 RAM 状态）
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]

        # 如果缓冲区中没有足够的帧
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1

        # 计算缺失的历史帧数
        missing_context = self.frame_history_len - (end_idx - start_idx)

        # 如果需要零填充或位于缓冲区边界
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0) # 返回格式为 (c, h, w)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w c
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        """
        将一帧图像存储在缓冲区的下一个可用索引中，如果必要会覆盖旧的帧。

        参数:
        frame: np.array
            要存储的帧，形状为 (img_h, img_w, img_c)

        返回:
        idx: int
            该帧的存储索引，供 `store_effect` 使用。
        """
        # 如果观察为图像
        if len(frame.shape) > 1:
            # 将图像帧从 h, w, c 转置为 c, h, w
            frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """
        存储在 `store_frame` 之后采取的动作的效果。

        参数:
        idx: int
            最近观察帧在缓冲区中的索引（由 `store_frame` 返回）。
        action: int
            执行的动作。
        reward: float
            执行动作后得到的奖励。
        done: bool
            如果动作结束了当前回合，则为 True。
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
