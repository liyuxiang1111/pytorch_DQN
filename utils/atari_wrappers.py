'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/atari_wrappers.py
'''

import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces


# NoopResetEnv 环境包装类：在重置环境时随机进行一些空操作以多样化初始状态
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """
        使用随机数量的空操作来采样初始状态。
        假设空操作对应的动作为 0。
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """在重置时执行 1 到 noop_max 个空操作。"""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0) # 执行空操作
        return obs # 返回新的观察值

# FireResetEnv 环境包装类：在重置环境时执行触发动作，用于固定需要触发的环境
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """对需要触发的环境在重置时执行动作。"""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE' # 确保支持触发动作
        assert len(env.unwrapped.get_action_meanings()) >= 3 # 确保有多个可用动作

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1) # 执行触发动作
        obs, _, _, _ = self.env.step(2) # 执行另一个动作以开始游戏
        return obs # 返回新的观察值

# EpisodicLifeEnv 环境包装类：在角色失去生命时结束回合
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """使生命耗尽等价于结束回合，但只有在游戏真正结束时才重置。"""
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0 # 当前剩余生命数
        self.was_real_done  = True # 用于判断游戏是否真正结束
        self.was_real_reset = False # 判断是否进行了真实的重置

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives() # 获取当前生命数
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True # 如果失去生命，将 done 标记为 True
        self.lives = lives
        return obs, reward, done, info # 返回观察、奖励、是否完成和附加信息

    def _reset(self):
        """只有在生命耗尽时才重置环境。"""
        if self.was_real_done:
            obs = self.env.reset() # 执行真正的重置
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0) # 在终止状态执行空操作
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives() # 更新当前生命数
        return obs # 返回新的观察值

# MaxAndSkipEnv 环境包装类：跳过多帧，每隔 skip 帧返回一次观察
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """仅返回每 skip 帧的观察结果"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2) # 用于存储最近的原始观察帧
        self._skip = skip # 每隔 skip 帧返回一次观察

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs) # 存储最近两帧观察
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0) # 对两帧取最大值
        return max_frame, total_reward, done, info # 返回最大帧，总奖励，是否完成和附加信息

    def _reset(self):
        """清除过去的帧缓冲区并获取初始观察值。"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs # 返回初始观察值

# _process_frame84 函数：将图像帧缩放为 84x84 的灰度图像
def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114 # 转换为灰度
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :] # 裁剪并调整大小
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8) # 返回 84x84 灰度图像

# ProcessFrame84 环境包装类：将图像帧处理为 84x84 的灰度图像
class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1)) # 设置新的观察空间

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info # 返回处理后的帧

    def _reset(self):
        return _process_frame84(self.env.reset())

# ClippedRewardsWrapper 环境包装类：对奖励进行裁剪，使其值变为 -1、0 或 1
class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info # 返回裁剪后的奖励

# wrap_deepmind_ram 函数：对环境进行一系列包装，不包含图像预处理
def wrap_deepmind_ram(env):
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClippedRewardsWrapper(env)
    return env # 返回包装后的环境

# wrap_deepmind 函数：对环境进行一系列包装，包含图像预处理
def wrap_deepmind(env):
    assert 'NoFrameskip' in env.spec.id # 确保环境支持跳帧
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env) # 包含图像预处理
    env = ClippedRewardsWrapper(env)
    return env # 返回包装后的环境
