'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/run_dqn_atari.py
'''

import gym
from gym import wrappers
import numpy as np
import random
from utils.atari_wrappers import *

def set_global_seeds(i):
    """
    设置全局随机种子，以保证实验的可重复性。

    参数:
    i: int
        要设置的随机种子值。
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i) # 设置 PyTorch 的随机种子
    np.random.seed(i) # 设置 numpy 的随机种子
    random.seed(i) # 设置 Python 内置的随机数生成器的种子

def get_env(task, seed, vid_dir_name, double_dqn, dueling_dqn):
    """
    创建和设置强化学习环境，并应用包装器。

    参数:
    task: object
        包含环境 ID 的任务对象。
    seed: int
        随机种子值。
    vid_dir_name: str
        用于存储视频和实验结果的目录名称。
    double_dqn: bool
        是否使用 Double DQN 算法。
    dueling_dqn: bool
        是否使用 Dueling DQN 算法。

    返回:
    env: gym.Env
        经过包装的 Gym 环境。
    """
    env_id = task.env_id # 获取任务的环境 ID

    env = gym.make(env_id) # 创建 Gym 环境

    set_global_seeds(seed) # 设置全局随机种子
    env.seed(seed) # 设置环境的随机种子

    # 根据所使用的 DQN 类型设置视频保存目录
    if double_dqn:
        expt_dir = 'tmp/%s/double/' %vid_dir_name
    elif dueling_dqn:
        expt_dir = 'tmp/%s/dueling/' %vid_dir_name
    else:
        expt_dir = 'tmp/%s/' %vid_dir_name

    # 使用 Gym 的 Monitor 包装器记录环境视频和数据
    env = wrappers.Monitor(env, expt_dir, force=True)

    # 使用自定义的 DeepMind 环境包装器
    env = wrap_deepmind(env)

    return env

def get_wrapper_by_name(env, classname):
    """
    在 Gym 环境中查找指定名称的包装器。

    参数:
    env: gym.Env
        包含包装器的 Gym 环境。
    classname: str
        要查找的包装器名称。

    返回:
    currentenv: gym.Env
        指定名称的包装器实例。

    异常:
    ValueError
        如果未找到指定名称的包装器则引发异常。
    """
    currentenv = env
    while True:
        # 检查当前环境是否与 classname 匹配
        if classname in currentenv.__class__.__name__:
            return currentenv
        # 如果当前环境是包装器，继续向下查找底层环境
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        # 如果没有找到匹配的包装器，抛出异常
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
