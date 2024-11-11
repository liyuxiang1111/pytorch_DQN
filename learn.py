"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
from torch.autograd import Variable
import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import * # 引入回放缓冲区
from utils.schedules import * # 引入时间表（如探索率的调整）
from utils.gym_setup import * # 引入 gym 相关的设置
from logger import Logger # 引入日志记录器
import time

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"]) # 定义优化器规范，用于优化器配置

# 设置 CUDA 变量
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Set the logger
logger = Logger('./logs')

def to_np(x):
    return x.data.cpu().numpy() # 将 tensor 转换为 numpy 数组

# dqn_learning 函数：实现 DQN 算法的主要训练循环
def dqn_learning(env,
          env_id,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          double_dqn=False,
          dueling_dqn=False):
    """运行深度 Q 学习算法，指定参数并初始化环境和模型。
    参数
    ----------
    env: gym.Env
        gym 环境实例
    env_id: string
        环境 ID，用于保存模型
    q_func: function
        Q 函数模型
    optimizer_spec: OptimizerSpec
        优化器的构造函数和参数
    exploration: Schedule
        随机探索的时间表
    stopping_criterion: function
        用于检查是否满足停止训练的条件
    replay_buffer_size: int
        回放缓冲区的大小
    batch_size: int
        每次采样的批量大小
    gamma: float
        折扣因子
    learning_starts: int
        开始训练前的步数
    learning_freq: int
        训练之间的步数
    frame_history_len: int
        传递给模型的帧数
    target_update_freq: int
        更新目标 Q 网络的频率
    double_dqn: bool
        是否使用双重 DQN
    dueling_dqn: bool
        是否使用对抗网络
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # 构建模型     #
    ###############
    # 确定输入形状：根据环境的观察空间决定
    if len(env.observation_space.shape) == 1:
        # 低维观察（如 RAM 数据）
        input_shape = env.observation_space.shape
        in_channels = input_shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c) # 将多帧历史叠加为输入
        in_channels = input_shape[2]
    num_actions = env.action_space.n # 获取动作空间的大小
    
    # 定义 Q 网络和目标 Q 网络
    Q = q_func(in_channels, num_actions).type(dtype) # 主 Q 网络
    Q_target = q_func(in_channels, num_actions).type(dtype) # 目标 Q 网络

    # 初始化优化器
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # 运行环境     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan') # 平均奖励
    best_mean_episode_reward = -float('inf') # 最佳平均奖励
    last_obs = env.reset() # 初始化环境观察
    LOG_EVERY_N_STEPS = 1000 # 日志记录间隔
    SAVE_MODEL_EVERY_N_STEPS = 100000 # 模型保存间隔

    for t in itertools.count():
        ### 1. 检查停止条件
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. 环境步进并存储过渡
        # 存储最后一个观察帧
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # 获取输入 Q 网络的观察数据（包含之前的帧）
        observations = replay_buffer.encode_recent_observation()

        # 选择动作（随机或 epsilon 贪婪）
        if t < learning_starts:
            action = np.random.randint(num_actions) # 随机选择动作
        else:
            # epsilon 贪婪策略
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                # 选择 Q 值最大的动作
                obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                # 随机选择动作
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        obs, reward, done, info = env.step(action) # 执行动作并获取观察和奖励

        # 对奖励进行裁剪（-1 到 1）
        reward = np.clip(reward, -1.0, 1.0)

        # 存储动作效果
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # 如果达到终点，重置环境
        if done:
            obs = env.reset()

        # 更新 last_obs
        last_obs = obs

        ### 3. 执行经验回放并训练网络
        # 如果回放缓冲区中包含足够的样本
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            # 从回放缓冲区采样批量数据
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
            act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
            rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
            obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            # 获取 Q 值
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            if (double_dqn):
                # 使用双重 DQN 更新
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = Q(obs_tp1).detach()
                _, a_prime = q_tp1_values.max(1)

                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(obs_tp1).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime 

                error = rew_t + gamma * q_target_s_a_prime - q_s_a
            else:
                # 使用标准 DQN 更新
                # ---------------
                #   regular DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime 

                # Compute Bellman error
                # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip
            # 截断误差并反向传播
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error.data.unsqueeze(1))

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            # 更新目标 Q 网络
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            # 记录参数的直方图
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)

        ### 4. 记录进度
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models") # 如果 models 文件夹不存在，则创建它

            # 根据是否使用 Double DQN 或 Dueling DQN 设置文件名中的标签
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            # 保存模型的路径，包含环境 ID、Double/Dueling 标签、当前步数和当前时间
            model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path) # 保存当前 Q 网络的参数

        # 获取最近的 100 个回合的奖励并计算平均值
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:]) # 计算最近 100 个回合的平均奖励
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward) # 更新最佳平均奖励

        # 每隔 LOG_EVERY_N_STEPS 步输出日志
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush() # 刷新标准输出

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            # 将每个标量值记录到 TensorBoard 中
            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            # 记录最新一轮的奖励
            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            # 记录最近 100 个回合的平均奖励和最佳平均奖励
            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)