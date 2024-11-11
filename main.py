import gym
import torch
import torch.optim as optim
import argparse

from model import DQN, Dueling_DQN # 导入 DQN 和 Dueling DQN 模型
from learn import dqn_learning, OptimizerSpec # 导入 DQN 学习函数和优化器规范
from utils.atari_wrappers import * # 导入 Atari 游戏环境包装器
from utils.gym_setup import * # 导入 gym 设置
from utils.schedules import * # 导入学习率和探索率调度函数

# Global Variables
# Extended data table 1 of nature paper
# 全局参数，参考 Nature DQN 论文中的扩展数据表 1
BATCH_SIZE = 32 # 批量大小
REPLAY_BUFFER_SIZE = 1000000 # 回放缓冲区大小
FRAME_HISTORY_LEN = 4 # 包含的历史帧数
TARGET_UPDATE_FREQ = 10000 # 更新目标网络的频率
GAMMA = 0.99 # 折扣因子
LEARNING_FREQ = 4 # 每隔多少步更新一次网络
LEARNING_RATE = 0.00025 # 学习率
ALPHA = 0.95 # RMSprop 优化器的 alpha 参数
EPS = 0.01 # RMSprop 优化器的 epsilon 参数
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1) # 线性探索率调度
LEARNING_STARTS = 50000 # 开始训练前的最小步数

# atari_learn 函数：执行 Atari 游戏的 DQN 训练过程
def atari_learn(env, env_id, num_timesteps, double_dqn, dueling_dqn):

    # 定义停止条件，当总步数达到 num_timesteps 时停止
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        # 注意这里的 t 是包装过的环境的步数，与底层环境步数不同
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # 优化器配置，使用 RMSprop 优化器并设置学习率和其他参数
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )

    # 选择 Dueling DQN 或普通 DQN，并调用 dqn_learning 函数开始训练
    if dueling_dqn:
        dqn_learning(
            env=env,
            env_id=env_id,
            q_func=Dueling_DQN, # 使用 Dueling DQN
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    else:
        dqn_learning(
            env=env,
            env_id=env_id,
            q_func=DQN, # 使用标准 DQN
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    env.close() # 训练完成后关闭环境


# main 函数：主函数，负责解析命令行参数并运行训练
def main():
    parser = argparse.ArgumentParser(description='RL agents for atari')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train an RL agent for atari games")
    train_parser.add_argument("--task-id", type=int, required=True, help="0 = BeamRider, 1 = Breakout, 2 = Enduro, 3 = Pong, 4 = Qbert, 5 = Seaquest, 6 = Spaceinvaders")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--double-dqn", type=int, default=0, help="double dqn - 0 = No, 1 = Yes")
    train_parser.add_argument("--dueling-dqn", type=int, default=0, help="dueling dqn - 0 = No, 1 = Yes")

    args = parser.parse_args() # 解析命令行参数

    # command
    # 设置 GPU 设备
    if (args.gpu != None):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print("CUDA Device: %d" %torch.cuda.current_device())

    # Get Atari games.
    # 获取 Atari 游戏的基准
    benchmark = gym.benchmark_spec('Atari40M') 

    # Change the index to select a different game.
    # 0 = BeamRider
    # 1 = Breakout
    # 2 = Enduro
    # 3 = Pong
    # 4 = Qbert
    # 5 = Seaquest
    # 6 = Spaceinvaders
    # 打印所有的 Atari 游戏任务
    for i in benchmark.tasks:
        print(i)

    # 根据任务 ID 获取特定的游戏任务
    task = benchmark.tasks[args.task_id]

    # Run training
    # 开始训练
    seed = 0 # 设置随机种子（建议根据需要随机化种子）
    double_dqn = (args.double_dqn == 1) # 是否使用 Double DQN
    dueling_dqn = (args.dueling_dqn == 1) # 是否使用 Dueling DQN
    env = get_env(task, seed, task.env_id, double_dqn, dueling_dqn) # 创建环境
    # 打印训练配置信息
    print("Training on %s, double_dqn %d, dueling_dqn %d" %(task.env_id, double_dqn, dueling_dqn))
    # 调用 atari_learn 函数，开始在 Atari 游戏上训练
    atari_learn(env, task.env_id, num_timesteps=task.max_timesteps, double_dqn=double_dqn, dueling_dqn=dueling_dqn)

# 如果文件作为主程序运行，则调用 main 函数
if __name__ == '__main__':
    main()
