# Vanilla DQN、Double DQN 和 Dueling DQN 的 PyTorch 实现

## 描述
此项目是使用 [PyTorch](https://www.pytorch.org/) 实现的 Vanilla DQN、Double DQN 和 Dueling DQN，基于以下论文：

- [通过深度强化学习实现人类水平的控制](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
- [使用双 Q 学习的深度强化学习](https://arxiv.org/abs/1509.06461)
- [深度强化学习中的对抗网络架构](https://arxiv.org/abs/1511.06581)

本项目的起始代码来自 [Berkeley CS 294 第三次作业](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3)，并经过修改以适配 PyTorch，同时参考了[此处](https://github.com/transedward/pytorch-dqn)。此外，加入了 Tensorboard 日志记录功能，感谢[这里](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard)的可视化指导，使训练过程中能够进行可视化分析，补充了 Gym Monitor 已有的功能。

## 背景
深度 Q 网络（DQN）使用神经网络作为动作值函数 Q 的近似器。这里使用的架构将 Atari 模拟器的帧作为输入（即状态），并将这些帧通过两个卷积层和两个全连接层，最终输出每个动作的 Q 值。

<p align="center">
    <img src="assets/nature_dqn_model.png" height="300px">
</p>

论文 [通过深度强化学习实现人类水平的控制](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) 引入了经验回放缓冲区，用于存储过去的观测并将其作为训练输入，以减少数据样本之间的相关性。它们还使用了一个独立的目标网络，该网络包含前一个时间步的权重，用于计算目标 Q 值。该网络的权重会定期更新，以匹配主 Q 网络最新的权重集，从而减少目标和当前 Q 值之间的相关性。Q 目标的计算方式如下图所示。

<p align="center">
    <img src="assets/nature_dqn_target.png" height="100px">
</p>

由于 Vanilla DQN 可能会高估动作值，[使用双 Q 学习的深度强化学习](https://arxiv.org/abs/1509.06461) 提出了一种替代的 Q 目标值：在输入下一步观测时，使用当前 Q 网络的 argmax 操作获得的动作，然后将这些动作和下一步观测输入冻结的目标网络，以生成每次更新的 Q 值。新的 Q 目标如下所示。

<p align="center">
    <img src="assets/double_q_target.png" height="70px">
</p>

最后，[深度强化学习中的对抗网络架构](https://arxiv.org/abs/1511.06581) 提出了一种不同的架构来近似 Q 函数。在最后一个卷积层之后，输出被分成两个流，分别估计状态值和在该状态下每个动作的优势。这两个估计随后结合在一起，通过下图中的公式生成一个 Q 值。图中展示了该架构与传统深度 Q 网络的对比。

<p align="center">
    <img src="assets/dueling_q_target.png" height="150px">
    <img src="assets/dueling_q_arch.png" height="300px">
</p>

## 依赖

- Python 2.7
- [PyTorch 0.2.0](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [OpenAI Gym](https://github.com/openai/gym)
- [OpenCV 3.3.0](https://pypi.python.org/pypi/opencv-python)
- [Tensorboard](https://github.com/tensorflow/tensorboard)

## 使用方法

- 执行以下命令以使用 Vanilla DQN 训练模型：

```bash
$ python main.py train --task-id $TASK_ID
