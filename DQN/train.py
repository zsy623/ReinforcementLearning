import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
import gym
from model import DQN_Agent

rewards = []
env_name = 'CartPole-v0'

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))

    # 创建数据框便于绘图
    data = pd.DataFrame({
        'Episode': range(1, len(rewards) + 1),
        'Reward': rewards,
    })

    # 绘制原始奖励曲线（半透明）
    sns.lineplot(
        x='Episode', y='Reward',
        data=data, alpha=0.3,
        label='Episode Reward'
    )

    # # 绘制移动平均曲线
    # sns.lineplot(
    #     x='Episode', y='Moving Avg',
    #     data=data, color='red',
    #     linewidth=2.5,
    #     label='100-Episode Moving Avg'
    # )

    # 添加标记线
    solved_threshold = 195 if env_name == "CartPole-v1" else None
    if solved_threshold:
        plt.axhline(
            y=solved_threshold,
            color='green',
            linestyle='--',
            label='Solved Threshold'
        )

    plt.title(f'DQN Performance on {env_name}', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)

    # # 突出显示学习趋势
    # plt.annotate(
    #     'Exploration Phase',
    #     xy=(len(rewards) * 0.2, max(rewards) * 0.3),
    #     xytext=(len(rewards) * 0.05, max(rewards) * 0.6),
    #     arrowprops=dict(arrowstyle='->'),
    #     fontsize=12
    # )
    #
    # plt.annotate(
    #     'Exploitation Phase',
    #     xy=(len(rewards) * 0.8, max(rewards) * 0.9),
    #     xytext=(len(rewards) * 0.6, max(rewards) * 0.7),
    #     arrowprops=dict(arrowstyle='->'),
    #     fontsize=12
    # )

    plt.tight_layout()
    plt.savefig(f'dqn_{env_name}_performance.png', dpi=300)
    plt.show()

def train(env_name = 'CartPole-v0',tot_episodes = 150):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN_Agent(state_size, action_size)
    steps = 0

    for episode in range(tot_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state

            agent.update()
            steps += 1

            if done:
                break

            if steps % agent.update_frequency == 0:
                agent.update_target()

        print("Episode: {}, Reward: {}".format(episode, episode_reward))
        rewards.append(episode_reward)

    env.close()

    plot_rewards(rewards)

    return agent
