import torch
import gym
from A2C import A2C
from matplotlib import pyplot as plt

def train(env_name='CartPole-v1', num_episode=2000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2C(state_size, action_size)

    total_rewards = []

    for episode in range(num_episode):
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        total_reward = 0

        while True:
            # 选择动作并获取 log probability
            action, log_prob = agent.select_action(state)

            # 在环境中执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 使用当前步的信息更新 agent
            # 将 log_prob 作为参数传递给 update 函数
            agent.update(state, action, reward, next_state, done, log_prob)

            # 更新状态
            state = torch.from_numpy(next_state).float().unsqueeze(0)

            if done:
                break

        total_rewards.append(total_reward)
        print('Episode: {}, total reward: {}'.format(episode, total_reward))

    episodes = list(range(1, len(total_rewards) + 1))

    # 创建画布和坐标轴
    plt.figure(figsize=(12, 6))

    # 绘制曲线
    plt.plot(episodes, total_rewards,
             linewidth=1.5,
             color='royalblue',
             label='Total Reward per Episode')

    # 添加标签和标题
    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Reward Progression', fontsize=14, pad=20)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend()

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train()
