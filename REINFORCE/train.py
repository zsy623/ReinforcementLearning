import torch
import gym
from REINFORCE import REINFORCE

def train(env_name='CartPole-v1', num_episode=500):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCE(state_size, action_size)

    for episode in range(num_episode):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        agent.update()
        print('Episode: {}, total reward: {}'.format(episode, total_reward))

train()
