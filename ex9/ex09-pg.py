import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    h= np.dot(state,theta)
    denominator = np.sum(np.exp(h))
    return [np.exp(h_up)/denominator for h_up in h]  # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states,
        the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    length_episode=[]
    for e in tqdm(range(10000)):

        if e % 3000 == 0:
            states, rewards, actions = generate_episode(env, theta, False)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        # print("episode: " + str(e) + " length: " + str(len(states)))
        length_episode.append(len(states))

        # TODO: implement the reinforce algorithm to improve the policy weights
        # Note: you may need to update theta within this loop
        for t in range(len(states)):
            G = sum([r * (1 ** i) for i, r in enumerate(rewards[t:])])
            # print(np.shape([states[t]]), np.shape([policy(states[t], theta)[(actions[t]+1)%2]/np.sum(policy(states[t], theta)) ,- policy(states[t], theta)[(actions[t]+1)%2]/np.sum(policy(states[t], theta))] ))
            grad = np.array([states[t]]).transpose() @ np.array([policy(states[t], theta)[(actions[t]+1)%2]/np.sum(policy(states[t], theta)) ,- policy(states[t], theta)[(actions[t]+1)%2]/np.sum(policy(states[t], theta))]).reshape(1,2)
            theta += 0.01 * G * grad

    return length_episode


def main():
    env = gym.make('CartPole-v1')
    length_episode=REINFORCE(env)
    env.close()
    mean = [np.mean(length_episode[i-100:i]) for i in range(100, len(length_episode))]
    plt.plot(length_episode)
    plt.show()


if __name__ == "__main__":
    main()
