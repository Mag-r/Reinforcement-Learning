import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def tabular_qlearning(state, action, reward, next_state, q_table, alpha, gamma):
    """ This is an example of tabular Q-learning """

    q_table[state[0],state[1], action] = q_table[state[0],state[1], action] + alpha * (reward + gamma * np.max(q_table[next_state[0],next_state[1], :]) - q_table[state[0],state[1], action])
    return q_table

def select_action(q_table, state, epsilon): 
    """ This is an example of selecting an action based on the Q-table """
    if np.random.rand() < epsilon:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(q_table[state[0],state[1],:])
    return action

def perform_qlearning(env, q_table, alpha, gamma, epsilon, num_episodes=6500):
    """ This is an example of performing Q-learning """
    success=[]
    steps = []
    succ=0
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        for step in range(202):
            discrete_state=[int(np.floor((state[0]+1.2)*20/1.8)), int(np.floor((state[1]+0.07)*20/0.14))]
            action = select_action(q_table, discrete_state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_discrete_state = [int(np.floor((next_state[0]+1.2)*20/1.8)), int(np.floor((next_state[1]+0.07)*20/0.14))]
            q_table = tabular_qlearning(discrete_state, action, reward, next_discrete_state, q_table, alpha, gamma)
            state = next_state
            total_reward += reward
            if done:
                if state[0] >= 0.5:
                    succ += 1
                success.append(succ)
                steps.append(step)
                break
        # print("Episode: ", episode, "Total reward: ", total_reward)
    return q_table, success,steps


def continuous_sarsa(env,alpha,gamma,epsilon, num_episodes=6500):
    weight_matrix = np.random.rand(3,2)
    success=[]
    steps = []
    loss =[]
    succ=0
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        action = np.argmax(np.dot(weight_matrix, state))
        l=[]
        for step in range(202):
            next_state, reward, done, info = env.step(action)
            q_value = np.dot(weight_matrix, state)
            action_new = np.argmax(q_value) if np.random.rand() > epsilon else np.random.randint(0, 3)
            q_new=np.copy(q_value)
            q_new[action] =reward + gamma * np.dot(weight_matrix, next_state)[action_new]
            q_new = q_new.reshape(3,1)
            state = state.reshape(1,2)
            q_value = q_value.reshape(3,1)
            weight_matrix = weight_matrix-alpha*np.dot(q_value-q_new,state)
            state = next_state
            action = action_new
            total_reward += reward
            l.append(np.linalg.norm(q_value-q_new))
            if done:
                if state[0] >= 0.5:
                    succ += 1
                    # print("Success")
                success.append(succ)
                steps.append(step)
                break
        loss.append(np.mean(l))
    return weight_matrix, success, steps, loss



def drive(env, q_table):
    for i in range(1000):
        state = env.reset()
        for step in range(200):
            discrete_state=[int(np.floor((state[0]+1.2)*20/1.8)), int(np.floor((state[1]+0.07)*20/0.14))]
            action = np.argmax(q_table[discrete_state[0],discrete_state[1],:])
            next_state, reward, done, info = env.step(action)
            state = next_state
            env.render()
            if done:
                break

def conti_drive(env, weight_matrix):
    for i in range(1000):
        state = env.reset()
        for step in range(200):
            action = np.argmax(np.dot(weight_matrix, state))
            next_state, reward, done, info = env.step(action)
            state = next_state
            env.render()
            if done:
                break

def main():
    env = gym.make('MountainCar-v0')
    
    # random_episode(env)#
    succ=[]
    step = []
    losses = []
    alpha = 0.03
    gamma = 1.0
    epsilon = 0.1
    for i in range(10):
        env.reset()
        q_table = np.zeros((20,20, 3))
        q_table, success, steps = perform_qlearning(env, q_table, alpha, gamma, epsilon, 20000)
        succ.append(success)
        step.append(steps)
    fig,ax =plt.subplots(4,2,figsize=(10,10))

    success = np.mean(succ, axis=0)
    last_hundred = [np.mean(success[i-100:i]-success[i-100]) for i in range(100, len(success))]
    steps = np.mean(step, axis=0)
    ax[0,0].plot(success)
    ax[0,0].set_title("discrete Q-learning; successful episodes")
    ax[0,0].set_xlabel("Episodes")
    ax[0,0].set_ylabel("Success")
    ax[1,0].plot(steps)
    ax[1,0].set_title("discrete Q-learning; steps per episode")
    ax[1,0].set_xlabel("Episodes")
    ax[1,0].set_ylabel("Steps")
    ax[3,0].plot(last_hundred)
    ax[3,0].set_title("discrete Q-learning; success of last 100 episodes")
    ax[3,0].set_xlabel("Episodes")
    ax[3,0].set_ylabel("Success of last 100 episodes")
    succ=[]
    step = []
    losses = []
    for i in range(10):
        env.reset()
        weight_matrix,success,steps, loss = continuous_sarsa(env,alpha,gamma,epsilon,20000)   
        succ.append(success)
        step.append(steps)
        losses.append(loss)
    print(np.shape(losses))
    success = np.mean(succ, axis=0)
    loss = np.mean(losses, axis=0)
    steps = np.mean(step, axis=0)
    last_hundred = [np.mean(success[i-100:i]-success[i-100]) for i in range(100, len(success))]
    ax[0,1].plot(success)
    ax[0,1].set_title("Continuous SARSA; successful episodes")
    ax[0,1].set_xlabel("Episodes")
    ax[0,1].set_ylabel("Success")
    ax[1,1].plot(steps)
    ax[1,1].set_title("Continuous SARSA; steps per episode")
    ax[1,1].set_xlabel("Episodes")
    ax[1,1].set_ylabel("Steps")
    ax[2,1].plot(loss)
    ax[2,1].set_title("Continuous SARSA; MSE loss per episode")
    ax[2,1].set_xlabel("Episodes")
    ax[2,1].set_ylabel("Loss")
    ax[3,1].plot(last_hundred)
    ax[3,1].set_title("Continuous SARSA; success of last 100 episodes")
    ax[3,1].set_xlabel("Episodes")
    ax[3,1].set_ylabel("Success of last 100 episodes")
    plt.show()
    
    # drive(env, q_table)
    conti_drive(env, weight_matrix)
    env.close()


if __name__ == "__main__":
    main()
