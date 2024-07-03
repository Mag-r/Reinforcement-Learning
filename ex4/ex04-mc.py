import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        # print("observation:", obs)
        states.append(obs)
        
        if obs[0] >= 20:
            # print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            # print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print("final observation:", obs)
    return states, ret


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 10000  # use whatever number of iterations you want
    for i in tqdm(range(maxiter)):
        states, reward = single_run_20()
        for j in range(len(states)):
            if states[j] not in states[j+1:]:
                state = states[j]
                returns[state[0]-12, state[1]-1, int(state[2])] += reward
                visits[state[0]-12, state[1]-1, int(state[2])] += 1
                V[state[0]-12, state[1]-1, int(state[2])] = returns[state[0]-12, state[1]-1, int(state[2])] /visits[state[0]-12, state[1]-1, int(state[2])]
            else:
                print("same state")
    return V, visits, np.sum(returns)/maxiter


def play_game(pi):
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    actions = []
    ret = 0.
    while not done:
        # print("observation:", obs)
        states.append(obs)
        
        action = pi[obs[0]-12, obs[1]-1, int(obs[2])]
        actions.append(action)
        if action == 1:
            # print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        else:
            # print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print("final observation:", obs)
    return states, ret, actions


def monte_carlo_es(pi = np.zeros((10, 10, 2))):
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:

    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 1000000  # use whatever number of iterations you want
    for i in tqdm(range(maxiter)):
        states, reward, actions = play_game(pi)
        for j in range(len(states)):
            state = states[j]
            action = int(actions[j])
            returns[state[0]-12, state[1]-1, int(state[2]), action] += reward
            visits[state[0]-12, state[1]-1, int(state[2]), action] += 1
            Q[state[0]-12, state[1]-1, int(state[2]), action] = returns[state[0]-12, state[1]-1, int(state[2]), action] /visits[state[0]-12, state[1]-1, int(state[2]), action]
            pi[state[0]-12, state[1]-1, int(state[2])] = np.argmax(Q[state[0]-12, state[1]-1, int(state[2]), :])
    
        # if i % 100000 == 0:
        #     print("Iteration: " + str(i))
        #     print(pi[:, :, 0])
        #     print(pi[:, :, 1])
    return pi,Q, np.sum(returns)/maxiter


def main():
    # single_run_20()
    V, visits, r=policy_evaluation()
    fig,ax=plt.subplots(2,subplot_kw={'projection': '3d'})
    X,Y=np.meshgrid(range(12,22),range(1,11))
    ax[0].plot_surface(X,Y,V[:,:,0])
    ax[0].set_title('No Usable Ace')
    ax[1].plot_surface(X,Y,V[:,:,1])
    ax[1].set_title('Usable Ace')
    print(r)
    plt.show()
    pi_init = np.load("pi.npy")
    pi,Q, R=monte_carlo_es(pi_init)
    print(np.linalg.norm(pi-pi_init))
    print(R)
    np.save("pi.npy", pi)

    V=np.max(Q,axis=3)
    fig,ax=plt.subplots(2,subplot_kw={'projection': '3d'})
    X,Y=np.meshgrid(range(12,22),range(1,11))
    ax[0].plot_surface(X,Y,V[:,:,0])
    ax[0].set_title('No Usable Ace')
    ax[1].plot_surface(X,Y,V[:,:,1])
    ax[1].set_title('Usable Ace')
    ax[0].set_xlabel('Player Sum')
    ax[0].set_ylabel('Dealer Card')
    ax[1].set_xlabel('Player Sum')
    ax[1].set_ylabel('Dealer Card')

    plt.savefig("Q.png")
    plt.show()

if __name__ == "__main__":
    main()
