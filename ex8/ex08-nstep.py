import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in policy]))


def select_action(q_table, state, epsilon):
    """ This is an example of selecting an action based on the Q-table """
    if np.random.rand() < epsilon:
        action = np.random.randint(0, 4)
    else:
        action = np.argmax(q_table[state, :])
    return action


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.999, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    q_table = np.ones((64, 4))
    success=0
    for i in range(num_ep):
        state = env.reset()
        action = select_action(q_table, state, epsilon)
        t=0
        done = False
        done_updating=False
        SARSA_list = [state, action]

        while not done_updating:
            if not done:
                next_state, reward, done, _ = env.step(action)
                next_action = select_action(q_table, next_state, epsilon)
                SARSA_list.append(reward)
                SARSA_list.append(next_state)
                SARSA_list.append(next_action)
                action=next_action
                state=next_state
                if reward ==1:
                    success+=1
                
            tau = t-n+1
            try:
                if tau >= 0:
                    G = 0
                    for i in range(tau, min(tau+n, int(len(SARSA_list)/3-1))+1):
                        G += np.power(gamma,i-tau-1)*SARSA_list[3*i+2]
                    if tau+n < len(SARSA_list)/3:
                        G += np.power(gamma,n)*q_table[SARSA_list[3*(tau+n)], SARSA_list[3*(tau+n)+1]]
                    q_table[SARSA_list[3*tau], SARSA_list[3*tau+1]] += alpha*(G-q_table[SARSA_list[3*tau], SARSA_list[3*tau+1]])
            except IndexError:
                done_updating=True
                
            t+=1
    return q_table,success/num_ep


env = gym.make('FrozenLake-v0', map_name="8x8",is_slippery=True)
# TODO: run multiple times, evaluate the performance for different n and alpha
alpha_list=np.linspace(0.001,0.4,20)
n_list=np.logspace(0,7,8,base=2)
for n in tqdm(n_list):
    succ_list=[]
    for alpha in alpha_list:
        q,success= nstep_sarsa(env,int(n),alpha)
        print("alpha: ",alpha,"n: ",n,"success rate: ",success)
        succ_list.append(success)
    plt.plot(alpha_list,succ_list,label='n='+str(n))
plt.xlabel('alpha')
plt.ylabel('success rate')
plt.legend()
plt.show()

# q,success= nstep_sarsa(env)
# print_policy(q, env)
# print(success)
