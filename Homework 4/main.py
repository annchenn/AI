 # pylint: disable=all
from BanditEnv import BanditEnv
from Agent import Agent
from random import random
import numpy as np
import matplotlib.pyplot as plt
k=10
#part 3
epsilon=[0,0.1,0.01]
runs=2000
steps=1000
total_rewards = [[0 for _ in range(steps)] for _ in range(len(epsilon))]
total_optimal = [[0 for _ in range(steps)] for _ in range(len(epsilon))]

avg_rewards=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
avg_optimal=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
env=BanditEnv(k)
for i in range(len(epsilon)):
    for j in range (runs):
        env.reset()
        optimal = np.argmax(env.arm_means)
        agent = Agent(k, epsilon[i])
        for s in range(steps):
            action = agent.select_action()
            if(action==optimal):
                total_optimal[i][s]+=1
            reward= env.step(action)
            agent.update_q(action, reward)
            total_rewards[i][s]+=reward
        
avg_rewards = [[total_rewards[i][s]/runs for s in range(steps)] for i in range(len(epsilon))]
avg_optimal = [[(total_optimal[i][s]/runs)*100 for s in range(steps)] for i in range(len(epsilon))]

for i in range(len(epsilon)):
    plt.plot(avg_rewards[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title('Average reward in stationary environment')
plt.legend()
plt.savefig('Average_Reward(stationary).png')
plt.close()

for i in range(len(epsilon)):
    plt.plot(avg_optimal[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Percentage of Optimal Action')
plt.title('The percentage of optimal action in stationary environment')
plt.legend()
plt.savefig('Percentage_of_Optimal_Action(stationary).png')
plt.close()
            
#part 5
runs=2000
steps=10000
total_rewards = [[0 for _ in range(steps)] for _ in range(len(epsilon))]
total_optimal = [[0 for _ in range(steps)] for _ in range(len(epsilon))]

avg_rewards=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
avg_optimal=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
env=BanditEnv(k,stationary=False)
for i in range(len(epsilon)):
    for j in range (runs):
        env.reset()
        agent = Agent(k, epsilon[i])
        optimal = np.argmax(env.arm_means)
        for s in range(steps):
            action = agent.select_action()
            if(action==optimal):
                total_optimal[i][s]+=1
            reward= env.step(action)
            agent.update_q(action, reward)
            total_rewards[i][s]+=reward
            optimal = np.argmax(env.arm_means)
        
avg_rewards = [[total_rewards[i][s]/runs for s in range(steps)] for i in range(len(epsilon))]
avg_optimal = [[(total_optimal[i][s]/runs)*100 for s in range(steps)] for i in range(len(epsilon))]

for i in range(len(epsilon)):
    plt.plot(avg_rewards[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title('The average reward in non-stationary environment')
plt.legend()
plt.savefig('Average_Reward(Non-stationary).png')
plt.close()

for i in range(len(epsilon)):
    plt.plot(avg_optimal[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Percentage of Optimal Action')
plt.title('The percentage of optimal action in non-stationary environment')
plt.legend()
plt.savefig('Percentage_of_Optimal_Action(Non-stationary).png')
plt.close()

#part 7
env.reset()
total_rewards = [[0 for _ in range(steps)] for _ in range(len(epsilon))]
total_optimal = [[0 for _ in range(steps)] for _ in range(len(epsilon))]

avg_rewards=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
avg_optimal=[[0 for _ in range(steps)] for _ in range(len(epsilon))]
for i in range(len(epsilon)):
    for j in range (runs):
        env.reset()
        agent = Agent(k, epsilon[i],0.1)
        optimal = np.argmax(env.arm_means)
        for s in range(steps):
            action = agent.select_action()
            if(action==optimal):
                total_optimal[i][s]+=1
            reward= env.step(action)
            agent.update_q(action, reward)
            total_rewards[i][s]+=reward
            optimal = np.argmax(env.arm_means)
        
avg_rewards = [[total_rewards[i][s]/runs for s in range(steps)] for i in range(len(epsilon))]
avg_optimal = [[(total_optimal[i][s]/runs)*100 for s in range(steps)] for i in range(len(epsilon))]

for i in range(len(epsilon)):
    plt.plot(avg_rewards[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title('The average reward in non-stationary environment with constant step size')
plt.legend()
plt.savefig('Average_Reward(Non-stationary_with_Constant_Step_Size).png')
plt.close()

for i in range(len(epsilon)):
    plt.plot(avg_optimal[i], label = f'epsilon = {epsilon[i]}')
plt.xlabel('Step')
plt.ylabel('Percentage of Optimal Action')
plt.title('The percentage of optimal action in non-stationary environment with constant step size')
plt.legend()
plt.savefig('Percentage_of_Optimal_Action(Non-stationary_with_Constant_Step_Size).png')
plt.close()