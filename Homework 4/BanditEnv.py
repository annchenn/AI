import numpy as np

class BanditEnv():
    def __init__(self, k,stationary = True):
        self.k = k
        self.stationary = stationary
        self.arm_means = np.random.normal(0,1,k)
        self.action_history = []
        self.reward_history = []
        
    def reset(self):
        self.arm_means = np.random.normal(0,1,self.k)
        self.action_history = []
        self.reward_history = []
        
    def step(self, action):
        reward = np.random.normal(self.arm_means[action],1)
        self.action_history.append(action)
        self.reward_history.append(reward)
        if self.stationary == False:
            for i in range(self.k):
                self.arm_means[i]+=np.random.normal(0,0.01)
        return reward
        
    def export_history(self):
        return self.action_history, self.reward_history
    
    # def print_rewards(self):
    #     for i in range(self.k):
    #         print(f'arm {i}: {self.arm_means[i]}')