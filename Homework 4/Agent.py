import numpy as np
import random
class Agent():
    def __init__(self,k, epsilon,alpha=None):
        self.k =k
        self.epsilon=epsilon
        self.q_val = [0]*k
        self.action_count = [0]*k
        self.alpha = alpha
    
    def select_action(self):
        if random.random()<self.epsilon:
            action = random.randint(0,self.k-1)
        else:
            action = np.argmax(self.q_val)
        return action
    
    def reset(self):
        self.action_count=[0]*self.k
        self.q_val=[0]*self.k
        
    def update_q(self, action, reward):
        if self.alpha == None:
            self.action_count[action]+=1
            self.q_val[action]=(self.q_val[action]*(self.action_count[action]-1)+reward)/self.action_count[action]
            
        else:
            self.q_val[action] = self.q_val[action]+self.alpha*(reward-self.q_val[action])