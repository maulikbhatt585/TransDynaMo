import random, csv, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import pprint as pp
import numpy as np
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # We can remove "reward" if we want!

### HELPER CLASSES AND FUNCTION ###
# Trajectory Memory
class Memory(object):
    def __init__(self, capacity):
        # deque is a double ended queue
        self.memory = [] # Capacity is the number of trajectories we want!

    # Save a Transition
    def push(self, *args):
        self.memory.append(Transition(*args))

    # Random Sampling
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def print(self):
        pp.pprint(self.memory)

    def to_file(self, name="trajFile"):
        with open(name, "w") as file:
            pp.pprint(self.memory, file)

    def to_csv(self, name="trajectories.csv"):
        t = self.memory[0]
        simple_form = [["State"]*len(t.state) + ["Action"] + ["Next State"]*len(t.next_state) + ["Reward"]]
        for t in self.memory:
            temp = np.append(t.state, t.action)
            temp = np.append(temp, t.next_state)
            temp = np.append(temp, t.reward)
            simple_form.append(temp)

        with open(name, "w", newline="") as csv_file:   # Check if already has data and use "a" instead?
            csvWriter = csv.writer(csv_file, delimiter=',')
            csvWriter.writerows(simple_form)
        

def policy(s):
   return env.action_space.sample()


env = gym.make("CartPole-v1", render_mode="human")
num_traj = 1
traj_length = 1000
memory = Memory(traj_length)
# trajdf = pd.DataFrame()

for t in range(num_traj):
    state, info = env.reset(seed=42)    # Initial State!
    for i in range(traj_length):
        action = policy(state)    # Random policy
        next_state, reward, terminated, truncated, info = env.step(action)  # Step Forward!
        if(terminated):
            next_state=[None]*len(state)
        memory.push(state, action, next_state, reward)
        state = next_state

        if terminated or truncated:
            break
    state, info = env.reset()
    # Store trajectory!
    memory.to_csv()
env.close()