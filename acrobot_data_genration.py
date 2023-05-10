import numpy as np
import torch
import random
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(env, n_traj, traj_len):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n = 0
    traj_states = torch.zeros((n_traj, traj_len, n_states)).to(device)
    traj_actions = torch.zeros((n_traj, traj_len)).to(device)
    traj_rewards = torch.zeros((n_traj, traj_len)).to(device)

    while n < n_traj:
        state, info = env.reset()
        state = torch.tensor(state).to(device)

        for i in range(traj_len):
            action = env.action_space.sample()
            state_next, reward, terminated, truncated, info = env.step(action)

            traj_actions[n, i] = action
            traj_rewards[n, i] = reward
            traj_states[n, i, :] = state

            state = torch.tensor(state_next).to(device)

            if (terminated or truncated) and (i < traj_len - 1):
                n = n - 1
                break

        env.close()
        n = n + 1

    torch.save(traj_states, "data/acrobot_states.pt")
    torch.save(traj_actions, "data/acrobot_actions.pt")

def main():
    env = gym.make("Acrobot-v1")
    n_traj = 1e6
    traj_len = 500

    generate_data(env, n_traj, traj_len)

if __name__ == "__main__":
   main()
