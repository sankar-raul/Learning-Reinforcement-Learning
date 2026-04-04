import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
cliffEnv = gym.make("CliffWalking-v1", render_mode="ansi")

q_table = np.zeros(shape=(48, 4))

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(low=0, high=4)
    return action

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500
# cliffEnv.metadata["render_fps"] = 120
max_reward = -np.inf
max_state = None
for ep in range(NUM_EPISODES):
    done = False
    state = cliffEnv.reset()[0]
    action = policy(state=state, explore=EPSILON)
    episode_length = 0
    total_reward = 0
    while not done:
        # print(cliffEnv.render())
        episode_length += 1
        next_state, reward, done, _, _ = cliffEnv.step(action=action)
        total_reward += reward
        next_action = policy(next_state, EPSILON)
        q_table[state][action] += ALPHA * ( reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])
        state = next_state
        action = next_action
        # print(f"{ep}-->{total_e}-->{["top","right","bottom","left"][action]} --> state --> {state}"
    if (max_reward < total_reward):
        max_reward = total_reward
        max_state = q_table
    print(f"Episode No --> {ep} | Episode length --> {episode_length} | Total Reward --> {total_reward} | Max Reward --> {max_reward}")

cliffEnv.close()