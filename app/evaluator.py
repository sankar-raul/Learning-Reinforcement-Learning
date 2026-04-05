import gymnasium as gym
import pickle as pkl
import numpy as np

cliffEnv = gym.make("CliffWalking-v1", render_mode="human")

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = np.random.randint(low=0, high=4)
    return action

q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))

done = False
state = cliffEnv.reset()[0]
action = policy(state=state)
while not done:
    state, reward, done, _, _ = cliffEnv.step(action=action)
    action = policy(state)
    