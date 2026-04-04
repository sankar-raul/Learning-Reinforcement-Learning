import gymnasium as gym
import numpy as np
import time

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

cliffEnv = gym.make("CliffWalking-v1", render_mode="human")
cliffEnv.metadata["render_fps"] = 60

reset_result = cliffEnv.reset()
state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
done = False
total_ep = 0
while not done:
    action = np.random.randint(low=0, high=4)
    step_result = cliffEnv.step(action=action)
    # print(cliffEnv.render())
    total_ep += 1
    print(total_ep)
    if len(step_result) == 5:
        state, reward, terminated, truncated, _ = step_result
        done = terminated or truncated
    else:
        state, reward, done, _ = step_result

    # time.sleep(0.01)

cliffEnv.close()
