import numpy as np
import random
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

def manhattan_distance(from_loc, to_loc):
    return abs(from_loc[0] - to_loc[0]) + abs(from_loc[1] - to_loc[1])

def potential_function(env, in_taxi):
    if not in_taxi:
        return -manhattan_distance(env.taxi_pos, env.passenger_loc)
    else:
        return -manhattan_distance(env.taxi_pos, env.destination)

def transform_state(env, in_taxi, grid_size):
    taxi_row, taxi_col = env.taxi_pos
    p_row, p_col = env.passenger_loc
    d_row, d_col = env.destination
    return (taxi_row/grid_size, taxi_col/grid_size, (taxi_row-p_row)/grid_size, (taxi_col-p_col)/grid_size, (taxi_row-d_row)/grid_size, (taxi_col-d_col)/grid_size, in_taxi)

def choose_action(state, Q, eps):
    if random.uniform(0, 1) < eps:
        return random.choice([0,1,2,3,4,5])
    if state not in Q:
        Q[state] = [0.0]*6
    return int(np.argmax(Q[state]))

def update_Q(Q, s, a, r, s_next, alpha, gamma):
    if s not in Q:
        Q[s] = [0.0]*6
    if s_next not in Q:
        Q[s_next] = [0.0]*6
    m = max(Q[s_next])
    Q[s][a] += alpha*(r + gamma*m - Q[s][a])

num_episodes = 150000
alpha = 0.05
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99997

Q_table = {}
scores = []
num_steps = []
times = []
successes = 0
old_success = 0

for episode in tqdm(range(num_episodes)):
    grid_size = random.randint(5, 10)
    env_config = {"grid_size": grid_size, "fuel_limit": 5000}
    env = SimpleTaxiEnv(**env_config)
    start = time.time()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    s = transform_state(env, False, grid_size)
    phi = potential_function(env, False)
    while not done:
        a = choose_action(s, Q_table, epsilon)
        nxt_obs, r, done, _ = env.step(a)
        in_taxi = s[-1]
        if a == 4 and r > -10:
            in_taxi = True
        elif a == 5 and s[-1]:
            in_taxi = False
        phi_next = potential_function(env, in_taxi)
        shaped_r = r + gamma*phi_next - phi
        total_reward += r
        s_next = transform_state(env, in_taxi, grid_size)
        update_Q(Q_table, s, a, shaped_r, s_next, alpha, gamma)
        s = s_next
        phi = phi_next
        steps += 1
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    scores.append(total_reward)
    num_steps.append(steps)
    times.append(time.time() - start)
    if total_reward >= 40:
        successes += 1
    if (episode + 1) % 1000 == 0:
        percent_success = (successes - old_success) / 1000.0
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}, Epsilon={epsilon:.3f}, Successes={successes}, %Success this 1000: {percent_success:3f}")
        old_success = successes

with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

fig, axs = plt.subplots(3, 2, figsize=(12, 8))
axs[0, 0].scatter(range(num_episodes), scores, s=2)
axs[0, 0].set_title("Total Rewards Over Episodes")
axs[0, 0].set_xlabel("Episodes")
axs[0, 0].set_ylabel("Total Reward")
axs[0, 1].scatter(range(num_episodes), num_steps, s=2)
axs[0, 1].set_title("Steps Over Episodes")
axs[0, 1].set_xlabel("Episodes")
axs[0, 1].set_ylabel("Steps")
axs[1, 0].scatter(range(num_episodes), times, s=2)
axs[1, 0].set_title("Time per Episode")
axs[1, 0].set_xlabel("Episodes")
axs[1, 0].set_ylabel("Time (seconds)")
rw = 100
r_avg = pd.Series(scores).rolling(rw).mean()
axs[1, 1].plot(r_avg, color="orange")
axs[1, 1].set_title(f"Rolling Avg of Total Rewards (window={rw})")
axs[1, 1].set_xlabel("Episodes")
axs[1, 1].set_ylabel("Mean Reward")
s_avg = pd.Series(num_steps).rolling(rw).mean()
axs[2, 0].plot(s_avg, color="green")
axs[2, 0].set_title(f"Rolling Avg of Steps (window={rw})")
axs[2, 0].set_xlabel("Episodes")
axs[2, 0].set_ylabel("Mean Steps")
axs[2, 1].axis("off")
plt.tight_layout()
plt.show()
