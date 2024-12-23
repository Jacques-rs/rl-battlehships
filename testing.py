# %% Imports
import pickle
import numpy as np
import logging
from battleships.environment import BattleshipsEnv

# %% Load the saved model
with open("./models/battleships_model.pkl", "rb") as f:
    agent = pickle.load(f)

# %% Create the Environment
env = BattleshipsEnv(
    grid_size=10,
    boat_sizes=np.array([2, 3, 3]),
)

# %% Test the modelh
state = env.reset()
done = False
env.render()

# Track actions taken to prevent redundant moves
taken_actions = []

while not done:
    # Choose action
    action = agent.choose_action(state, taken_actions)

    if action is None:
        print("No valid actions left!")
        break

    # Validate the action
    assert (
        0 <= action[0] < env.grid_size and 0 <= action[1] < env.grid_size
    ), "Action out of bounds!"

    # Add action to the taken actions list
    taken_actions.append(action)

    # Take step in the environment
    next_state, reward, done = env.step(action)
    env.render()
    state = next_state

print("Game Over!")
