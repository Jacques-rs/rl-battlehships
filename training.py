# %% Imports
from battleships.environment import BattleshipsEnv
from battleships.agent import QLearningAgent
import numpy as np
from collections import deque

# %% Set up environment and agent
env = BattleshipsEnv(
    grid_size=10,
    boat_sizes=np.array([2, 3, 3]),
)
agent = QLearningAgent(
    grid_size=10,
    learning_rate=0.2,
    discount_factor=0.75,
    decay=0.9999,
    exploration_rate=20,
)

# %% Train the model
# Parameters
num_episodes = 2_500
reward_threshold = 10
reward_window = 100  # Number of episodes for calculating the rolling average

# Rolling reward tracker
recent_rewards = deque(maxlen=reward_window)

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    taken_actions = []

    while not done:
        action = agent.choose_action(state, taken_actions)
        if action is None:
            break

        taken_actions.append(action)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    recent_rewards.append(total_reward)  # Track total reward for the episode

    # Adjust learning rate based on recent performance
    if len(recent_rewards) == reward_window:
        avg_reward = sum(recent_rewards) / reward_window
        agent.adjust_learning_rate(
            recent_average_reward=avg_reward, reward_threshold=reward_threshold
        )

    # Decay exploration rate
    agent.decay_exploration(method="exponential", min_epsilon=0.25)

    # Log progress
    if episode % 100 == 0:
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        print(
            f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Learning Rate: {agent.lr:.4f}, Epsilon: {agent.epsilon:.2f}"
        )

print("Training complete!")


# %% Save the model
import pickle

with open("./models/battleships_model.pkl", "wb") as f:
    pickle.dump(agent, f)
