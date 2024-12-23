import numpy as np
import random
from typing import Literal
import logging


class QLearningAgent:
    def __init__(
        self,
        grid_size,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        decay=0.99,
    ):
        """Initializes a Q-learning agent for the Battleships game.

        Configures the agent's learning parameters and sets up the initial Q-table for value estimation.

        Args:
          - `grid_size` (int): Size of the game grid.
          - `learning_rate` (float, optional): Rate at which the agent learns from new information.
            Defaults to 0.1.
          - `discount_factor` (float, optional): Importance of future rewards. Defaults to 0.9.
              A higher value will prioritize long-term rewards.
              A lower value will prioritize short-term rewards.
          - `exploration_rate` (float, optional): Initial probability of choosing a random action.
            Defaults to 1.0. Ranges
            A higher value will prioritize exploration.
            A lower value will prioritize exploitation.
          - `decay` (float, optional):
              The decay value determines how quickly the agent shifts from exploration to exploitation.
              Rate of exploration decay over time.
              Defaults to 0.99.
              A value closer to 1 will decay slowly.
              A value closer to 0 will decay quickly.

        """

        self.grid_size: int = grid_size
        self.lr: float = learning_rate
        self.gamma: float = discount_factor
        self.epsilon: float = exploration_rate
        self.decay: float = decay
        self.q_table: dict = {}

    def choose_action(self, state, taken_actions):
        state_key = self.get_state_key(state)
        logging.info(f"Choosing action for state: {state_key}")
        q_values = self.get_q_values(state_key)
        logging.info(f"Q-values: {q_values}")
        valid_actions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in taken_actions
        ]
        logging.info(f"Valid actions: {valid_actions}")
        if random.random() < self.epsilon:
            logging.info("Exploring...")
            return random.choice(valid_actions) if valid_actions else None
        if valid_actions:
            logging.info("Exploiting...")
            return max(valid_actions, key=lambda action: q_values[action])
        return None

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a specific state-action pair using a dynamic learning rate.

        Args:
            state (np.ndarray): The current state of the agent.
            action (tuple): The action taken by the agent.
            reward (float): The reward received for taking the action.
            next_state (np.ndarray): The next state of the agent after taking the action.
        """

        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        q_values = self.get_q_values(state_key)
        next_q_values = self.get_q_values(next_state_key)

        x, y = action
        best_next_q = np.max(next_q_values)
        current_q = q_values[x, y]
        learning_rate = np.max(self.lr, 0.99)
        q_values[x, y] = current_q + learning_rate * (
            reward + self.gamma * best_next_q - current_q
        )

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((self.grid_size, self.grid_size))
        return self.q_table[state_key]

    def get_state_key(self, agent_grid):
        logging.info(f"Getting state key for agent grid: {agent_grid}")
        return tuple(agent_grid.flatten())

    def decay_exploration(
        self,
        method: Literal["exponential", "linear", "inverse_t", "slow_exponential"],
        min_epsilon=0.1,
    ):
        """
        Reduces the exploration rate for the agent's policy.

        Args:
            method (str): The method to use for decaying the exploration rate.
                Options are 'exponential', 'linear', 'inverse_t', and 'slow_exponential'.
            min_epsilon (float): The minimum value for epsilon to prevent over-decay.
        """
        if method == "exponential":
            self.epsilon *= self.decay
        elif method == "linear":
            self.epsilon -= self.decay
        elif method == "inverse_t":
            self.epsilon = 1 / (1 + self.decay * self.epsilon)
        elif method == "slow_exponential":
            # A slower exponential decay
            self.epsilon = self.epsilon * np.exp(-self.decay)
        else:
            raise ValueError(f"Unknown decay method: {method}")

        # Clamp epsilon to ensure it doesn't fall below the minimum threshold
        self.epsilon = max(self.epsilon, min_epsilon)

    def adjust_learning_rate(
        self,
        recent_average_reward,
        reward_threshold=10,
    ):
        """
        Adjusts the learning rate based on the recent average reward.

        Args:
            recent_average_reward (float): The average reward over recent episodes.
            reward_threshold (float): The reward threshold for reducing the learning rate.
        """
        if recent_average_reward >= reward_threshold:
            self.lr *= 0.9  # Reduce learning rate by 10%
            self.lr = max(self.lr, 0.01)  # Ensure it doesn't go below a minimum value
            logging.info(f"Learning rate adjusted to: {self.lr}")
