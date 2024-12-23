import numpy as np
import random
import logging


class BattleshipsEnv:
    def __init__(self, grid_size=5, boat_sizes=None):
        """
        A reinforcement learning environment for the Battleships game.

        Args:
            grid_size (int): Size of the game grid.
            boat_sizes (numpy.ndarray): Array of integers representing boat sizes.
        """
        self.grid_size = grid_size
        self.boat_sizes = boat_sizes
        if self.boat_sizes is None:
            raise ValueError("Boat sizes must be provided as a numpy array.")
        self.num_boats = len(self.boat_sizes)
        self.previous_hit = None  # Track the previous hit position and state
        self.previous_hit_penalty = (
            0  # Track penalty for consecutive misses after a hit
        )
        self.reset()

    def reset(self):
        """
        Resets the game environment.
        """
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.boat_positions = {}  # Map boat_id to set of cells
        self.agent_grid = np.zeros_like(self.grid)
        self.hits = 0
        self.previous_hit = None  # Reset previous hit state
        self.previous_hit_penalty = 0  # Reset the penalty tracker

        # Place boats randomly
        for boat_id, size in enumerate(self.boat_sizes, start=1):
            while not self._place_boat(boat_id, size):
                pass

        return self.agent_grid

    def _place_boat(
        self,
        boat_id: int,
        size: int,
    ):
        """
        Places a boat of a given size randomly on the grid.

        Args:
            boat_id (int): ID of the boat.
            size (int): Size of the boat.

        Returns:
            bool: True if the boat was placed successfully, False otherwise.
        """
        orientation: str = random.choice(["horizontal", "vertical"])
        if orientation == "horizontal":
            logging.info(f"Placing boat {boat_id} horizontally")
            row: int = random.randint(0, self.grid_size - 1)
            col_start: int = random.randint(0, self.grid_size - size)
            cells: list[tuple[int, int]] = [(row, col_start + i) for i in range(size)]
        else:  # Vertical
            logging.info(f"Placing boat {boat_id} vertically")
            col: int = random.randint(0, self.grid_size - 1)
            row_start: int = random.randint(0, self.grid_size - size)
            cells: list[tuple[int, int]] = [(row_start + i, col) for i in range(size)]

        if any(self.grid[row, col] != 0 for row, col in cells):
            logging.info(f"Boat {boat_id} placement failed")
            return False

        for row, col in cells:
            logging.info(f"Placing boat {boat_id} at ({row}, {col})")
            self.grid[row, col] = boat_id
        self.boat_positions[boat_id] = set(cells)
        return True

    def step(self, action):
        """
        Takes a step in the game by attacking a grid cell.

        Args:
            action (tuple): Coordinates of the grid cell to attack.

        Returns:
            tuple: (agent_grid, reward, done) after the action is taken.
        """
        # Unpack the action coordinates
        x, y = action

        # Validate action bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            raise ValueError(f"Action {action} is out of bounds!")

        # Initialize reward
        reward = 0

        # Check if the action hits a boat
        if any((x, y) in positions for positions in self.boat_positions.values()):
            logging.info(f"Hit at ({x}, {y})!")
            reward = 1  # Base reward for hitting a boat
            self.agent_grid[x, y] = 1  # Mark as hit
            boat_id = self.grid[x, y]

            # Remove the hit cell from the boat's positions
            logging.info(f"Removing ({x}, {y}) from boat {boat_id}")
            self.boat_positions[boat_id].remove((x, y))

            if not self.boat_positions[boat_id]:  # Check if the entire boat is sunk
                logging.info(f"Boat {boat_id} has been sunk!")
                reward += 10  # Bonus for sinking the boat
                del self.boat_positions[boat_id]
                self.previous_hit = None  # Reset previous hit if the boat is sunk
                self.previous_hit_penalty = 0  # Reset penalty
            else:
                logging.info(f"Boat {boat_id} is still afloat!")
                reward += 2  # Bonus for hitting part of a boat
                self.previous_hit = (
                    x,
                    y,
                    boat_id,
                )  # Record the hit position and boat ID
                self.previous_hit_penalty = (
                    -1
                )  # Initialize penalty for subsequent misses

                # Add a small bonus for targeting adjacent cells of a hit
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.grid_size
                        and 0 <= ny < self.grid_size
                        and (nx, ny) in self.boat_positions[boat_id]
                    ):
                        reward += 0.5  # Adjacent bonus

        else:
            # Action misses
            logging.info(f"Miss at ({x}, {y})")
            reward = -2  # Penalize misses
            self.agent_grid[x, y] = -1  # Mark as miss

            # Adjust penalty for consecutive misses after a hit
            if self.previous_hit is not None:
                prev_x, prev_y, prev_boat_id = self.previous_hit
                if (
                    prev_boat_id in self.boat_positions
                ):  # Check if the boat is still afloat
                    reward += self.previous_hit_penalty  # Apply penalty
                    self.previous_hit_penalty -= 1  # Increment penalty further

        # Game is done if all boats are sunk
        done = len(self.boat_positions) == 0

        # Return the updated agent's view, the reward, and whether the game is finished
        return self.agent_grid, reward, done

    def render(self):
        """
        Renders the agent's view of the game board.
        """
        print("Agent's View:")
        print(self.agent_grid)
