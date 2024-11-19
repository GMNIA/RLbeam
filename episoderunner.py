import random
from rundat import RunDat
import os

class EpisodeRunner:
    """
    A class to manage episodes in a reinforcement learning environment
    where an agent moves through a grid and receives rewards based on its trajectory.
    """
    def __init__(self, max_y, max_z):
        """
        Initialize the EpisodeRunner.

        Args:
            max_y (int): Maximum y-dimension of the grid.
            max_z (int): Maximum z-dimension of the grid.
        """
        # Define attributes
        self.max_y = max_y
        self.max_z = max_z
        self.current_y = 0
        self.current_z = 0
        self.trajectory = [(self.current_y, self.current_z)]
        self.reward = 0

        # Define possible moves
        self.moves = {
            "top": (0, 1),
            "top_right": (1, 1),
            "right": (1, 0),
            "right_bottom": (1, -1),
            "bottom": (0, -1)
        }


    def next_move(self):
        """
        Calculate the next move based on valid positions within the grid
        and update the current position and trajectory.
        """
        valid_moves = []
        
        # Filter valid moves to stay within grid boundaries
        for move, (dy, dz) in self.moves.items():
            new_z = self.current_z + dz
            new_y = self.current_y + dy
            if -self.max_z <= new_z <= self.max_z and 0 <= new_y <= self.max_y:
                valid_moves.append((move, new_y, new_z))

        # Randomly choose a valid move
        chosen_move = random.choice(valid_moves)
        self.current_y, self.current_z = chosen_move[1], chosen_move[2]
        self.trajectory.append((self.current_y, self.current_z))


    def run_episode(self):
        """
        Execute a full episode by repeatedly moving until the terminal
        condition is met. Adds a final step to ensure the trajectory ends at (max_y, 0).
        """
        while not self.is_terminal():
            self.next_move()
            self.reward += self.add_reward()

        # Ensure the trajectory ends at (max_y, 0) if not already there
        if self.trajectory[-1] != (self.max_y, 0):
            self.current_y, self.current_z = self.max_y, 0
            self.trajectory.append((self.max_y, 0))
            self.reward += self.add_reward()


    def is_terminal(self):
        """
        Check if the episode has reached its terminal condition.

        Returns:
            bool: True if the agent is at (max_y, 0), False otherwise.
        """
        return self.current_y == self.max_y


    def add_reward(self):
        """
        Calculate and return the reward for the last move in the trajectory.

        Returns:
            float: The calculated reward for the last move.
        """
        if self.is_terminal():
            return 0
        
        # Calculate the negative reward based on the distance of the last move
        elif len(self.trajectory) > 1:
            last_y, last_z = self.trajectory[-1]
            prev_y, prev_z = self.trajectory[-2]
            return -((last_y - prev_y) ** 2 + (last_z - prev_z) ** 2) ** 0.5
        
        return 0


    def get_reward(self):
        """
        Retrieve the total accumulated reward for the episode.

        Returns:
            float: The total reward.
        """
        return self.reward


    def update_reward_with_results(self):
        """
        Generate a SOFiSTiK input file from the trajectory, run it, and
        update the reward based on simulation results.

        Returns:
            float: The updated reward.
        """
        # Create a folder to save the SOFiSTiK .dat file if it does not exist
        run_dat = RunDat(self.trajectory)
        saved_file_folder = os.path.join(os.getcwd(), 'saved_dat')
        os.makedirs(saved_file_folder, exist_ok=True)
        
        # Run the SOFiSTiK simulation and save the file
        run_dat.run(dat_path_to_save=os.path.join(saved_file_folder, 'temp.dat'))
        
        # Extract minimum displacement from simulation results
        min_z = min(displacement[-1] for displacement in run_dat.displacements)

        # Update reward based on a predefined rule
        self.reward += (self.max_y + self.max_z) * (1 + 20 * min_z)
        return self.reward


if __name__ == "__main__":
    """
    Main execution: set up the grid, run an episode, and calculate the final reward.
    """
    # Square grid size
    max_y_int = 16

    # Create an EpisodeRunner instance
    episode_runner = EpisodeRunner(max_y_int, max_y_int)
    
    # Run the episode
    episode_runner.run_episode()
    
    # Update the reward based on simulation results
    reward = episode_runner.update_reward_with_results()
    print('Reward of the episode is', reward)
