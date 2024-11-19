import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from rundat import RunDat
import os


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EpisodeRunner:
    """
    A class to manage episodes in a reinforcement learning environment
    where an agent moves through a grid and receives rewards based on its trajectory.
    """
    def __init__(self, max_y, max_z, learning_rate=0.001, gamma=0.99):
        """
        Initialize the EpisodeRunner.

        Args:
            max_y (int): Maximum y-dimension of the grid.
            max_z (int): Maximum z-dimension of the grid.
            learning_rate (float): Learning rate for the DQN.
            gamma (float): Discount factor for future rewards.
        """
        # Define attributes
        self.max_y = max_y
        self.max_z = max_z
        self.current_y = 0
        self.current_z = 0
        self.trajectory = [(self.current_y, self.current_z)]
        self.episode_moves = []
        self.reward = 0

        # Define possible moves
        self.moves = {
            "top": (0, 1),
            "top_right": (1, 1),
            "right": (1, 0),
            "right_bottom": (1, -1),
            "bottom": (0, -1)
        }
        self.move_names = list(self.moves.keys())

        # Deep Q-Network initialisation
        self.input_size = 2  # (current_y, current_z)
        self.output_size = len(self.moves)
        self.gamma = gamma
        self.dqn = DeepQNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()


    def next_move(self):
        """
        Calculate the next move based on DQN predictions and update the current position and trajectory.
        """
        # Prepare input for DQN
        state = torch.tensor([self.current_y, self.current_z], dtype=torch.float32)

        # Epsilon-greedy action selection
        epsilon = 0.1
        if random.random() < epsilon:
            # Exploration: Random move
            move_index = random.randint(0, len(self.move_names) - 1)
        else:
            # Exploitation: Use DQN to predict the best move
            with torch.no_grad():
                q_values = self.dqn(state)
                move_index = torch.argmax(q_values).item()

        # Get the selected move
        move_name = self.move_names[move_index]
        dy, dz = self.moves[move_name]
        new_y = self.current_y + dy
        new_z = self.current_z + dz

        # Ensure the move is within grid boundaries
        if -self.max_z <= new_z <= self.max_z and 0 <= new_y <= self.max_y:
            self.current_y, self.current_z = new_y, new_z
            self.trajectory.append((self.current_y, self.current_z))
            self.episode_moves.append(move_name)


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
            if self.current_z > 0:
                self.episode_moves.append("bottom")
            else:
                self.episode_moves.append("top")
            self.reward += self.add_reward()

        # Train the DQN after the episode
        self.train_dqn()

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
        # TODO improve negative reward definition
        elif len(self.trajectory) > 1:
            last_y, last_z = self.trajectory[-1]
            prev_y, prev_z = self.trajectory[-2]
            return -((last_y - prev_y) ** 2 + (last_z - prev_z) ** 2) ** 0.5
        return 0

    def train_dqn(self):
        """
        Train the Deep Q-Network using the trajectory collected during the episode.
        """
        for i, _ in enumerate(self.trajectory[:-1]):
            current_state = torch.tensor(self.trajectory[i], dtype=torch.float32)
            next_state = torch.tensor(self.trajectory[i + 1], dtype=torch.float32)

            # Predict Q-values for the current state
            q_values = self.dqn(current_state)

            # Calculate target Q-value
            with torch.no_grad():
                next_q_values = self.dqn(next_state)
                max_next_q = torch.max(next_q_values).item()
                target_q_value = self.add_reward() + self.gamma * max_next_q

            # Update the Q-value for the taken action
            action_index = self.move_names.index(self.episode_moves[i])
            target = q_values.clone()
            target[action_index] = target_q_value

            # Compute loss and update the network
            loss = self.loss_fn(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def save_model(self, file_path):
        """
        Save the current model to a file.

        Args:
            file_path (str): Path to save the model.
        """
        torch.save(self.dqn.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Load a model from a file.

        Args:
            file_path (str): Path to load the model from.
        """
        self.dqn.load_state_dict(torch.load(file_path))
        self.dqn.eval()

if __name__ == "__main__":
    """
    Main execution: set up the grid, run multiple episodes, and save the final model.
    """

    # Square grid size
    max_y_int = 16
    num_episodes = 1000

    # Create an EpisodeRunner instance
    episode_runner = EpisodeRunner(max_y_int, max_y_int)
    
    # Run multiple episodes to train the DQN
    for episode in range(num_episodes):
        episode_runner.run_episode()
        episode_runner.current_y, episode_runner.current_z = 0, 0
        episode_runner.trajectory = [(episode_runner.current_y, episode_runner.current_z)]
        episode_runner.reward = 0

    # Save the trained model
    model_path = os.path.join(os.getcwd(), 'saved_model.pth')
    episode_runner.save_model(model_path)
    
    # Load the trained model and run a test episode
    episode_runner.load_model(model_path)
    episode_runner.run_episode()
    reward = episode_runner.get_reward()
    print(episode_runner.trajectory)
    print('Reward of the test episode is', reward)