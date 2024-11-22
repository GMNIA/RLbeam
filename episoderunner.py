import random
import torch
import torch.nn as nn
import torch.optim as optim
from rundat import RunDat
import os


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)

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
    def __init__(self, max_y, max_z, learning_rate=0.001, gamma=0.99, use_dqn=True):
        """
        Initialize the EpisodeRunner.

        Args:
            max_y (int): Maximum y-dimension of the grid.
            max_z (int): Maximum z-dimension of the grid.
            learning_rate (float): Learning rate for the DQN.
            gamma (float): Discount factor for future rewards.
            use_dqn (bool): Whether to use the DQN for decision-making.
        """
        # Define attributes
        self.max_y = max_y
        self.max_z = max_z
        self.current_y = 0
        self.current_z = 0
        self.trajectory = [(self.current_y, self.current_z)]
        self.episode_moves = []
        self.reward = 0
        self.use_dqn = use_dqn
        self.dat_path_to_save = ''

        # Define possible moves
        self.moves = {
            "top": (0, 1),
            "top_right": (1, 1),
            "right": (1, 0),
            "bottom_right": (1, -1),
            "bottom": (0, -1)
        }
        self.move_names = list(self.moves.keys())

        # Deep Q-Network initialisation
        if self.use_dqn:
            self.input_size = 2
            self.output_size = len(self.moves)
            self.gamma = gamma
            self.dqn = DeepQNetwork(self.input_size, self.output_size)
            self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()


    def next_move(self):
        """
        Calculate the next move based on DQN predictions or random selection,
        and update the current position and trajectory.
        """

        if self.use_dqn:
            # Prepare input for DQN
            state = torch.tensor([self.current_y / self.max_y, self.current_z / self.max_z], dtype=torch.float32)

        # Determine valid moves
        valid_moves = {}
        for move_name, (dy, dz) in self.moves.items():
            new_y = self.current_y + dy
            new_z = self.current_z + dz

            # Check if the new position is within grid boundaries
            if 0 <= new_y <= self.max_y and -self.max_z <= new_z <= self.max_z:
                valid_moves[move_name] = (dy, dz)

        # Check if there are valid moves
        if not valid_moves:
            raise ValueError("No valid moves available. Check grid boundaries and constraints.")

        # Epsilon-greedy action selection
        if self.use_dqn:
            epsilon = 0.1
            if random.random() < epsilon:
                # Exploration: Random move
                move_name = random.choice(list(valid_moves.keys()))
            else:
                with torch.no_grad():
                    # Exploitation: Use DQN to predict the best move
                    q_values = self.dqn(state)

                    # Mask invalid moves by setting their Q-values to a very low number
                    min_value = torch.min(q_values).item() - 1
                    q_values_masked = torch.full_like(q_values, min_value)
                    for i, move_name in enumerate(self.moves.keys()):
                        if move_name in valid_moves:
                            q_values_masked[i] = q_values[i]

                    # Choose the proposed move from dqn and apply it with name
                    move_index = torch.argmax(q_values_masked).item()
                    move_name = self.move_names[move_index]
        else:
            # Run without DQN
            move_name = random.choice(list(valid_moves.keys()))

        # Update position based on the selected move
        dy, dz = valid_moves[move_name]
        self.current_y += dy
        self.current_z += dz

        # Save new trajectory move and reward for the episode
        self.trajectory.append((self.current_y, self.current_z))
        self.episode_moves.append(move_name)
        self.reward += self.add_reward()

    def run_episode(self):
        """
        Execute a full episode by repeatedly moving until the terminal
        condition is met. Adds a final step to ensure the trajectory ends at (max_y, 0).
        """
        # Make trajectory with moves, add final reward
        while not self.is_terminal():
            # Must do another terminal check if while training model beam lengths gets too long
            if self.calculate_trajectory_length() > 3 * (self.max_y + 2 *self.max_z):
                self.trajectory.append((self.max_y, 0))
                if self.trajectory[-1][1] > self.trajectory[-2][1]:
                    self.episode_moves.append(self.move_names[0])
                else:
                    self.episode_moves.append(self.move_names[-1])
                self.reward += self.add_reward()
                break

            # Try next move
            self.next_move()

        # When all steps are predicted the fem is run to calculate most important reward of model
        self.update_reward_with_fem_results()

        # Train the DQN after the episode
        if self.use_dqn:
            self.train_dqn()

    def is_terminal(self):
        """
        Check if the episode has reached its terminal condition.

        Returns:
            bool: True if the agent is at (max_y, 0), False otherwise.
        """
        return self.current_y == self.max_y and self.current_z == 0

    def calculate_trajectory_length(self):
        """
        Calculate the total length of the trajectory based on the Euclidean distance
        between consecutive points.

        Returns:
            float: The total length of the trajectory.
        """
        # No length if there's only one point in the trajectory
        if len(self.trajectory) < 2:
            return 0
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            # Get the current and previous points
            y1, z1 = self.trajectory[i - 1]
            y2, z2 = self.trajectory[i]

            # Calculate the Euclidean distance between the points
            distance = ((y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
            total_length += distance
        return total_length

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

    def update_reward_with_fem_results(self):
        """
        Generate a SOFiSTiK input file from the trajectory, run it, and
        update the reward based on simulation results.

        Returns:
            float: The updated reward.
        """
        # Create a folder to save the SOFiSTiK .dat file if it does not exist
        run_dat = RunDat(self.trajectory)

        # Run the SOFiSTiK simulation with / without saving
        run_dat.run(dat_path_to_save=self.dat_path_to_save)

        # Extract minimum displacement from simulation results
        min_z = min(displacement[-1] for displacement in run_dat.displacements)

        # Update reward based on a predefined rule where 1 / max_deformation_in_meters is coefficient
        self.reward += (self.max_y + self.max_z) * (1 + (1 / 0.05) * min_z)
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
    num_episodes = 100
    num_digits = len(str(num_episodes))

    # Create an EpisodeRunner instance and folder for saving
    episode_runner = EpisodeRunner(max_y_int, max_y_int, use_dqn=True)
    saved_data_path = os.path.join(os.getcwd(), 'saved_episodes')
    os.makedirs(saved_data_path, exist_ok=True)
    
    # Run multiple episodes to train the DQN
    for iEpisode in range(num_episodes):
        episode_runner.dat_path_to_save = os.path.join(saved_data_path, f'{iEpisode:0{num_digits}d}.dat')
        episode_runner.run_episode()
        episode_runner.current_y, episode_runner.current_z = 0, 0
        episode_runner.trajectory = [(episode_runner.current_y, episode_runner.current_z)]
        episode_runner.reward = 0

    # Save and load trining data after episodes
    if episode_runner.use_dqn:
        # Save the trained model
        model_path = os.path.join(os.getcwd(), 'saved_model.pth')
        episode_runner.save_model(model_path)
    
        # Load the trained model and run a test episode
        episode_runner.load_model(model_path)

    # Run one last episode
    episode_runner.run_episode()
    print(episode_runner.trajectory)
    print('Reward of the test episode is', episode_runner.reward)
