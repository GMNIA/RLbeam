# RLbeam

Deep Q-Network (DQN) Based Reinforcement Learning Model for Beam Design
This project implements a Deep Q-Network (DQN) to optimize beam designs in a simulated reinforcement learning environment. The agent learns to move through a grid while minimizing deflection and optimizing trajectory length. The final model is evaluated using SOFiSTiK Finite Element Method (FEM) simulations.

Features
Grid-Based Movement:

The agent moves through a grid in 2D space, starting from (0, 0) and attempting to reach the terminal state (max_y, 0).
Deep Q-Network (DQN):

Neural network architecture:
2 input features (normalized y and z coordinates).
2 hidden layers with 128 neurons each.
Output layer for Q-values corresponding to 5 possible moves.
Uses ReLU activation and weight initialization (Kaiming and Xavier).
Reinforcement Learning:

Epsilon-Greedy Strategy: Balances exploration (random moves) and exploitation (best Q-value moves).
Rewards:
Negative rewards for long or inefficient moves.
Final rewards from FEM simulation results.
Trajectory Length Constraint: Automatically ends the episode if the trajectory becomes excessively long.
SOFiSTiK FEM Integration:

Generates .dat files for SOFiSTiK input from the trajectory.
Updates rewards based on minimum deflection (min_z) from FEM results.
Episode Management:

Saves trajectory data (.dat files) for each episode in a saved_episodes directory.
Supports custom termination conditions based on grid size and trajectory length.
Model Persistence:

Trained DQN models are saved to disk (saved_model.pth).
Supports loading pre-trained models for evaluation.
Model Architecture
Deep Q-Network
Layer	Size	Activation
Input	2 (normalized y, z)	N/A
Hidden Layer 1	128	ReLU
Hidden Layer 2	128	ReLU
Output	5 (Q-values for moves)	N/A
Agent Actions
The agent can choose from five possible moves:

Top: (0, 1)
Top-Right: (1, 1)
Right: (1, 0)
Bottom-Right: (1, -1)
Bottom: (0, -1)
Usage
Initialize the Environment:

python
Copy code
episode_runner = EpisodeRunner(max_y=16, max_z=16, use_dqn=True)
Run Episodes:

Train the agent for a specified number of episodes.
Trajectory data is saved as .dat files in the saved_episodes directory.
python
Copy code
for iEpisode in range(num_episodes):
    episode_runner.run_episode()
Save and Load Models:

Save the trained model:
python
Copy code
episode_runner.save_model('saved_model.pth')
Load a pre-trained model:
python
Copy code
episode_runner.load_model('saved_model.pth')
Test the Model:

Run a test episode after training:
python
Copy code
episode_runner.run_episode()
print("Trajectory:", episode_runner.trajectory)
print("Reward:", episode_runner.reward)
Directory Structure
graphql
Copy code
RLbeam/
├── episoderunner.py          # Main implementation
├── rundat.py                 # SOFiSTiK integration module
├── saved_episodes/           # Directory for episode `.dat` files
├── saved_model.pth           # Trained DQN model file
├── README.md                 # Project documentation
└── __pycache__/              # Ignored cache files
Requirements
Python 3.8 or higher
Required libraries:
torch
numpy
SOFiSTiK (for FEM simulation)
Install dependencies:
bash
Copy code
pip install torch numpy
Future Improvements
Reward Design:
Refine reward functions for better convergence.
Dynamic Grid Size:
Allow dynamic resizing of the grid for more complex scenarios.
Advanced FEM Integration:
Incorporate more complex simulations and structural constraints.
This model provides a robust framework for reinforcement learning in structural design optimization and can be extended to solve real-world engineering problems.

