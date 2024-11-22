# RLbeam

**Deep Q-Network (DQN) Based Reinforcement Learning Model for Beam Design**

This project implements a **Deep Q-Network (DQN)** to optimize beam designs in a simulated reinforcement learning environment. The agent learns to move through a grid while minimizing deflection and optimizing trajectory length. The final model is evaluated using **SOFiSTiK Finite Element Method (FEM)** simulations.

---

## Features

- **Grid-Based Movement**:
  The agent moves through a grid in 2D space, starting at `(0, 0)` and aiming to reach the terminal state `(max_y, 0)`.

- **Deep Q-Network (DQN)**:
  - Neural network architecture:
    ```python
    Input size: 2 (normalized `y`, `z`)
    Hidden layers: 2 layers, 128 neurons each
    Output: Q-values for 5 moves
    Activation: ReLU
    ```

- **Epsilon-Greedy Strategy**: Balances exploration and exploitation.
- **SOFiSTiK FEM Integration**:
  Generates `.dat` files for FEM simulations and updates rewards based on minimum deflection (`min_z`).

---

## Model Architecture

| Layer          | Size                     | Activation |
|-----------------|--------------------------|------------|
| Input           | 2 (normalized `y`, `z`) | N/A        |
| Hidden Layer 1  | 128 neurons             | ReLU       |
| Hidden Layer 2  | 128 neurons             | ReLU       |
| Output          | 5 (Q-values for moves)  | N/A        |

---

## Agent Actions

The agent can choose from the following moves:

1. **Top**: `(0, 1)`
2. **Top-Right**: `(1, 1)`
3. **Right**: `(1, 0)`
4. **Bottom-Right**: `(1, -1)`
5. **Bottom**: `(0, -1)`

---

## Usage

### Initialize the Environment

```python
episode_runner = EpisodeRunner(max_y=16, max_z=16, use_dqn=True)
```

### Run Episodes

```python
for iEpisode in range(num_episodes):
    episode_runner.run_episode()
```

### Save and Load Models

```python
# Save the trained model
episode_runner.save_model('saved_model.pth')

# Load a pre-trained model
episode_runner.load_model('saved_model.pth')
```

### Test the Model

```python
episode_runner.run_episode()
print("Trajectory:", episode_runner.trajectory)
print("Reward:", episode_runner.reward)
```

---

## Directory Structure

```
RLbeam/
├── episoderunner.py          # Main implementation
├── rundat.py                 # SOFiSTiK integration module
├── saved_episodes/           # Directory for episode `.dat` files
├── saved_model.pth           # Trained DQN model file
├── README.md                 # Project documentation
└── __pycache__/              # Ignored cache files
```

---

## Requirements

```bash
# Python: 3.8 or higher
pip install torch numpy
```

- **Libraries**:
  - `torch`
  - `numpy`
  - **SOFiSTiK** (for FEM simulation)

# Optional: Additional FEM setup
Configure SOFiSTiK for your platform
