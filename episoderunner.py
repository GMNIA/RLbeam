import random
from rundat import RunDat
import os

class EpisodeRunner:
    def __init__(self, max_y, max_z):
        self.max_y = max_y  # Size of the nxn grid
        self.max_z = max_z  # Size of the nxn grid
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
        # Filter valid moves (must stay within the grid and move to the right)
        valid_moves = []
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
        # Run the episode until y has reached the other end of the space (max_y)
        while not self.is_terminal():
            self.next_move()
            self.reward += self.add_reward()

        # If the trajectory is not exactly at z == 0 then impose last step to end
        if self.trajectory[-1] != (self.max_y, 0):
            self.current_y, self.current_z = self.max_y, 0
            self.trajectory.append((self.max_y, 0))
            self.reward += self.add_reward()


    def is_terminal(self):
        # Terminal condition: reaching the end point (y = max_y, z = 0)
        return self.current_y == self.max_y


    def add_reward(self):
        # No reward is given only if the end point is reached
        if self.is_terminal():
            return 0
        elif len(self.trajectory) > 1:
            # Set length as negative reward
            last_y, last_z = self.trajectory[-1]
            prev_y, prev_z = self.trajectory[-2]
            return -((last_y - prev_y) ** 2 + (last_z - prev_z) ** 2) ** 0.5
        else:
            return 0


    def get_reward(self):
        return self.reward


    def update_reward_with_results(self):
        run_dat = RunDat(self.trajectory)
        run_dat.run(dat_path_to_save=os.path.join(os.getcwd(), 'temp', 'temp.dat'))
        min_z = min(displacement[-1] for displacement in run_dat.displacements)

        # Calculate reward based on a fix 0.05 meter maximum allowed downward displacement
        # TODO generalise displacement based reward
        if min_z > 0.05:
            self.reward += min(2 * (self.max_y + self.max_z), 1 / abs(min_z))
        else:
            self.reward = min(2 * (self.max_y + self.max_z), 1 / abs(min_z))
        self.reward += (self.max_y + self.max_z) * (1 + 20 * min_z)
        return self.reward


if __name__ == "__main__":
    # Set up the grid size (n)
    max_y_int = 16

    # Run an episode
    episode_runner = EpisodeRunner(max_y_int, max_y_int)
    episode_runner.run_episode()
    reward = episode_runner.update_reward_with_results()
    print('Reward of the episode is', reward)
