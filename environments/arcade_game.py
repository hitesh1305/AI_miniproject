# environments/arcade_game.py
import numpy as np
import random

class SimpleArcadeGame:
    """
    Discrete actions: 0=left, 1=stay, 2=right
    State vector (normalized floats):
      [player_x, goal_x, goal_y, obs1_x, obs1_y, obs2_x, obs2_y, obs3_x, obs3_y]
    Episode ends on collision (−1) or reaching goal (+1). No step-limit timeout.
    """
    def __init__(self, width=10, height=12, max_steps=None, n_obstacles=3):
        self.W = width
        self.H = height
        self.max_steps = max_steps  # None disables timeout termination
        self.n_obstacles = n_obstacles
        self.player_x = None
        self.goal_x = None
        self.goal_y = None
        self.obstacles = None  # list of (x, y)
        self.done = False
        self.steps = 0
        self.state_size = 2 + 1 + 2 * n_obstacles  # player_x, goal_x, goal_y, obs(x,y)*n
        self.action_space = [0, 1, 2]  # left, stay, right

    def reset(self):
        self.player_x = self.W // 2
        self.goal_x = random.randint(0, self.W - 1)
        self.goal_y = 0
        self.obstacles = [
            (random.randint(0, self.W - 1), random.randint(1, self.H // 2))
            for _ in range(self.n_obstacles)
        ]
        self.done = False
        self.steps = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {"error": "episode_done"}

        # player moves
        if action == 0:
            self.player_x = max(0, self.player_x - 1)
        elif action == 2:
            self.player_x = min(self.W - 1, self.player_x + 1)

        reward = -0.01  # small step penalty to encourage efficient goal-seeking
        info = {}

        # obstacles fall
        new_obstacles = []
        for (ox, oy) in self.obstacles:
            ny = oy + 1
            if ny >= self.H:
                ox = random.randint(0, self.W - 1)
                ny = 1
            new_obstacles.append((ox, ny))
        self.obstacles = new_obstacles

        # goal falls slowly (every two steps)
        if self.steps % 2 == 0:
            self.goal_y += 1
            if self.goal_y >= self.H:
                self.goal_x = random.randint(0, self.W - 1)
                self.goal_y = 0

        # collision check (player is at bottom row y=H-1)
        for (ox, oy) in self.obstacles:
            if oy == self.H - 1 and ox == self.player_x:
                self.done = True
                reward = -1.0
                info["terminal"] = "collision"
                return self._get_state(), reward, True, info

        # goal check
        if self.goal_y == self.H - 1 and self.goal_x == self.player_x:
            self.done = True
            reward = +1.0
            info["terminal"] = "goal"
            return self._get_state(), reward, True, info

        # advance step counter; no timeout if max_steps is None
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), 0.0, True, {"terminal": "timeout"}

        return self._get_state(), reward, False, info

    def get_valid_actions(self):
        return self.action_space

    def _get_state(self):
        # normalized [0,1] values
        s = [
            self.player_x / (self.W - 1),
            self.goal_x / (self.W - 1),
            self.goal_y / (self.H - 1),
        ]
        for (ox, oy) in self.obstacles:
            s.append(ox / (self.W - 1))
            s.append(oy / (self.H - 1))
        return np.array(s, dtype=np.float32)
