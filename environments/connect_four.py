# environments/connect_four.py
import numpy as np

class ConnectFour:
    """
    6 rows x 7 columns, players: +1 (X) and -1 (O), empty: 0
    Drop a disc in a column; it occupies the lowest empty row.
    """
    def __init__(self):
        self.rows, self.cols = 6, 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def _drop_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return -1  # full

    def step(self, action):
        if self.done:
            return self.board.copy(), 0.0, True, {"error": "episode_done"}

        col = int(action)
        if col < 0 or col >= self.cols:
            self.done = True
            self.winner = -self.current_player
            return self.board.copy(), -10.0, True, {"error": "invalid_col"}

        r = self._drop_row(col)
        if r == -1:
            self.done = True
            self.winner = -self.current_player
            return self.board.copy(), -10.0, True, {"error": "column_full"}

        self.board[r, col] = self.current_player

        if self._check_win(r, col, self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1.0 if self.current_player == 1 else -1.0, True, {"winner": self.current_player}

        if not self.get_valid_actions():
            self.done = True
            self.winner = 0
            return self.board.copy(), 0.0, True, {"winner": 0}

        self.current_player *= -1
        return self.board.copy(), 0.0, False, {}

    def _check_win(self, r, c, p):
        # Check 4-in-a-row in 4 directions
        return any(self._count_dir(r, c, dr, dc, p) >= 4 for dr, dc in [(0,1), (1,0), (1,1), (1,-1)])

    def _count_dir(self, r, c, dr, dc, p):
        count = 1
        # forward
        rr, cc = r + dr, c + dc
        while 0 <= rr < self.rows and 0 <= cc < self.cols and self.board[rr, cc] == p:
            count += 1; rr += dr; cc += dc
        # backward
        rr, cc = r - dr, c - dc
        while 0 <= rr < self.rows and 0 <= cc < self.cols and self.board[rr, cc] == p:
            count += 1; rr -= dr; cc -= dc
        return count

    def render(self):
        print("\n  " + " ".join(map(str, range(self.cols))))
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                v = self.board[i, j]
                row.append('X' if v == 1 else 'O' if v == -1 else '.')
            print(f"{i} " + " ".join(row))
        print()
