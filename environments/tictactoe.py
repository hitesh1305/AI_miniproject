# environments/tictactoe.py
import numpy as np

class TicTacToe:
    """
    Board: 3x3; Players: 1 (X), -1 (O); Empty: 0
    """
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int)
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
        return [(r,c) for r in range(3) for c in range(3) if self.board[r,c] == 0]

    def _check_win(self, p):
        # rows/cols
        for i in range(3):
            if all(self.board[i,:] == p): return True
            if all(self.board[:,i] == p): return True
        # diagonals
        if all(np.diag(self.board) == p): return True
        if all(np.diag(np.fliplr(self.board)) == p): return True
        return False

    def step(self, action):
        r, c = action
        if self.board[r,c] != 0:
            self.done = True
            self.winner = -self.current_player
            return self.board.copy(), -10, True, {"error":"invalid"}
        self.board[r,c] = self.current_player
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1 if self.current_player==1 else -1, True, {"winner": self.current_player}
        if not self.get_valid_actions():
            self.done = True
            self.winner = 0
            return self.board.copy(), 0, True, {"winner": 0}
        self.current_player *= -1
        return self.board.copy(), 0, False, {}

    def render(self):
        sym = {1:'X', -1:'O', 0:'.'}
        print("\n  0 1 2")
        for i in range(3):
            print(f"{i} " + " ".join(sym[self.board[i,j]] for j in range(3)))
        print()
