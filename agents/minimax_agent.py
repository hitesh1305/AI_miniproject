# agents/minimax_agent.py

from math import inf
import numpy as np
from agents.base_agent import BaseAgent

class MinimaxAgent(BaseAgent):
    def __init__(self, player=1, max_depth=9):
        super().__init__(name="Minimax")
        self.player = player
        self.max_depth = max_depth

    def act(self, state, valid_actions=None):
        board = state.copy()
        ordered = self._ordered_actions(board, valid_actions or self._all_valid(board))
        best_score, best_move = -inf, ordered[0]
        alpha, beta = -inf, inf
        for move in ordered:
            r, c = move
            board[r, c] = self.player
            score = self._minimax(board, depth=1, maximizing=False, alpha=alpha, beta=beta)
            board[r, c] = 0
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        return best_move

    # NEW: implement abstract method so the class is concrete
    def train(self, experience=None):
        # Minimax is a search algorithm; no training needed
        return None

    def _minimax(self, board, depth, maximizing, alpha, beta):
        if self._win(board, self.player): return 10 - depth
        if self._win(board, -self.player): return -10 + depth
        valid = self._all_valid(board)
        if not valid or depth >= self.max_depth: return 0
        if maximizing:
            value = -inf
            for (r, c) in self._ordered_actions(board, valid):
                board[r, c] = self.player
                value = max(value, self._minimax(board, depth+1, maximizing=False, alpha=alpha, beta=beta))
                board[r, c] = 0
                alpha = max(alpha, value)
                if beta <= alpha: break
            return value
        else:
            value = inf
            for (r, c) in self._ordered_actions(board, valid):
                board[r, c] = -self.player
                value = min(value, self._minimax(board, depth+1, maximizing=True, alpha=alpha, beta=beta))
                board[r, c] = 0
                beta = min(beta, value)
                if beta <= alpha: break
            return value

    def _ordered_actions(self, board, actions):
        center = [(1, 1)]
        corners = [(0,0), (0,2), (2,0), (2,2)]
        edges = [(0,1), (1,0), (1,2), (2,1)]
        return [a for a in center if a in actions] + \
               [a for a in corners if a in actions] + \
               [a for a in edges if a in actions]

    def _all_valid(self, board):
        return [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]

    def _win(self, board, p):
        if any((board[i, :] == p).all() for i in range(3)): return True
        if any((board[:, j] == p).all() for j in range(3)): return True
        if np.diag(board).tolist().count(p) == 3: return True
        if np.diag(np.fliplr(board)).tolist().count(p) == 3: return True
        return False
