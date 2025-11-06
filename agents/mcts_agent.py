# agents/mcts_agent.py
import math
import numpy as np
from copy import deepcopy
from agents.base_agent import BaseAgent
from environments.connect_four import ConnectFour

class MCTSNode:
    def __init__(self, state_board, current_player, parent=None, action=None, root_player=1):
        self.state_board = state_board  # numpy array (6x7)
        self.current_player = current_player
        self.parent = parent
        self.action = action
        self.root_player = root_player
        self.children = []
        self.untried_actions = None
        self.visits = 0
        self.wins = 0.0  # from root_player perspective

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search for Connect Four.
    num_simulations: number of rollouts per move
    c: exploration constant (sqrt(2) default)
    """
    def __init__(self, player=1, num_simulations=1000, c=1.41, rollout_depth_limit=100):
        super().__init__(name="MCTS")
        self.player = player
        self.num_simulations = num_simulations
        self.c = c
        self.rollout_depth_limit = rollout_depth_limit

    def act(self, state, valid_actions=None):
        # Build root from current state
        env = ConnectFour()
        env.board = state.copy()
        env.current_player = self.player

        root = MCTSNode(env.board.copy(), env.current_player, parent=None, action=None, root_player=self.player)
        root.untried_actions = valid_actions if valid_actions is not None else env.get_valid_actions()

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_env = ConnectFour()
            sim_env.board = node.state_board.copy()
            sim_env.current_player = node.current_player

            # Selection
            while node.untried_actions is not None and len(node.untried_actions) == 0 and node.children:
                node = max(node.children, key=lambda n: n.ucb1(self.c))
                _, _, _, _ = sim_env.step(node.action)

            # Expansion
            if node.untried_actions is None:
                node.untried_actions = sim_env.get_valid_actions()
            if node.untried_actions:
                a = node.untried_actions.pop()
                next_state, _, _, _ = sim_env.step(a)
                child = MCTSNode(next_state.copy(), sim_env.current_player, parent=node, action=a, root_player=node.root_player)
                child.untried_actions = sim_env.get_valid_actions()
                node.children.append(child)
                node = child

            # Simulation (rollout)
            result = self._rollout(sim_env, node.root_player)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.wins += result
                result = 1.0 - result  # invert perspective for the opponent
                node = node.parent

        # Choose the most visited child
        if not root.children:
            # Fallback: choose a random valid action if no children expanded
            acts = env.get_valid_actions()
            return np.random.choice(acts) if acts else 0
        best = max(root.children, key=lambda n: n.visits)
        return best.action

    def _rollout(self, sim_env, root_player):
        # Random playout until terminal or depth limit
        steps = 0
        while not sim_env.done and steps < self.rollout_depth_limit:
            acts = sim_env.get_valid_actions()
            if not acts:
                break
            a = np.random.choice(acts)
            _, _, done, info = sim_env.step(a)
            steps += 1
            if done:
                w = info.get("winner", 0)
                if w == root_player:
                    return 1.0
                elif w == 0:
                    return 0.5
                else:
                    return 0.0
        return 0.5  # treat non-terminal depth limit as draw for stability

    def train(self, experience=None):
        # MCTS is search-based (no gradient training)
        return None
