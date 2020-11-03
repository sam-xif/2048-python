"""
Defines an abstract base agent class for playing 2048.
"""
import random
import src.game.constants as c

from src.game.gamestate import BaseGameState, GameStateImpl

import numpy as np


class Base2048Agent(object):
    """
    An abstract class representing a base 2048 agent implementation.
    """

    def __init__(self):
        pass

    def evaluate(self, game_state):
        return NotImplemented

    def decide(self, game_state):
        return NotImplemented


class RandomAgent(Base2048Agent):
    def __init__(self):
        super().__init__()

    def decide(self, game_state):
        return random.choice([c.ACTION_UP, c.ACTION_DOWN, c.ACTION_LEFT, c.ACTION_RIGHT])


class DepthLimitedExpectimax(Base2048Agent):
    weight_grid = np.array([
        [-25, -25, -25, -25],
        [-10, -10, -10, -10],
        [-10, -10, -10, -10],
        [100, 10, 1, 0]
    ])

    def __init__(self):
        super().__init__()

    def decide(self, game_state: BaseGameState):
        """
        Implement depth-limited expectimax
        """
        mx = self._max_decision(game_state, depth=2)
        return mx

    def evaluate(self, game_state):
        if game_state.state() == 'lose':
            return -1000000

        alpha = 1
        beta = 2  # Weigh more to encourage merging of higher tiles
        gamma = 0.5
        delta = 1.7

        mat = game_state.matrix
        # Do this in one loop
        # beta_term = 0
        # gamma_term = 0
        # for row in mat:
        #     for x in row:
        #         beta_term += x**2
        #         gamma_term += (1 if x == 0 else 0)

        mat_dot_weight = sum([np.dot(w, m) for w, m in zip(DepthLimitedExpectimax.weight_grid, mat)])
        # return alpha * game_state.get_score() \
        #        + beta * beta_term \
        #        + gamma * game_state.get_score()**2 * gamma_term \
        #        + delta * mat_dot_weight

        return alpha * game_state.get_score() \
               + beta * sum([sum([x ** 2 for x in row]) for row in mat]) \
               + gamma * game_state.get_score() ** 2 * sum([sum([1 if x == 0 else 0 for x in row]) for row in mat]) \
               + delta * mat_dot_weight

        # This eval function can get us very close to 2048, but not past it. What's missing?

        # Other heuristic ideas to incorporate
        # * Keeping high valued tiles close to each other
        # * Keeping tiles in monotonically decreasing order along one or both axes
        # * Weight corners more (multipliers)

    def _max_decision(self, game_state: BaseGameState, depth=4):
        if depth < 0:
            raise ValueError('Depth cannot be less than 0')

        if depth == 0:
            return self.evaluate(game_state)

        # max over all player actions
        actions = game_state.get_allowed_actions(c.PLAYER)
        successor_states = [game_state.get_successor(act, c.PLAYER) for act in actions]
        arr = [self._expectation_value(st, depth) for st in successor_states]
        max_ = np.max(arr)
        indices = [i for i, x in enumerate(arr) if abs(x - max_) <= 0.001]
        return actions[random.choice(indices)]  # Randomly break ties

    def _expectation_value(self, game_state: BaseGameState, depth):
        # branch into all possible placements of 2-tile
        successor_states = game_state.get_successors(c.ADVERSARY)
        if len(successor_states) == 0:
            return self.evaluate(game_state)
        return np.average([self._max_value(st, depth - 1) for st in successor_states])

    def _max_value(self, game_state: BaseGameState, depth):
        if depth <= 1:
            # We stop at depth 1 because there is no point in expanding the expectation nodes at the last level.
            # It saves us some work, because all of those states with just a 2 added to them evaluate to exactly
            #  the same value. Therefore, the average of those values is that value.
            return self.evaluate(game_state)

        successor_states = game_state.get_successors(c.PLAYER)
        if len(successor_states) == 0:
            return self.evaluate(game_state)

        return np.max([self._expectation_value(st, depth) for st in successor_states])


class VariableDepthExpectimax(DepthLimitedExpectimax):
    def __init__(self):
        super().__init__()

    def decide(self, game_state: BaseGameState):
        """
        Implement depth-limited expectimax
        """

        # The depth of the expectimax tree will go up when there are fewer open squares, and go down when more.
        # This helps with optimization, but is also a heuristic, because the game generally requires more thought
        #  when the grid is more constrained.
        num_possible_adversary_actions = len(game_state.get_allowed_actions(c.ADVERSARY))
        # max of this value is 14, min is 0
        #  from 10-14, depth of 2. from 4-10, depth of 3, from 0-4, depth of 4
        depth = 0
        if num_possible_adversary_actions <= 16 and num_possible_adversary_actions >= 10:
            depth = 3
        elif num_possible_adversary_actions < 10 and num_possible_adversary_actions >= 5:
            depth = 3
        elif num_possible_adversary_actions < 5 and num_possible_adversary_actions >= 0:
            depth = 4

        mx = self._max_decision(game_state, depth=depth)
        return mx
