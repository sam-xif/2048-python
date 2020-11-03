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


class ExpectimaxAgent(Base2048Agent):
    def __init__(self):
        super().__init__()

    def decide(self, game_state: BaseGameState):
        """
        Implement depth-limited expectimax
        """
        mx = self._max_decision(game_state, depth=3)
        return mx

    def evaluate(self, game_state):
        mat = game_state.matrix
        return game_state.get_score() + np.sum([np.sum([np.log(x + 1) for x in row]) for row in mat])

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
        indices = [i for i, x in enumerate(arr) if x == max_]
        return random.choice(indices)  # Randomly break ties

    def _expectation_value(self, game_state: BaseGameState, depth):
        # branch into all possible placements of 2-tile
        successor_states = game_state.get_successors(c.ADVERSARY)
        if len(successor_states) == 0:
            return self.evaluate(game_state)
        return np.average([self._max_value(st, depth - 1) for st in successor_states])

    def _max_value(self, game_state: BaseGameState, depth):
        if depth == 0:
            return self.evaluate(game_state)

        successor_states = game_state.get_successors(c.PLAYER)
        if len(successor_states) == 0:
            return self.evaluate(game_state)

        return np.max([self._expectation_value(st, depth) for st in successor_states])
