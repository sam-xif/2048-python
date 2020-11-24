"""
Defines an abstract base agent class for playing 2048.
"""
import random
import src.game.constants as c

from src.game.gamestate import BaseGameState, GameStateImpl

import math
import numpy as np

def eval_function(game_state):
    # Other heuristic ideas to incorporate
    # * Keeping high valued tiles close to each other
    # * Keeping tiles in monotonically decreasing order along one or both axes
    # * Weight corners more (multipliers)

    # Considerations:
    # * delta cannot be too high because the agent will then avoid risks in order to keep the numbers
    #   in a specific configuration. Sometimes the agent actually needs to deviate for a few moves to
    #   make important merges.

    if game_state.state() == 'lose':
        return -10000000

    # weight = [[1, 2, 3, 4], [8, 7, 6, 5], [9, 10, 11, 12], [16, 15, 14, 13]]
    # weight = [[1, 2**1, 2**2, 2**3], [2**7, 2**6, 2**5, 2**4], [2**8, 2**9, 2**10, 2**11], [4**11, 4**10, 4**9, 4**8]]
    weight = [[1, 2 ** 1, 2 ** 2, 2 ** 3], [2 ** 7, 2 ** 6, 2 ** 5, 2 ** 4], [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11],
              [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12]]
    mat = game_state.matrix
    weightValue = 0
    sameWeightNeighbour = 0

    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            weightValue += (weight[i][j] * mat[i][j])

            if i - 1 >= 0 and mat[i - 1][j] == mat[i][j]:
                sameWeightNeighbour += 1

            if i + 1 < c.GRID_LEN and mat[i + 1][j] == mat[i][j]:
                sameWeightNeighbour += 1

            if j - 1 >= 0 and mat[i][j - 1] == mat[i][j]:
                sameWeightNeighbour += 1

            if j + 1 < c.GRID_LEN and mat[i][j + 1] == mat[i][j]:
                sameWeightNeighbour += 1

    return weightValue + game_state.get_score() \
           + 2 ** 6 * (np.sum([np.sum([1 if x == 0 else 0 for x in row]) for row in mat])) \
           + 2 ** 4 * (sameWeightNeighbour / 2)


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
    def __init__(self):
        super().__init__()

    def decide(self, game_state: BaseGameState):
        """
        Implement depth-limited expectimax
        """
        mx = self._max_decision(game_state, depth=2)
        return mx

    def evaluate(self, game_state):
        return eval_function(game_state)

    def _max_decision(self, game_state: BaseGameState, depth=4):
        if depth < 0:
            raise ValueError('Depth cannot be less than 0')

        if depth == 0:
            return self.evaluate(game_state)
        # print("Matrix", game_state.matrix)
        # input("Press Enter to continue...")
        # max over all player actions
        actions = game_state.get_allowed_actions(c.PLAYER)
        successor_states = [game_state.get_successor(act, c.PLAYER) for act in actions]
        arr = [self._expectation_value(st, depth) for st in successor_states]
        max_ = np.max(arr)
        print('MEUs of actions:', arr, actions)
        indices = [i for i, x in enumerate(arr) if abs(x - max_) <= 0.001]
        return actions[random.choice(indices)]  # Randomly break ties

    def _expectation_value(self, game_state: BaseGameState, depth):
        if depth <= 1:
            return self.evaluate(game_state)

        # branch into all possible placements of 2-tile
        successor_states = game_state.get_successors(c.ADVERSARY)
        if len(successor_states) == 0:
            return self.evaluate(game_state)
        return np.average([self._max_value(st, depth - 1) for st in successor_states])

    def _max_value(self, game_state: BaseGameState, depth):
        if depth < 1:
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
            depth = 2
        elif num_possible_adversary_actions < 10 and num_possible_adversary_actions >= 5:
            depth = 3
        elif num_possible_adversary_actions < 5 and num_possible_adversary_actions >= 0:
            depth = 4

        mx = self._max_decision(game_state, depth=depth)
        return mx

class MonteCarloAgent(Base2048Agent):
    def __init__(self):
        super().__init__()
        # Stores number of games played for each state
        self.play = {}
        # Stores number of games won for each state
        self.win = {}
        self.numGames = 1000
        self.run_simulation()

    def run_simulation(self):        
        for i in range(self.numGames):
            game_state = GameStateImpl()
            visitedState = []
            visitedState.append(game_state.toString())

            while len(game_state.get_allowed_actions(c.PLAYER)) != 0: # game is not over yet
                if game_state.toString() in self.play:
                    actions = game_state.get_allowed_actions(c.PLAYER)
                    maxValue = float('-inf')

                    for action in actions:
                        successor_state = game_state.get_successor(action, c.PLAYER) 

                        if successor_state.toString() not in self.play:
                            self.play[successor_state.toString()] = 0
                            self.win[successor_state.toString()] = 0

                        # select naxt action based on UCB formula
                        ucb = self.getUCBValue(game_state, successor_state)                        
                        if ucb > maxValue:
                            maxValue = ucb
                            selectedAction = action
                else:
                    actions = game_state.get_allowed_actions(c.PLAYER)
                    if game_state.toString() not in self.play:
                        self.play[game_state.toString()] = 0
                        self.win[game_state.toString()] = 0

                    for action in actions:
                        successor_state = game_state.get_successor(action, c.PLAYER) 
                        if successor_state.toString() not in self.play:
                            self.play[successor_state.toString()] = 0
                            self.win[successor_state.toString()] = 0

                    # randomly expands tree
                    selectedAction = actions[random.randint(0, len(actions) - 1)]
                
                moved = game_state.execute_action(selectedAction, c.PLAYER)
                if moved:
                    game_state.add_new_tile()
                visitedState.append(game_state.toString())

            # Backpropagation
            for stateMatrix in visitedState:
                if stateMatrix not in self.play:
                    self.play[stateMatrix] = 1
                else:
                    self.play[stateMatrix] += 1
                if game_state.toString().__contains__("512"): # currently takes 512 as winning state
                    if stateMatrix not in self.win:
                        self.win[stateMatrix] = 1
                    else:
                        self.win[stateMatrix] += 1              

    def decide(self, game_state: BaseGameState):
        """
        Implement depth-limited expectimax
        """
        if game_state.toString() in self.play:
            actions = game_state.get_allowed_actions(c.PLAYER)
            maxValue = float('-inf')

            for action in actions:
                successor_state = game_state.get_successor(action, c.PLAYER) 
                if successor_state.toString() not in self.win or self.win[successor_state.toString()] == 0 \
                    or successor_state.toString() not in self.play or self.play[successor_state.toString()] == 0:
                    winWeight = 0
                else:
                    winWeight = self.win[successor_state.toString()] / self.play[successor_state.toString()]

                # decides action based on winweight (#wins / #gamesPlayed)
                if winWeight > maxValue:
                    maxValue = winWeight
                    selectedAction = action
        else:
            actions = game_state.get_allowed_actions(c.PLAYER)    
            selectedAction = actions[random.randint(0, len(actions) - 1)]

        return selectedAction

    def getUCBValue(self, game_state, successor_state):
        const = math.sqrt(2)

        if game_state.toString() not in self.play or self.play[game_state.toString()] == 0 \
            or successor_state.toString() not in self.play or self.play[successor_state.toString()] == 0:
            ucb = float('inf')
        else:
            ucb = (self.win[successor_state.toString()] / self.play[successor_state.toString()]) \
                + (const * math.sqrt(math.log(self.play[game_state.toString()]) / self.play[successor_state.toString()]))
        
        return ucb
