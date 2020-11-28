"""
Defines an abstract base agent class for playing 2048.
"""
import random
import src.game.constants as c

from src.game.gamestate import BaseGameState, GameStateImpl

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


class QLearningAgent(Base2048Agent):
    def __init__(self, epochs, alpha, epsilon, discount, stop=2048, weights=None):
        """

        :param epochs:
        :param alpha:
        :param epsilon:
        :param discount:
        :param stop: Train on games that have a win condition when the tile with value `stop` is reached
        """
        super().__init__()
        self.epochs = epochs
        self.alpha = alpha
        if isinstance(epsilon, type(int)):
            self.epsilon = lambda x: epsilon
        elif epsilon == 'decay':
            self.epsilon = lambda x: np.log(self.epochs - x + 1) / np.log(self.epochs + 1)
        elif epsilon == 'oscillate':
            self.epsilon = lambda x: (np.cos(x * 2*np.pi / (self.epochs / 10)) + 1) / 2
        self.discount = discount
        self.stop = stop
        if weights is None:
            self.weights = [1 for i in range(16)]
            self.train()
        else:
            self.weights = weights

    def evaluate(self, game_state):
        # component-wise multiplies weights and board values
        arr = np.log(np.ravel(game_state.matrix) + 1)
        return np.dot(self.weights, arr)

    def _update_weights(self, game_state: BaseGameState, action, td_error):
        intermediate_state = game_state.get_successor(action, c.PLAYER)
        next_states = intermediate_state.get_successors(c.ADVERSARY)
        next_state_matrices = list(map(lambda st: np.ravel(st.matrix), next_states))

        for i, weight in enumerate(self.weights):
            feature_value = np.average(np.array(list(map(lambda mat: np.log(mat[i] + 1), next_state_matrices))))
            term = (self.alpha * td_error * feature_value)

            self.weights[i] = weight + term

    def q(self, game_state: BaseGameState, action):
        # return the average difference between new_score of successors and prev_score
        intermediate_state = game_state.get_successor(action, c.PLAYER)
        next_states = intermediate_state.get_successors(c.ADVERSARY)
        return np.average(np.array(list(map(lambda st: self.evaluate(st), next_states))))

    def _reward(self, old_game_state: BaseGameState, new_game_state: BaseGameState):
        if new_game_state.state() == 'win':
            return 1000000
        elif new_game_state.state() == 'lose':
            return -1000000

        # Calculate the square of the score delta.
        # It is squared to encode the increasing marginal returns of merging more tiles
        return (new_game_state.get_score() - old_game_state.get_score())**2

    def _td_error(self, old_state: BaseGameState, action, new_state: BaseGameState):
        reward = self._reward(old_state, new_state)
        # (r + gamma max over a' Q(s', a')) - Q(s, a)
        if len(new_state.get_allowed_actions(c.PLAYER)) == 0:
            max_q = 0
        else:
            max_q = max([self.q(new_state, act) for act in new_state.get_allowed_actions(c.PLAYER)])

        td_error = (reward
                   + self.discount
                   * max_q) \
                   - self.q(old_state, action)
        return td_error

    def train(self):
        for epoch in range(self.epochs):
            print('training epoch', epoch, '; epsilon =', self.epsilon(epoch))
            game_state = GameStateImpl(stop=self.stop)

            while game_state.state() == 'not over':
                # Make a move, experience reward, update weights
                old_game_state = game_state.clone()
                chosen_action = None

                explore_exploit = random.random()
                if explore_exploit < self.epsilon(epoch):
                    # explore (choose random action)
                    chosen_action = random.choice(game_state.get_allowed_actions(c.PLAYER))
                else:
                    # exploit (take action with highest q-value)
                    best_action = max(game_state.get_allowed_actions(c.PLAYER),
                                      key=lambda act: self.q(game_state, act))
                    chosen_action = best_action

                # Complete move
                game_state.execute_action(chosen_action, c.PLAYER)
                game_state.add_new_tile()

                # experience reward, calculate td_error
                td_error = self._td_error(old_game_state, chosen_action, game_state)

                self._update_weights(old_game_state, chosen_action, td_error)

            print(game_state.state(), game_state.matrix)
            print('epoch finished. weights:', self.weights)

    def decide(self, game_state):
        # exploit (take action with highest q-value)
        best_action = max(game_state.get_allowed_actions(c.PLAYER),
                          key=lambda act: self.q(game_state, act))
        return best_action


