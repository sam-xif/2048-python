"""
Defines functions for testing and generating reports on agent performance
"""

from src.agents.agents import RandomAgent
from src.game.gamestate import GameStateImpl

import time
import src.game.constants as c

import pickle

def run_test(num_trials=100, with_ui=False, agent=RandomAgent):
    infos = []
    for i in range(num_trials):
        print('running trial', i)
        st = time.time()
        gs = GameStateImpl()
        turn = 0
        while gs.state() != 'lose':
            if turn % 100 == 0:
                print('turn', turn)
            act = agent.decide(gs)
            moved = gs.execute_action(act, c.PLAYER)
            if moved:
                gs.add_new_tile()

            turn += 1

        et = time.time()

        highest_tile = max([max(row) for row in gs.matrix])
        info_tuple = (i, et - st, gs.get_score(), turn, highest_tile, gs.matrix)
        infos.append(info_tuple)
        print('trial {} complete. time elapsed: {:0.2f}s, score: {}, highest tile: {}, num of turns: {}\nfinal matrix: {}'
              .format(*info_tuple))

        with open('out.pkl', 'wb+') as f:
            pickle.dump(infos, f)

