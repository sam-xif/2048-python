"""
Defines functions for testing and generating reports on agent performance
"""

from src.agents.agents import RandomAgent
from src.game.gamestate import GameStateImpl

import src.game.constants as c

def run_test(num_trials=100, with_ui=False, agent=RandomAgent):
    infos = []
    for i in range(num_trials):
        print('running trial', i)
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

        highest_tile = max([max(row) for row in gs.matrix])
        infos.append((gs.score(), turn, highest_tile, gs.matrix))
        print('trial complete. score:', gs.score(), '; highest tile:', highest_tile)