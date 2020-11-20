"""
Defines functions for testing and generating reports on agent performance
"""

from src.agents.agents import RandomAgent, VariableDepthExpectimax
from src.game.gamestate import GameStateImpl

import time
import src.game.constants as c

import pickle
from multiprocessing import Pool, Lock


def run_test(num_trials=100, with_ui=False, agent=RandomAgent):
    print_lock = Lock()
    pkl_lock = Lock()
    params = range(num_trials)
    p = Pool(8, initializer=init, initargs=(print_lock, pkl_lock))
    infos = p.imap_unordered(test_runner, params)
    with open('out.pkl', 'wb+') as f:
        pickle.dump(list(infos), f)
    p.close()
    p.join()

    # for i in range(num_trials):
    #     print('running trial', i)
    #     st = time.time()
    #     gs = GameStateImpl()
    #     turn = 0
    #     while gs.state() != 'lose':
    #         if turn % 100 == 0:
    #             print('turn', turn)
    #         act = agent.decide(gs)
    #         moved = gs.execute_action(act, c.PLAYER)
    #         if moved:
    #             gs.add_new_tile()
    #
    #         turn += 1
    #
    #     et = time.time()
    #
    #     highest_tile = max([max(row) for row in gs.matrix])
    #     info_tuple = (i, et - st, gs.get_score(), highest_tile, turn, gs.matrix)
    #     infos.append(info_tuple)
    #     print('trial {} complete. time elapsed: {:0.2f}s, score: {}, highest tile: {}, num of turns: {}\nfinal matrix: {}'
    #           .format(*info_tuple))
    #
    #     with open('out.pkl', 'wb+') as f:
    #         pickle.dump(infos, f)


def init(print_l, pkl_l):
    global print_lock
    global pkl_lock
    print_lock = print_l
    pkl_lock = pkl_l


def print_with_lock(lock, *args, **kwargs):
    lock.acquire()
    try:
        print(*args, **kwargs)
    finally:
        lock.release()


def test_runner(index):
    agent = VariableDepthExpectimax()
    print_with_lock(print_lock, 'running trial', index)
    st = time.time()
    gs = GameStateImpl()
    turn = 0
    while gs.state() != 'lose':
        if turn % 10 == 0:
            print_with_lock(print_lock, 'TRIAL', index, ': turn', turn)
        act = agent.decide(gs)
        moved = gs.execute_action(act, c.PLAYER)
        if moved:
            gs.add_new_tile()

        turn += 1

    et = time.time()

    highest_tile = max([max(row) for row in gs.matrix])
    info_tuple = (index, et - st, gs.get_score(), highest_tile, turn, gs.matrix)
    print_with_lock(print_lock, 'trial {} complete. time elapsed: {:0.2f}s, score: {}, highest tile: {}, num of turns: {}\nfinal matrix: {}'
          .format(*info_tuple))

    return info_tuple
