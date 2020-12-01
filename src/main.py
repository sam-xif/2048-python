import random
from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax, QLearningAgent, IDEQLearningAgent

from src.agent_tester import run_test

UI = False

if __name__ == '__main__':
    random.seed(42)
    gamegrid = GameGrid(agent=VariableDepthExpectimax())
    #     100000, 0.00001, 0.3, 0.98, stop=2048,
    #     feature_set='basic_plus_rows',
    #     # weights after playing 31163 games.
    #     weights=[-3112.2445061215803, -3242.6803219632898, -1306.5932108730733, 5944.4838639591599, -3902.9822137008841, -1974.753017512375, -995.74281492784473, 1749.3690628341419, -3060.3139077837345, -1685.5336537368482, -1468.9915063335682, 274.44448396863368, -113.78404744241017, -3252.9594550289557, -3463.0218415816266, 1221.0429779323053, -1717.0341749992574, -5124.1089833069, -5940.3945838858845, -5608.7223661220805]
    # ))
