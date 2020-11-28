from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax, QLearningAgent

from src.agent_tester import run_test

UI = False

if __name__ == '__main__':
    gamegrid = GameGrid(agent=QLearningAgent(
        2000, 0.00001, 'oscillate', 0.95, stop=2048,
        #weights=[-2832.2224747594496, -5657.742126834166, -6793.489373018032, -801.3072338667864, -5681.267006108246, -3657.628556709968, -3760.335227843232, -1084.449039300277, -5953.579225038403, -5112.241068704368, -2262.0733982705196, -2543.24326353655, -1131.0787503267268, -1825.5350704121722, -1108.7653559834212, 6559.395455920096]
    ))
