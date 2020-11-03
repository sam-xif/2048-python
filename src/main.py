from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax

from src.agent_tester import run_test

UI = True

if __name__ == '__main__':
    gamegrid = GameGrid(agent=VariableDepthExpectimax())
