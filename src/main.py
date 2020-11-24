from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax, MonteCarloAgent

from src.agent_tester import run_test

UI = False

if __name__ == '__main__':
    gamegrid = GameGrid(agent=MonteCarloAgent())
    # gamegrid = GameGrid(agent=VariableDepthExpectimax())
