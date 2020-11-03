from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax

if __name__ == '__main__':
    gamegrid = GameGrid(agent=VariableDepthExpectimax())
