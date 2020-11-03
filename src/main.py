from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, ExpectimaxAgent

if __name__ == '__main__':
    gamegrid = GameGrid(agent=ExpectimaxAgent())
