from src.game.puzzle import GameGrid
from src.agents.agents import RandomAgent, DepthLimitedExpectimax, VariableDepthExpectimax, QLearningAgent

from src.agent_tester import run_test

UI = False

if __name__ == '__main__':
    gamegrid = GameGrid(agent=QLearningAgent(
        2000, 0.00001, 'decaylinear', 0.95, stop=2048,
        feature_set='basic_plus_rows'
        # weights=[-6697.210478914676, -8096.405527146442, -7870.483575571659, -4360.95375572444,
        #          -11071.292221002843, -7443.233588948385, -5227.816925377352, -3721.621357580199,
        #          -10958.228818896632, -7812.524233283419, -2746.94675685422, 3608.6548148178213,
        #          -6651.664310821655, -7222.486864040597, -1732.113352032885, 8691.588031201189]
    ))
