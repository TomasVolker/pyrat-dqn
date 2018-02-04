import random

from agents.agent import Agent
from pyrat.components import State

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

class RandomAgent(Agent):

    def act(self, state: State) -> str:
        return random.choice([MOVE_UP, MOVE_RIGHT, MOVE_LEFT, MOVE_DOWN])

