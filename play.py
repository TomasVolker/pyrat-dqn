from agents.agent import Agent
from agents.greedy_agent import GreedyAgent

from pyrat.default_maze import default_maze
from pyrat.game import PyratGame

from time import sleep

from pyrat.components import State

RENDER = True

TIME_LIMIT = 200

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

EPISODE = 5000


def buildAgent1() -> Agent:
    # directory = "./dqn/episode-%d/" % EPISODE
    # return DqnAgent(maze, network_directory=("%sdqn_agent-%d" % (directory, EPISODE)))
    return GreedyAgent()


def buildAgent2() -> Agent:
    return GreedyAgent()


if __name__ == "__main__":

    player1 = buildAgent1()
    player2 = buildAgent2()

    env = PyratGame(maze=default_maze)

    observation = env.initialize(cheese_count=40)

    player1.prepare(State(observation, 1))
    player2.prepare(State(observation, 2))

    for i in range(TIME_LIMIT):

        if RENDER:
            print(observation)
            sleep(0.2)

        observation = env.step(
            action1=player1.act(State(observation, 1)),
            action2=player2.act(State(observation, 2)),
        )

        if observation.finished:
            break

    print(env.observation)



