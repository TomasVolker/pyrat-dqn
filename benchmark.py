from agents.agent import Agent
from agents.greedy_agent import GreedyAgent
from pyrat.default_maze import default_maze
from pyrat.game import PyratGame
from pyrat.components import *

TIME_LIMIT = 200

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

EPISODE = 80000

MATCHES = 100


def buildAgent1() -> Agent:
    # directory = "./dqn/episode-%d/" % EPISODE
    # return DqnAgent(maze, network_directory=("%sdqn_agent-%d" % (directory, EPISODE)))
    return GreedyAgent()


def buildAgent2() -> Agent:
    return GreedyAgent()


if __name__ == "__main__":

    directory = "./dqn/episode-%d/" % EPISODE

    player1 = buildAgent1()
    player2 = buildAgent2()

    player1_wins = 0
    ties = 0

    env = PyratGame(maze=default_maze)

    for match in range(MATCHES):

        observation = env.initialize(cheese_count=40)

        player1.prepare(State(observation, 1))
        player2.prepare(State(observation, 2))

        for i in range(TIME_LIMIT):

            observation = env.step(
                action1=player1.act(State(observation, 1)),
                action2=player2.act(State(observation, 2)),
            )

            if observation.finished:
                break

        result = "\t Lost"

        if observation.player1_info.is_winner:
            player1_wins += 1
            result = "\t Won"
        elif not observation.player2_info.is_winner:
            ties += 1
            result = "\t Tied"

        print("Match: {0:d} \t Player1 score: {1:.1f} \t Player2 score: {2:.1f} {3:s}"
            .format(
                match+1,
                observation.player1_info.score,
                observation.player2_info.score,
                result
            )
        )

    print("Matches: {0:d} \t Wins: {1:d} \t Ties: {2:d} \t Losses: {3:d}\t"
        .format(
            MATCHES,
            player1_wins,
            ties,
            MATCHES - player1_wins - ties
        )
    )

