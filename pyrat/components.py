
import random
from typing import List

from copy import deepcopy

from .definitions import position, MazeGraph


class Maze:

    def __init__(self, graph: MazeGraph, width: int, height: int) -> None:
        self.graph = graph
        self.height = height
        self.width = width

    def random_position(self) -> position:
        return (random.randrange(0, self.width), random.randrange(0, self.height))

    def area(self) -> int:
        return self.width * self.height


class Player:

    def __init__(self, current_position: position=None) -> None:
        self.current_position = current_position
        self.score: float = 0.0
        self.blocked: float = 0.0
        self.current_movement: position = None

    def clear(self) -> None:
        self.current_position = None
        self.score = 0
        self.blocked = 0
        self.current_movement = None


class PlayerInfo:

    def __init__(self, current_position: position, score: float, is_winner: bool) -> None:
        self.current_position = current_position
        self.score = score
        self.is_winner = is_winner

    def set_from(self, player: Player) -> None:
        self.current_position = player.current_position
        self.score = player.score


class Observation:

    def __init__(
            self,
            maze: Maze,
            player1_info: PlayerInfo,
            player2_info: PlayerInfo,
            cheese_list: List[position],
            finished: bool,
            time_steps: int) -> None:
        self.maze = maze
        self.player1_info = player1_info
        self.player2_info = player2_info
        self.cheese_list = cheese_list
        self.finished = finished
        self.time_steps = time_steps

    def __str__(self) -> str:

        result = "Player 1 score: {0:.1f} - Player 2 score: {1:.1f}\nTime: {2:d}\n"\
            .format(
            self.player1_info.score,
            self.player2_info.score,
            self.time_steps
        )

        result += "Game finished: {0}\n".format(self.finished)

        if self.finished:
            result += "Player 1 win: {0} \t Player 2 win: {1}\n"\
                .format(
                self.player1_info.is_winner,
                self.player2_info.is_winner
            )

        for x in range(0, self.maze.width):
            result += '+-'

        result += '+\n'

        for y in range(self.maze.height-1, -1, -1):

            result += '|'

            for x in range(0, self.maze.width):

                if self.player1_info.current_position == (x, y):
                    result += 'R'
                elif self.player2_info.current_position == (x, y):
                    result += 'P'
                elif (x, y) in self.cheese_list:
                    result += 'C'
                else:
                    result += ' '

                if (x+1, y) in self.maze.graph.get((x, y), []):

                    if self.maze.graph.get((x, y)).get((x+1, y)) > 1:
                        result += '.'
                    else:
                        result += ' '

                else:
                    result += '|'

            result += '\n+'

            for x in range(0, self.maze.width):

                if (x, y-1) in self.maze.graph.get((x, y), []):

                    if self.maze.graph.get((x, y)).get((x, y-1)) > 1:
                        result += '.+'
                    else:
                        result += ' +'

                else:
                    result += '-+'

            result += '\n'

        return result


class State:

    def __init__(
            self,
            observation: Observation,
            player: int=1,
            previous_state: 'State'=None
    ) -> None:
        self.maze = observation.maze
        self.cheese_list = deepcopy(observation.cheese_list)
        self.finished = observation.finished
        self.time_steps = observation.time_steps

        self.positions_saved = 4

        if player == 1:
            self.player_info = deepcopy(observation.player1_info)
            self.opponent_info = deepcopy(observation.player2_info)
        elif player == 2:
            self.player_info = deepcopy(observation.player2_info)
            self.opponent_info = deepcopy(observation.player1_info)

        self.last_positions = [self.player_info.current_position for i in range(self.positions_saved)]

        if previous_state is not None:

            for i in range(1, self.positions_saved):
                self.last_positions[i] = previous_state.last_positions[i-1]

