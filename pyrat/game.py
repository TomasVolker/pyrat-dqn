from typing import List

from .definitions import position, add
from .components import Maze, Player, Observation, PlayerInfo


class PyratGame:

    def __init__(self, maze: Maze) -> None:

        self.maze = maze
        self.player1 = Player()
        self.player2 = Player()
        self.cheese_list: List[position] = []
        self.total_cheese = 0
        self.time_steps = 0
        self.observation = Observation(
            maze=self.maze,
            player1_info=PlayerInfo(current_position=(0, 0), score=0.0, is_winner=False),
            player2_info=PlayerInfo(current_position=(0, 0), score=0.0, is_winner=False),
            cheese_list=self.cheese_list,
            finished=False,
            time_steps=0
        )

    def clear(self) -> None:
        self.player1.clear()
        self.player2.clear()
        self.cheese_list = []
        self.time_steps = 0

    def is_position_empty(self, coordinates: position) -> bool:
        is_player_position = (self.player1.current_position == coordinates or
                              self.player2.current_position == coordinates)
        return not is_player_position and coordinates not in self.cheese_list

    def initialize(self, cheese_count: int, player1_position: position=None, player2_position: position=None) -> Observation:

        if cheese_count + 2 > self.maze.area():
            raise Exception("too many cheeses for the maze size")

        self.clear()

        self.player1.current_position = player1_position
        self.player2.current_position = player2_position

        self.total_cheese = cheese_count

        if player1_position is None:

            while True:
                candidate = self.maze.random_position()
                if self.is_position_empty(candidate):
                    self.player1.current_position = candidate
                    break

        if player2_position is None:

            while True:
                candidate = self.maze.random_position()
                if self.is_position_empty(candidate):
                    self.player2.current_position = candidate
                    break

        while cheese_count > 0:

            candidate = self.maze.random_position()

            if self.is_position_empty(candidate):
                self.cheese_list.append(candidate)
                cheese_count -= 1

        return self.update_observation()

    def get_delta(self, char: str) -> position:
        if char == 'L':
            return (-1, 0)
        elif char == 'R':
            return (1, 0)
        elif char == 'U':
            return (0, 1)
        elif char == 'D':
            return (0, -1)
        else:
            return (0, 0)


    def move_player(self, player: Player, delta: position) -> None:

        #Not blocked
        if player.blocked <= 0:

            current = player.current_position
            candidate = add(current, delta)
            distance = self.maze.graph[current].get(candidate, None)

            #Nodes are connected
            if distance is not None:
                player.current_movement = delta

                #No blocking
                if distance <= 1:
                    player.current_position = candidate
                #Blocking
                else:
                    player.blocked = distance-1

        #Unblocking
        elif player.blocked == 1:

            player.current_position = add(player.current_position, player.current_movement)

            player.blocked = 0
        #Blocked
        else:
            player.blocked -= 1

    def check_player_and_cheese_collision(self) -> None:

        #If they get the cheese at the same time
        if self.player1.current_position == self.player2.current_position:

            try:
                self.cheese_list.remove(self.player1.current_position)
                self.player1.score += 0.5
                self.player2.score += 0.5
            except ValueError:
                pass

        #Different places
        else:

            try:
                self.cheese_list.remove(self.player1.current_position)
                self.player1.score += 1
            except ValueError:
                pass

            try:
                self.cheese_list.remove(self.player2.current_position)
                self.player2.score += 1
            except ValueError:
                pass

    def is_there_a_winner(self) -> bool:

        if self.player1.score > self.total_cheese/2.0:
            return True
        elif self.player2.score > self.total_cheese/2.0:
            return True
        elif not self.cheese_list:
            return True
        else:
            return False

    def update_observation(self) -> Observation:

        self.observation.time_steps = self.time_steps

        self.observation.maze = self.maze

        self.observation.cheese_list = self.cheese_list
        self.observation.player1_info.set_from(self.player1)
        self.observation.player2_info.set_from(self.player2)

        self.observation.player1_info.is_winner = False
        self.observation.player2_info.is_winner = False

        self.observation.finished = False

        if self.is_there_a_winner():

            if self.player1.score > self.player2.score:
                self.observation.player1_info.is_winner = True
            elif self.player1.score < self.player2.score:
                self.observation.player2_info.is_winner = True

            self.observation.finished = True

        return self.observation

    def step(self, action1: str, action2: str) -> Observation:

        self.move_player(self.player1, self.get_delta(action1))
        self.move_player(self.player2, self.get_delta(action2))

        self.check_player_and_cheese_collision()

        self.time_steps += 1

        self.update_observation()

        return self.observation

    def get_observation(self) -> Observation:
        return self.observation

