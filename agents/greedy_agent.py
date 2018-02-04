
import random

import heapq
from typing import Dict, Tuple, List

import numpy as np

from agents.agent import Agent
from pyrat.definitions import position, MazeGraph
from pyrat.components import State

RouteMap = Dict[position, position]
DistanceMap = Dict[position, int]

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

char_to_index: Dict[str, int] = {
    'R': 0,
    'L': 1,
    'U': 2,
    'D': 3
}

index_to_char: Dict[int, str] = {
    0: 'R',
    1: 'L',
    2: 'U',
    3: 'D'
}


class GreedyAgent(Agent):

    def prepare(self, state: State):
        pass

    def act(self, state: State, exploration_rate: float=0.0) -> str:

        if np.random.rand() < exploration_rate:
            return index_to_char[random.randrange(4)]

        # We use Dijkstra's algorithm from the current location
        (routes, distances) = self.dijkstra(state.maze.graph, state.player_info.current_position)

        # We find the closest pieces of cheese using the distances
        closestPieceOfCheese = self.findClosestPieceOfCheese(distances, state.cheese_list)

        # Using the routes, we find the next move
        resultMoves = self.pathToMoves(self.routesToPath(routes, closestPieceOfCheese))

        return resultMoves[0]

    def locationsToMove(self, location1: position, location2: position) -> str:
        difference = (location2[0] - location1[0], location2[1] - location1[1])
        if difference == (-1, 0):
            return MOVE_LEFT
        elif difference == (1, 0):
            return MOVE_RIGHT
        elif difference == (0, 1):
            return MOVE_UP
        elif difference == (0, -1):
            return MOVE_DOWN
        else:
            raise Exception("Invalid location provided")

    def dijkstra(self, mazeMap: MazeGraph, initialLocation: position) -> Tuple[RouteMap, DistanceMap]:
        minHeap: List[Tuple[int, position, position]] = [(0, initialLocation, None)]
        distances: DistanceMap = {}
        routes: RouteMap = {}

        # Main loop
        while len(minHeap) != 0:
            (distance, location, predecessor) = heapq.heappop(minHeap)
            if location not in distances:
                distances[location] = distance
                routes[location] = predecessor
                for neighbor in mazeMap[location]:
                    newDistanceToNeighbor = distance + mazeMap[location][neighbor]
                    heapq.heappush(minHeap, (newDistanceToNeighbor, neighbor, location))

        # Result
        return (routes, distances)

    def routesToPath(self, routes: RouteMap, targetNode: position) -> List[position]:
        # Recursive reconstruction
        if not targetNode:
            return []
        elif targetNode in routes:
            return self.routesToPath(routes, routes[targetNode]) + [targetNode]
        else:
            raise Exception("Impossible to reach target")

    def pathToMoves(self, path: List[position]) -> List[str]:
        # Recursive reconstruction
        if len(path) <= 1:
            return []
        else:
            return [self.locationsToMove(path[0], path[1])] + self.pathToMoves(path[1:])

    def findClosestPieceOfCheese(self, distances: DistanceMap, piecesOfCheese: List[position]) -> position:
        # We return the cheese associated with the minimum distance
        distancesToCheese = {cheese: distances[cheese] for cheese in piecesOfCheese}
        return min(distancesToCheese, key=distancesToCheese.get)


