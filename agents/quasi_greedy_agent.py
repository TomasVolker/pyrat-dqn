
import random

import heapq

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'


class QuasiGreedyAgent:

    def __init__(self):
        self.first_piece_taken = False
        self.first_piece_target = (-1, -1)


    def act(self, state):

        if self.first_piece_target != (-1, -1) and self.first_piece_target not in state.cheese_list:
            self.first_piece_taken = True

        # We use Dijkstra's algorithm from the current location
        (routes, distances) = self.dijkstra(state.maze.graph, state.player_info.position)

        if self.first_piece_taken:
            closestPieceOfCheese = self.findClosestPieceOfCheese(distances, state.cheese_list)

        else:
            if self.first_piece_target == (-1, -1):
                self.first_piece_target = state.cheese_list[random.randint(0, len(state.cheese_list))]
            closestPieceOfCheese = self.first_piece_target

        # Using the routes, we find the next move
        resultMoves = self.pathToMoves(self.routesToPath(routes, closestPieceOfCheese))
        return resultMoves[0]


    def locationsToMove(self, location1, location2):
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

    def dijkstra(self, mazeMap, initialLocation):

        minHeap = [(0, initialLocation, None)]
        distances = {}
        routes = {}

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

    def routesToPath(self, routes, targetNode):

        # Recursive reconstruction
        if not targetNode:
            return []
        elif targetNode in routes:
            return self.routesToPath(routes, routes[targetNode]) + [targetNode]
        else:
            raise Exception("Impossible to reach target")

    def pathToMoves(self, path):

        # Recursive reconstruction
        if len(path) <= 1:
            return []
        else:
            return [self.locationsToMove(path[0], path[1])] + self.pathToMoves(path[1:])

    def findClosestPieceOfCheese(self, distances, piecesOfCheese):

        # We return the cheese associated with the minimum distance
        distancesToCheese = {cheese: distances[cheese] for cheese in piecesOfCheese}
        return min(distancesToCheese, key=distancesToCheese.get)

    def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
        pass
