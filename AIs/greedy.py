from typing import Dict, Tuple

from pyrat.definitions import position, MazeGraph

TEAM_NAME = "Greedy"

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

import heapq

RouteMap = Dict[position, position]
DistanceMap = Dict[position, int]

def locationsToMove (location1: position, location2: position) -> str:
    difference = (location2[0] - location1[0], location2[1] - location1[1])
    if difference == (-1, 0) :
        return MOVE_LEFT
    elif difference == (1, 0) :
        return MOVE_RIGHT
    elif difference == (0, 1) :
        return MOVE_UP
    elif difference == (0, -1) :
        return MOVE_DOWN
    else :
        raise Exception("Invalid location provided")


def dijkstra (mazeMap: MazeGraph, initialLocation: position) -> Tuple[RouteMap, DistanceMap]:
    
    minHeap = [(0, initialLocation, None)]
    distances: DistanceMap = {}
    routes: RouteMap = {}
    
    # Main loop
    while len(minHeap) != 0 :
        (distance, location, predecessor) = heapq.heappop(minHeap)
        if location not in distances :
            distances[location] = distance
            routes[location] = predecessor
            for neighbor in mazeMap[location] :
                newDistanceToNeighbor = distance + mazeMap[location][neighbor]
                heapq.heappush(minHeap, (newDistanceToNeighbor, neighbor, location))
    
    # Result
    return (routes, distances)


def routesToPath (routes, targetNode) :
    
    # Recursive reconstruction
    if not targetNode :
        return []
    elif targetNode in routes :
        return routesToPath(routes, routes[targetNode]) + [targetNode]
    else :
        raise Exception("Impossible to reach target")

def pathToMoves (path) :
    
    # Recursive reconstruction
    if len(path) <= 1 :
        return []
    else :
        return [locationsToMove(path[0], path[1])] + pathToMoves(path[1:])

def findClosestPieceOfCheese (distances, piecesOfCheese) :
    
    # We return the cheese associated with the minimum distance
    distancesToCheese = {cheese : distances[cheese] for cheese in piecesOfCheese}
    return min(distancesToCheese, key=distancesToCheese.get)

def preprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed) :
    pass
    
def turn (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed) :
    
    # We use Dijkstra's algorithm from the current location
    (routes, distances) = dijkstra(mazeMap, playerLocation)
    
    # We find the closest pieces of cheese using the distances
    closestPieceOfCheese = findClosestPieceOfCheese(distances, piecesOfCheese)
    
    # Using the routes, we find the next move
    resultMoves = pathToMoves(routesToPath(routes, closestPieceOfCheese))
    return resultMoves[0]
    
def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed) :
    
    pass
