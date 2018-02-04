from typing import Tuple, Dict

position = Tuple[int, int]

MazeGraph = Dict[position, Dict[position, int]]


def add(coordinates1: position, coordinates2: position) -> position:
    return (coordinates1[0] + coordinates2[0], coordinates1[1] + coordinates2[1])

