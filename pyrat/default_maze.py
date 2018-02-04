from .components import Maze
from .definitions import MazeGraph

maze_graph: MazeGraph = {
(7, 3): {(6, 3): 1, (7, 4): 1, (8, 3): 1},
(6, 9): {
    (5, 9): 1,
    (6, 10): 1,
    (6, 8): 1,
    (7, 9): 2,
    },
(12, 1): {(13, 1): 2, (12, 2): 1, (12, 0): 1},
(11, 11): {
    (11, 12): 1,
    (11, 10): 1,
    (10, 11): 1,
    (12, 11): 1,
    },
(16, 6): {(17, 6): 1, (16, 5): 2, (16, 7): 1},
(7, 12): {(6, 12): 1, (7, 11): 1, (8, 12): 1},
(14, 4): {(15, 4): 1, (14, 3): 1, (14, 5): 1},
(13, 4): {(12, 4): 2, (13, 3): 2},
(12, 12): {
    (12, 13): 1,
    (13, 12): 1,
    (11, 12): 1,
    (12, 11): 1,
    },
(19, 4): {(20, 4): 2, (18, 4): 1, (19, 5): 1},
(18, 4): {
    (19, 4): 1,
    (18, 5): 1,
    (18, 3): 1,
    (17, 4): 1,
    },
(0, 7): {(0, 8): 1, (1, 7): 1},
(15, 1): {(16, 1): 1, (14, 1): 1, (15, 2): 1},
(20, 7): {(19, 7): 1, (20, 6): 1},
(1, 6): {
    (0, 6): 1,
    (2, 6): 1,
    (1, 5): 1,
    (1, 7): 1,
    },
(0, 10): {(0, 9): 1, (1, 10): 2},
(3, 7): {
    (2, 7): 1,
    (3, 8): 2,
    (4, 7): 1,
    (3, 6): 1,
    },
(2, 5): {(1, 5): 1, (3, 5): 1},
(1, 11): {(0, 11): 1, (2, 11): 1},
(8, 5): {
    (8, 4): 1,
    (8, 6): 1,
    (7, 5): 1,
    (9, 5): 1,
    },
(5, 8): {(5, 9): 1, (5, 7): 1, (6, 8): 1},
(4, 0): {(3, 0): 1, (4, 1): 2, (5, 0): 1},
(10, 8): {
    (11, 8): 1,
    (10, 7): 1,
    (9, 8): 1,
    (10, 9): 1,
    },
(9, 0): {(10, 0): 1, (8, 0): 1, (9, 1): 1},
(6, 7): {
    (5, 7): 1,
    (6, 6): 1,
    (6, 8): 1,
    (7, 7): 1,
    },
(5, 5): {(4, 5): 1, (5, 6): 2, (5, 4): 1},
(11, 5): {(12, 5): 1, (10, 5): 1, (11, 6): 2},
(10, 7): {
    (11, 7): 2,
    (10, 6): 1,
    (9, 7): 2,
    (10, 8): 1,
    },
(16, 3): {(15, 3): 1, (17, 3): 1},
(6, 10): {(6, 11): 1, (6, 9): 1, (5, 10): 1},
(12, 6): {(13, 6): 1, (12, 5): 1, (12, 7): 1},
(19, 14): {(18, 14): 1, (19, 13): 1},
(8, 8): {(8, 9): 1, (7, 8): 1, (8, 7): 1},
(17, 2): {
    (18, 2): 1,
    (17, 3): 1,
    (17, 1): 1,
    (16, 2): 1,
    },
(15, 11): {
    (15, 12): 1,
    (14, 11): 1,
    (16, 11): 1,
    (15, 10): 1,
    },
(14, 1): {
    (13, 1): 1,
    (14, 2): 1,
    (14, 0): 1,
    (15, 1): 1,
    },
(13, 7): {
    (13, 6): 1,
    (12, 7): 1,
    (14, 7): 1,
    (13, 8): 1,
    },
(20, 9): {(19, 9): 2, (20, 8): 1},
(19, 3): {(18, 3): 1, (20, 3): 1},
(18, 9): {(19, 9): 1, (17, 9): 1},
(0, 4): {(0, 3): 1, (1, 4): 1},
(15, 4): {
    (14, 4): 1,
    (15, 5): 1,
    (16, 4): 1,
    (15, 3): 1,
    },
(20, 4): {(20, 5): 1, (19, 4): 2},
(1, 1): {(0, 1): 1, (1, 0): 1, (2, 1): 1},
(4, 10): {(3, 10): 2, (5, 10): 1},
(3, 2): {
    (4, 2): 1,
    (3, 1): 1,
    (3, 3): 1,
    (2, 2): 1,
    },
(2, 6): {(2, 7): 2, (1, 6): 1},
(9, 14): {(8, 14): 1, (10, 14): 1},
(8, 2): {
    (8, 1): 1,
    (9, 2): 1,
    (8, 3): 1,
    (7, 2): 1,
    },
(5, 11): {(5, 12): 2, (5, 10): 1, (4, 11): 1},
(4, 5): {(4, 4): 1, (5, 5): 1},
(10, 13): {(9, 13): 1, (10, 12): 1, (11, 13): 1},
(9, 3): {
    (9, 2): 1,
    (8, 3): 1,
    (10, 3): 1,
    (9, 4): 1,
    },
(6, 0): {(6, 1): 1, (7, 0): 2},
(11, 0): {(10, 0): 1, (12, 0): 1},
(7, 5): {
    (7, 4): 1,
    (7, 6): 1,
    (8, 5): 1,
    (6, 5): 2,
    },
(12, 11): {
    (12, 10): 1,
    (12, 12): 1,
    (11, 11): 1,
    (13, 11): 1,
    },
(19, 13): {(20, 13): 1, (18, 13): 1, (19, 14): 1},
(17, 13): {
    (17, 14): 1,
    (18, 13): 1,
    (16, 13): 2,
    (17, 12): 1,
    },
(15, 14): {(16, 14): 1, (15, 13): 1},
(14, 2): {
    (14, 3): 2,
    (14, 1): 1,
    (13, 2): 1,
    (15, 2): 2,
    },
(13, 10): {
    (12, 10): 1,
    (14, 10): 1,
    (13, 11): 1,
    (13, 9): 1,
    },
(20, 14): {(20, 13): 1},
(18, 10): {(17, 10): 1, (19, 10): 1},
(0, 1): {(0, 0): 1, (1, 1): 1, (0, 2): 1},
(3, 12): {
    (3, 11): 1,
    (4, 12): 1,
    (2, 12): 1,
    (3, 13): 1,
    },
(1, 12): {(1, 13): 1, (2, 12): 1, (0, 12): 1},
(8, 12): {
    (7, 12): 1,
    (8, 11): 1,
    (8, 13): 1,
    (9, 12): 1,
    },
(3, 1): {
    (3, 0): 1,
    (3, 2): 1,
    (4, 1): 2,
    (2, 1): 1,
    },
(2, 11): {(1, 11): 1, (3, 11): 1, (2, 10): 1},
(9, 9): {(8, 9): 1, (9, 8): 2, (10, 9): 1},
(5, 14): {(4, 14): 1, (6, 14): 1},
(10, 14): {(9, 14): 1, (11, 14): 1},
(6, 13): {
    (6, 12): 1,
    (7, 13): 1,
    (5, 13): 1,
    (6, 14): 1,
    },
(7, 8): {(8, 8): 1, (7, 7): 1, (7, 9): 1},
(14, 8): {(14, 9): 2, (14, 7): 1, (13, 8): 1},
(13, 0): {(13, 1): 2, (14, 0): 1, (12, 0): 1},
(12, 8): {(12, 7): 2, (12, 9): 1, (13, 8): 2},
(19, 8): {
    (18, 8): 1,
    (19, 9): 1,
    (20, 8): 1,
    (19, 7): 1,
    },
(18, 0): {(17, 0): 1, (18, 1): 1},
(17, 8): {(16, 8): 1, (17, 9): 1, (17, 7): 1},
(15, 13): {
    (15, 12): 1,
    (15, 14): 1,
    (16, 13): 1,
    (14, 13): 1,
    },
(13, 13): {(13, 14): 1, (14, 13): 2},
(20, 3): {(20, 2): 1, (19, 3): 1},
(0, 14): {(0, 13): 1, (1, 14): 2},
(3, 11): {
    (2, 11): 1,
    (3, 10): 1,
    (3, 12): 1,
    (4, 11): 1,
    },
(2, 1): {
    (2, 0): 1,
    (3, 1): 1,
    (1, 1): 1,
    (2, 2): 1,
    },
(8, 9): {
    (8, 8): 1,
    (8, 10): 1,
    (7, 9): 1,
    (9, 9): 1,
    },
(4, 12): {(4, 13): 1, (5, 12): 1, (3, 12): 1},
(2, 12): {(2, 13): 1, (3, 12): 1, (1, 12): 1},
(9, 4): {
    (9, 5): 1,
    (9, 3): 1,
    (10, 4): 2,
    (8, 4): 1,
    },
(7, 11): {
    (6, 11): 2,
    (7, 12): 1,
    (8, 11): 1,
    (7, 10): 2,
    },
(5, 1): {
    (4, 1): 1,
    (6, 1): 1,
    (5, 2): 1,
    (5, 0): 1,
    },
(10, 3): {
    (10, 2): 2,
    (9, 3): 1,
    (10, 4): 1,
    (11, 3): 1,
    },
(7, 2): {(6, 2): 1, (8, 2): 1},
(6, 14): {(5, 14): 1, (7, 14): 1, (6, 13): 1},
(12, 2): {
    (13, 2): 1,
    (11, 2): 1,
    (12, 1): 1,
    (12, 3): 1,
    },
(11, 10): {
    (12, 10): 1,
    (11, 9): 1,
    (11, 11): 1,
    (10, 10): 2,
    },
(17, 6): {(16, 6): 1, (17, 7): 2, (17, 5): 2},
(16, 10): {
    (17, 10): 2,
    (16, 9): 1,
    (16, 11): 2,
    (15, 10): 1,
    },
(14, 5): {
    (14, 4): 1,
    (15, 5): 1,
    (13, 5): 2,
    (14, 6): 1,
    },
(13, 3): {
    (14, 3): 2,
    (13, 4): 2,
    (13, 2): 1,
    (12, 3): 1,
    },
(12, 13): {(12, 12): 1, (12, 14): 1, (11, 13): 2},
(19, 7): {
    (19, 8): 1,
    (18, 7): 1,
    (20, 7): 1,
    (19, 6): 1,
    },
(18, 5): {
    (17, 5): 1,
    (18, 6): 1,
    (18, 4): 1,
    (19, 5): 1,
    },
(17, 11): {(16, 11): 1, (17, 12): 1},
(15, 0): {(16, 0): 1, (14, 0): 1},
(20, 0): {(19, 0): 2, (20, 1): 1},
(1, 5): {
    (0, 5): 2,
    (2, 5): 1,
    (1, 6): 1,
    (1, 4): 2,
    },
(0, 11): {(1, 11): 1, (0, 12): 1},
(3, 6): {(3, 7): 1, (4, 6): 1, (3, 5): 1},
(2, 2): {(1, 2): 2, (3, 2): 1, (2, 1): 1},
(1, 10): {(0, 10): 2, (1, 9): 1, (2, 10): 1},
(8, 6): {(7, 6): 2, (8, 5): 1, (8, 7): 2},
(4, 1): {
    (4, 0): 2,
    (5, 1): 1,
    (3, 1): 2,
    (4, 2): 1,
    },
(10, 9): {(10, 8): 1, (11, 9): 1, (9, 9): 1},
(9, 7): {
    (10, 7): 2,
    (9, 8): 1,
    (9, 6): 1,
    (8, 7): 1,
    },
(6, 4): {
    (6, 3): 1,
    (7, 4): 1,
    (5, 4): 1,
    (6, 5): 1,
    },
(5, 4): {
    (6, 4): 1,
    (4, 4): 1,
    (5, 5): 1,
    (5, 3): 1,
    },
(11, 4): {(10, 4): 1, (11, 3): 1},
(10, 4): {(11, 4): 1, (10, 3): 1, (9, 4): 2},
(16, 4): {(15, 4): 1, (17, 4): 2},
(6, 11): {(7, 11): 2, (6, 10): 1, (6, 12): 2},
(12, 7): {
    (11, 7): 1,
    (13, 7): 1,
    (12, 6): 1,
    (12, 8): 2,
    },
(11, 9): {
    (11, 8): 1,
    (11, 10): 1,
    (12, 9): 1,
    (10, 9): 1,
    },
(17, 1): {(17, 2): 1, (16, 1): 2, (18, 1): 1},
(15, 10): {
    (15, 11): 1,
    (14, 10): 1,
    (16, 10): 1,
    (15, 9): 1,
    },
(14, 6): {(15, 6): 1, (14, 7): 1, (14, 5): 1},
(13, 6): {(13, 7): 1, (12, 6): 1, (13, 5): 1},
(20, 10): {(19, 10): 1, (20, 11): 1},
(19, 2): {(20, 2): 1, (19, 1): 1, (18, 2): 1},
(18, 6): {(18, 7): 1, (18, 5): 1},
(0, 5): {(1, 5): 2, (0, 6): 1},
(15, 7): {
    (15, 8): 1,
    (15, 6): 1,
    (14, 7): 1,
    (16, 7): 1,
    },
(20, 5): {(20, 4): 1, (20, 6): 1},
(1, 0): {(2, 0): 1, (1, 1): 1},
(0, 8): {(0, 9): 1, (1, 8): 1, (0, 7): 1},
(4, 11): {(3, 11): 1, (5, 11): 1},
(3, 5): {(2, 5): 1, (3, 4): 1, (3, 6): 1},
(2, 7): {
    (3, 7): 1,
    (1, 7): 1,
    (2, 6): 2,
    (2, 8): 1,
    },
(9, 13): {(10, 13): 1, (9, 12): 1},
(8, 3): {
    (7, 3): 1,
    (9, 3): 1,
    (8, 2): 1,
    (8, 4): 1,
    },
(5, 10): {
    (5, 9): 1,
    (4, 10): 1,
    (6, 10): 1,
    (5, 11): 1,
    },
(4, 6): {(5, 6): 1, (4, 7): 1, (3, 6): 1},
(10, 10): {(9, 10): 1, (11, 10): 2, (10, 11): 1},
(9, 2): {
    (9, 1): 2,
    (9, 3): 1,
    (8, 2): 1,
    (10, 2): 2,
    },
(16, 13): {
    (17, 13): 2,
    (15, 13): 1,
    (16, 14): 2,
    (16, 12): 1,
    },
(6, 1): {(5, 1): 1, (6, 0): 1, (7, 1): 2},
(5, 7): {
    (5, 6): 1,
    (4, 7): 1,
    (6, 7): 1,
    (5, 8): 1,
    },
(11, 3): {
    (11, 4): 1,
    (11, 2): 1,
    (10, 3): 1,
    (12, 3): 1,
    },
(7, 4): {
    (6, 4): 1,
    (7, 5): 1,
    (7, 3): 1,
    (8, 4): 1,
    },
(14, 12): {(14, 11): 1, (13, 12): 1},
(12, 4): {(12, 5): 1, (13, 4): 2, (12, 3): 1},
(19, 12): {(20, 12): 2, (18, 12): 2, (19, 11): 1},
(17, 12): {
    (17, 13): 1,
    (17, 11): 1,
    (18, 12): 1,
    (16, 12): 1,
    },
(15, 9): {(16, 9): 1, (15, 8): 2, (15, 10): 1},
(14, 3): {(14, 4): 1, (14, 2): 2, (13, 3): 2},
(13, 9): {
    (13, 10): 1,
    (14, 9): 2,
    (12, 9): 1,
    (13, 8): 1,
    },
(19, 1): {
    (19, 0): 1,
    (19, 2): 1,
    (20, 1): 1,
    (18, 1): 1,
    },
(18, 11): {(19, 11): 2},
(0, 2): {(0, 1): 1, (1, 2): 2, (0, 3): 1},
(1, 3): {
    (1, 2): 1,
    (0, 3): 1,
    (2, 3): 2,
    (1, 4): 1,
    },
(8, 13): {(8, 14): 1, (7, 13): 2, (8, 12): 1},
(4, 8): {(3, 8): 1, (4, 7): 1, (4, 9): 2},
(3, 0): {(3, 1): 1, (4, 0): 1},
(2, 8): {(2, 7): 1, (2, 9): 1},
(9, 8): {(10, 8): 1, (9, 9): 2, (9, 7): 1},
(8, 0): {(8, 1): 1, (9, 0): 1, (7, 0): 1},
(5, 13): {(4, 13): 1, (5, 12): 1, (6, 13): 1},
(6, 2): {(6, 3): 1, (7, 2): 1},
(11, 14): {(10, 14): 1, (12, 14): 1, (11, 13): 1},
(16, 14): {(17, 14): 1, (15, 14): 1, (16, 13): 2},
(14, 9): {(13, 9): 2, (14, 10): 1, (14, 8): 2},
(12, 9): {
    (13, 9): 1,
    (12, 10): 1,
    (11, 9): 1,
    (12, 8): 1,
    },
(19, 11): {
    (19, 12): 1,
    (19, 10): 1,
    (18, 11): 2,
    (20, 11): 1,
    },
(18, 1): {
    (18, 0): 1,
    (19, 1): 1,
    (17, 1): 1,
    (18, 2): 1,
    },
(15, 12): {(15, 11): 1, (15, 13): 1},
(13, 12): {(14, 12): 1, (12, 12): 1},
(20, 12): {(20, 13): 1, (19, 12): 2, (20, 11): 1},
(18, 12): {(18, 13): 1, (19, 12): 2, (17, 12): 1},
(3, 10): {
    (4, 10): 2,
    (3, 9): 1,
    (3, 11): 1,
    (2, 10): 1,
    },
(1, 14): {(0, 14): 2, (1, 13): 1},
(8, 10): {(8, 9): 1, (8, 11): 1, (7, 10): 2},
(4, 13): {
    (4, 12): 1,
    (4, 14): 1,
    (3, 13): 2,
    (5, 13): 1,
    },
(2, 13): {
    (1, 13): 1,
    (2, 14): 1,
    (2, 12): 1,
    (3, 13): 1,
    },
(9, 11): {
    (9, 10): 1,
    (8, 11): 1,
    (10, 11): 1,
    (9, 12): 1,
    },
(5, 0): {(5, 1): 1, (4, 0): 1},
(16, 2): {(17, 2): 1, (16, 1): 1, (15, 2): 1},
(17, 7): {
    (18, 7): 1,
    (17, 8): 1,
    (17, 6): 2,
    (16, 7): 1,
    },
(10, 0): {(11, 0): 1, (9, 0): 1},
(7, 9): {(8, 9): 1, (6, 9): 2, (7, 8): 1},
(16, 9): {(16, 10): 1, (15, 9): 1},
(12, 3): {
    (12, 2): 1,
    (12, 4): 1,
    (11, 3): 1,
    (13, 3): 1,
    },
(11, 13): {
    (12, 13): 2,
    (10, 13): 1,
    (11, 12): 2,
    (11, 14): 1,
    },
(17, 5): {
    (17, 6): 2,
    (18, 5): 1,
    (16, 5): 2,
    (17, 4): 1,
    },
(7, 14): {(8, 14): 1, (7, 13): 2, (6, 14): 1},
(7, 6): {
    (8, 6): 2,
    (7, 5): 1,
    (7, 7): 1,
    (6, 6): 1,
    },
(13, 2): {(12, 2): 1, (14, 2): 1, (13, 3): 1},
(12, 14): {(13, 14): 1, (12, 13): 1, (11, 14): 1},
(19, 6): {(19, 7): 1, (20, 6): 1, (19, 5): 1},
(18, 2): {(17, 2): 1, (19, 2): 1, (18, 1): 1},
(17, 10): {(18, 10): 1, (17, 9): 1, (16, 10): 2},
(15, 3): {(15, 4): 1, (16, 3): 1, (15, 2): 2},
(7, 13): {(7, 14): 2, (8, 13): 2, (6, 13): 1},
(20, 1): {(20, 2): 1, (19, 1): 1, (20, 0): 1},
(1, 4): {
    (1, 5): 2,
    (1, 3): 1,
    (2, 4): 1,
    (0, 4): 1,
    },
(0, 12): {(0, 11): 1, (0, 13): 1, (1, 12): 1},
(3, 9): {
    (3, 8): 2,
    (3, 10): 1,
    (4, 9): 2,
    (2, 9): 1,
    },
(2, 3): {(1, 3): 2},
(1, 9): {(1, 10): 1, (1, 8): 1, (2, 9): 1},
(8, 7): {
    (8, 6): 2,
    (8, 8): 1,
    (7, 7): 1,
    (9, 7): 1,
    },
(4, 2): {(3, 2): 1, (4, 1): 1, (4, 3): 2},
(2, 14): {(2, 13): 1, (3, 14): 1},
(9, 6): {(9, 5): 1, (10, 6): 1, (9, 7): 1},
(6, 5): {(6, 4): 1, (7, 5): 2, (6, 6): 2},
(5, 3): {
    (6, 3): 1,
    (5, 4): 1,
    (5, 2): 1,
    (4, 3): 1,
    },
(11, 7): {
    (11, 8): 1,
    (10, 7): 2,
    (12, 7): 1,
    (11, 6): 1,
    },
(10, 5): {(9, 5): 1, (10, 6): 1, (11, 5): 1},
(16, 5): {(17, 5): 2, (15, 5): 1, (16, 6): 2},
(6, 8): {(6, 9): 1, (6, 7): 1, (5, 8): 1},
(12, 0): {(11, 0): 1, (12, 1): 1, (13, 0): 1},
(11, 8): {(10, 8): 1, (11, 9): 1, (11, 7): 1},
(17, 0): {(18, 0): 1, (16, 0): 1},
(16, 8): {(17, 8): 1, (15, 8): 1, (16, 7): 1},
(14, 7): {
    (14, 8): 1,
    (13, 7): 1,
    (15, 7): 1,
    (14, 6): 1,
    },
(13, 5): {(13, 6): 1, (12, 5): 1, (14, 5): 2},
(20, 11): {(20, 10): 1, (20, 12): 1, (19, 11): 1},
(19, 5): {(18, 5): 1, (19, 4): 1, (19, 6): 1},
(18, 7): {
    (18, 8): 2,
    (19, 7): 1,
    (18, 6): 1,
    (17, 7): 1,
    },
(0, 6): {(0, 5): 1, (1, 6): 1},
(15, 6): {(15, 5): 1, (15, 7): 1, (14, 6): 1},
(17, 3): {
    (17, 2): 1,
    (16, 3): 1,
    (18, 3): 1,
    (17, 4): 1,
    },
(20, 6): {(20, 5): 1, (20, 7): 1, (19, 6): 1},
(1, 7): {
    (2, 7): 1,
    (1, 8): 1,
    (1, 6): 1,
    (0, 7): 1,
    },
(0, 9): {(0, 8): 1, (0, 10): 1},
(3, 4): {(4, 4): 2, (2, 4): 1, (3, 5): 1},
(2, 4): {(3, 4): 1, (1, 4): 1},
(9, 12): {
    (9, 13): 1,
    (9, 11): 1,
    (10, 12): 1,
    (8, 12): 1,
    },
(8, 4): {
    (7, 4): 1,
    (8, 3): 1,
    (8, 5): 1,
    (9, 4): 1,
    },
(5, 9): {
    (4, 9): 1,
    (6, 9): 1,
    (5, 10): 1,
    (5, 8): 1,
    },
(4, 7): {
    (3, 7): 1,
    (5, 7): 1,
    (4, 6): 1,
    (4, 8): 1,
    },
(10, 11): {
    (10, 12): 2,
    (9, 11): 1,
    (11, 11): 1,
    (10, 10): 1,
    },
(9, 1): {
    (8, 1): 2,
    (9, 2): 2,
    (9, 0): 1,
    (10, 1): 1,
    },
(6, 6): {(7, 6): 1, (6, 7): 1, (6, 5): 2},
(5, 6): {(5, 7): 1, (4, 6): 1, (5, 5): 2},
(11, 2): {
    (12, 2): 1,
    (11, 1): 1,
    (11, 3): 1,
    (10, 2): 1,
    },
(10, 6): {
    (10, 7): 1,
    (10, 5): 1,
    (9, 6): 1,
    (11, 6): 1,
    },
(7, 7): {
    (6, 7): 1,
    (7, 6): 1,
    (7, 8): 1,
    (8, 7): 1,
    },
(14, 13): {(15, 13): 1, (13, 13): 2, (14, 14): 1},
(12, 5): {
    (12, 6): 1,
    (12, 4): 1,
    (13, 5): 1,
    (11, 5): 1,
    },
(14, 10): {
    (14, 11): 1,
    (14, 9): 1,
    (13, 10): 1,
    (15, 10): 1,
    },
(15, 8): {(16, 8): 1, (15, 7): 1, (15, 9): 2},
(14, 0): {(14, 1): 1, (15, 0): 1, (13, 0): 1},
(13, 8): {
    (13, 9): 1,
    (13, 7): 1,
    (14, 8): 1,
    (12, 8): 2,
    },
(20, 8): {(19, 8): 1, (20, 9): 1},
(19, 0): {(19, 1): 1, (20, 0): 2},
(18, 8): {(19, 8): 1, (18, 7): 2},
(0, 3): {(1, 3): 1, (0, 2): 1, (0, 4): 1},
(15, 5): {
    (15, 4): 1,
    (15, 6): 1,
    (16, 5): 1,
    (14, 5): 1,
    },
(3, 14): {(2, 14): 1, (4, 14): 1},
(1, 2): {(1, 3): 1, (0, 2): 2, (2, 2): 2},
(16, 1): {
    (16, 2): 1,
    (16, 0): 1,
    (17, 1): 2,
    (15, 1): 1,
    },
(4, 9): {(5, 9): 1, (3, 9): 2, (4, 8): 2},
(3, 3): {(3, 2): 1, (4, 3): 1},
(2, 9): {
    (2, 8): 1,
    (3, 9): 1,
    (1, 9): 1,
    (2, 10): 1,
    },
(8, 1): {(8, 0): 1, (9, 1): 2, (8, 2): 1},
(5, 12): {
    (6, 12): 2,
    (4, 12): 1,
    (5, 13): 1,
    (5, 11): 2,
    },
(4, 4): {
    (4, 5): 1,
    (5, 4): 1,
    (3, 4): 2,
    (4, 3): 2,
    },
(10, 12): {
    (11, 12): 2,
    (10, 13): 1,
    (10, 11): 2,
    (9, 12): 1,
    },
(6, 3): {
    (7, 3): 1,
    (5, 3): 1,
    (6, 2): 1,
    (6, 4): 1,
    },
(11, 1): {(11, 2): 1, (10, 1): 1},
(16, 7): {
    (16, 6): 1,
    (16, 8): 1,
    (15, 7): 1,
    (17, 7): 1,
    },
(7, 10): {(7, 11): 2, (8, 10): 2},
(14, 14): {(13, 14): 2, (14, 13): 1},
(3, 13): {(4, 13): 2, (2, 13): 1, (3, 12): 1},
(12, 10): {
    (13, 10): 1,
    (11, 10): 1,
    (12, 9): 1,
    (12, 11): 1,
    },
(19, 10): {
    (20, 10): 1,
    (19, 9): 2,
    (19, 11): 1,
    (18, 10): 1,
    },
(17, 14): {(16, 14): 1, (17, 13): 1},
(13, 11): {(14, 11): 1, (13, 10): 1, (12, 11): 1},
(20, 13): {(20, 12): 1, (20, 14): 1, (19, 13): 1},
(18, 13): {
    (17, 13): 1,
    (18, 14): 1,
    (18, 12): 1,
    (19, 13): 1,
    },
(0, 0): {(0, 1): 1},
(8, 14): {(8, 13): 1, (7, 14): 1, (9, 14): 1},
(1, 13): {
    (0, 13): 1,
    (2, 13): 1,
    (1, 14): 1,
    (1, 12): 1,
    },
(8, 11): {
    (7, 11): 1,
    (9, 11): 1,
    (8, 10): 1,
    (8, 12): 1,
    },
(16, 11): {
    (15, 11): 1,
    (17, 11): 1,
    (16, 10): 2,
    (16, 12): 2,
    },
(4, 14): {(4, 13): 1, (5, 14): 1, (3, 14): 1},
(7, 1): {(6, 1): 2, (7, 0): 1},
(2, 10): {
    (1, 10): 1,
    (3, 10): 1,
    (2, 11): 1,
    (2, 9): 1,
    },
(9, 10): {(9, 11): 1, (10, 10): 1},
(10, 1): {(11, 1): 1, (9, 1): 1, (10, 2): 1},
(6, 12): {
    (6, 11): 2,
    (7, 12): 1,
    (5, 12): 2,
    (6, 13): 1,
    },
(11, 12): {
    (11, 13): 2,
    (12, 12): 1,
    (11, 11): 1,
    (10, 12): 2,
    },
(17, 4): {
    (17, 5): 1,
    (17, 3): 1,
    (18, 4): 1,
    (16, 4): 2,
    },
(16, 12): {(16, 11): 2, (16, 13): 1, (17, 12): 1},
(14, 11): {
    (14, 12): 1,
    (15, 11): 1,
    (14, 10): 1,
    (13, 11): 1,
    },
(13, 1): {(14, 1): 1, (12, 1): 2, (13, 0): 2},
(7, 0): {(8, 0): 1, (6, 0): 2, (7, 1): 1},
(19, 9): {
    (19, 8): 1,
    (19, 10): 2,
    (20, 9): 2,
    (18, 9): 1,
    },
(18, 3): {(17, 3): 1, (19, 3): 1, (18, 4): 1},
(17, 9): {(17, 10): 1, (17, 8): 1, (18, 9): 1},
(15, 2): {
    (15, 3): 2,
    (14, 2): 2,
    (16, 2): 1,
    (15, 1): 1,
    },
(13, 14): {(14, 14): 2, (13, 13): 1, (12, 14): 1},
(20, 2): {(19, 2): 1, (20, 1): 1, (20, 3): 1},
(18, 14): {(18, 13): 1, (19, 14): 1},
(0, 13): {(0, 14): 1, (0, 12): 1, (1, 13): 1},
(3, 8): {(3, 7): 2, (3, 9): 2, (4, 8): 1},
(2, 0): {(1, 0): 1, (2, 1): 1},
(1, 8): {(0, 8): 1, (1, 9): 1, (1, 7): 1},
(16, 0): {(16, 1): 1, (17, 0): 1, (15, 0): 1},
(4, 3): {
    (4, 2): 2,
    (4, 4): 2,
    (3, 3): 1,
    (5, 3): 1,
    },
(9, 5): {
    (8, 5): 1,
    (10, 5): 1,
    (9, 6): 1,
    (9, 4): 1,
    },
(5, 2): {(5, 1): 1, (5, 3): 1},
(11, 6): {(11, 7): 1, (10, 6): 1, (11, 5): 2},
(10, 2): {
    (9, 2): 2,
    (11, 2): 1,
    (10, 3): 2,
    (10, 1): 1,
    },
}

default_maze = Maze(
    graph=maze_graph,
    width=21,
    height=15
)
