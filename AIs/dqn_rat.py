import random

import keras
import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten

TEAM_NAME = "DQN Rat"

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

action_map = {
    'R':0,
    'L':1,
    'U':2,
    'D':3,
    0:'R',
    1:'L',
    2:'U',
    3:'D'
}

direction_map = {
    'R': (1, 0),
    'L': (-1, 0),
    'U': (0, 1),
    'D': (0, -1),
    0: (1, 0),
    1: (-1, 0),
    2: (0, 1),
    3: (0, -1)
}


class PlayerInfo:

    def __init__(self, position, score):
        self.position = position
        self.score = score


class State:

    def __init__(self, maze, player_info, opponent_info, piecesOfCheese, previous_state=None):
        self.maze = maze
        self.cheese_list = piecesOfCheese

        self.positions_saved = 4

        self.player_info = player_info
        self.opponent_info = opponent_info

        self.last_positions = [self.player_info.position for i in range(self.positions_saved)]

        if previous_state is not None:

            for i in range(1, self.positions_saved):
                self.last_positions[i] = previous_state.last_positions[i-1]


class Maze:

    def __init__(self, graph, width, height):

        self.height = height
        self.width = width
        self.graph = graph

    def random_position(self):
        return (random.randrange(0, self.width), random.randrange(0, self.height))

    def area(self):
        return self.width * self.height


def add(coordinates1, coordinates2):
    return (coordinates1[0] + coordinates2[0], coordinates1[1] + coordinates2[1])


def base_from_div2_time(t):
    return 2 ** (-1.0 / t)


class DqnAgent:

    def __init__(self, maze, network_directory=None):

        #hyper params
        self.gamma = base_from_div2_time(20)  # discount rate

        self.positions_saved = 4

        self.near_view_radius = 5
        self.near_view_size = 2 * self.near_view_radius + 1
        self.far_view_radius = 16
        self.far_view_size = 2 * self.far_view_radius + 1

        self.action_size = 4
        self.batchs_trained = 0

        self.map_node_value = np.zeros([maze.width, maze.height])
        self.map_width = maze.width
        self.map_height = maze.height

        for x in range(maze.width):
            for y in range(maze.height):
                self.map_node_value[x, y] = self.get_wall_mud_score(maze.graph, (x, y))

        self.target_network = self.build_model()

        if network_directory is not None:
            self.load(network_directory)

    def prepare(self, state):
        pass

    def state_vector_representation(self, state):

        player_position = state.player_info.position

        player_x = player_position[0]
        player_y = player_position[1]

        opponent_position = state.opponent_info.position

        opponent_x = opponent_position[0]
        opponent_y = opponent_position[1]

        far_view_tensor = np.zeros([self.far_view_size, self.far_view_size, 2])

        def visible(radius, x, y):
            return player_x - radius <= x <= player_x + radius and \
                player_y - radius <= y <= player_y + radius

        def view_x(view_radius, absolute_x):
            return view_radius+absolute_x-player_x

        def view_y(view_radius, absolute_y):
            return view_radius+absolute_y-player_y

        if visible(self.far_view_radius, opponent_x, opponent_y):
            far_view_tensor[
                view_x(self.far_view_radius, opponent_x),
                view_y(self.far_view_radius, opponent_y),
                0
            ] = 1.0

        for cheese in state.cheese_list:
            if visible(self.far_view_radius, cheese[0], cheese[1]):
                far_view_tensor[
                    view_x(self.far_view_radius, cheese[0]),
                    view_y(self.far_view_radius, cheese[1]),
                    1
                ] = 1.0

        near_view_tensor = np.zeros([self.near_view_size, self.near_view_size, 3])

        if visible(self.near_view_radius, opponent_x, opponent_y):
            near_view_tensor[
                view_x(self.near_view_radius, opponent_x),
                view_y(self.near_view_radius, opponent_y),
                0
            ] = 1.0

        for cheese in state.cheese_list:
            if visible(self.near_view_radius, cheese[0], cheese[1]):
                near_view_tensor[
                    view_x(self.near_view_radius, cheese[0]),
                    view_y(self.near_view_radius, cheese[1]),
                    1
                ] = 1.0


        for x in range(self.near_view_size):
            for y in range(self.near_view_size):

                abs_x = player_x + x - self.near_view_radius
                abs_y = player_y + y - self.near_view_radius

                if 0 <= abs_x < self.map_width and 0 <= abs_y < self.map_height:
                    near_view_tensor[x, y, 2] = self.map_node_value[abs_x, abs_y]


        collision_vector = np.zeros(4)

        nodes = state.maze.graph.get(player_position)

        for i in range(4):
            distance = nodes.get(add(player_position, direction_map[i]), None)

            if distance is None:
                collision_vector[i] = 1.0
            elif distance > 1:
                collision_vector[i] = 0.1

        last_positions_tensor = np.array([
            [state.last_positions[i][0],
             state.last_positions[i][1]]
            for i in range(self.positions_saved)
        ])

        score_array = np.array([
            state.player_info.score,
            state.opponent_info.score
        ])

        return (far_view_tensor, near_view_tensor, collision_vector, last_positions_tensor, score_array)


    def get_wall_mud_score(self, graph, node):

        result = 1

        neighbors = graph.get(node, None)

        #Out of map
        if neighbors is None:
            return 0

        for i in range(4):

            distance = neighbors.get(add(node, direction_map[i]), None)

            if distance is None:
                result += -0.25 #Wall score
            elif distance > 1:
                result += -0.025 #Mud score

        return result


    def action_vector_representation(self, action):
        result = np.zeros([4])
        result[action_map[action]] = 1
        return result


    def build_model(self, log=False):

        map_tensor = Input(shape=[self.far_view_size, self.far_view_size, 2])
        view_tensor = Input(shape=[self.near_view_size, self.near_view_size, 3])
        collision_vector = Input(shape=[4])
        last_positions_tensor = Input(shape=[self.positions_saved, 2])
        score_vector = Input(shape=[2])

        map_conv1 = Conv2D(
            filters=32,
            kernel_size=(7, 7),
            padding="same",
            activation="relu"
        )(map_tensor)

        map_pool2 = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same"
        )(map_conv1)

        map_conv3 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(map_pool2)

        map_pool3 = keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            padding="same"
        )(map_conv3)

        flatten_map = Flatten()(map_pool3)

        flatten_view = Flatten()(view_tensor)

        flatten_positions = Flatten()(last_positions_tensor)

        merged_vector = keras.layers.concatenate(
            [flatten_map, flatten_view, collision_vector, flatten_positions, score_vector],
            axis=-1
        )

        dense1 = Dense(
            units=128,
            activation="relu"
        )(merged_vector)

        # dense1_dropout = Dropout(
        #     rate=0.5
        # )(dense1)

        dense3 = Dense(
            units=4,
            activation="linear"
        )(dense1)

        model = Model(
            inputs=[map_tensor, view_tensor, collision_vector, last_positions_tensor, score_vector],
            outputs=dense3
        )

        if(log):
            print(model.summary())

        return model


    def act(self, state):

        vector_state = self.stack_state([self.state_vector_representation(state)])

        act_values = self.target_network.predict(vector_state)

        return action_map[np.argmax(act_values[0])]


    def stack_state(self, state_batch):
        result = []

        for i in range(len(state_batch[0])):
            result.append(
                np.stack([state[i] for state in state_batch], axis=0)
            )

        return result

    def load(self, name):
        self.target_network.load_weights("%s.target_network.h5" % name)


maze = None
agent = None

previous_state = None


def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):

    global maze, agent, previous_state

    maze = Maze(mazeMap, mazeWidth, mazeHeight)

    agent = DqnAgent(maze, "dqn_agent")

    player_info = PlayerInfo(playerLocation, 0)
    opponent_info = PlayerInfo(opponentLocation, 0)

    state = State(maze, player_info, opponent_info, piecesOfCheese)

    agent.prepare(state)

    previous_state = state




def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):

    global maze, agent, previous_state

    player_info = PlayerInfo(playerLocation, playerScore)
    opponent_info = PlayerInfo(opponentLocation, opponentScore)

    state = State(maze, player_info, opponent_info, piecesOfCheese, previous_state)

    action = agent.act(state)

    #Check timesteps

    previous_state = state

    return action
