import random
from time import sleep

import os

import time
import keras
import numpy as np
from collections import deque

from keras import Input
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

from agents.agent import Agent
from agents.greedy_agent import GreedyAgent
from pyrat.components import State, Observation
from pyrat.game import PyratGame
from pyrat.definitions import add, position
from pyrat.default_maze import *

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

EPISODES = 100000
TIME_LIMIT = 200

NO_MOVEMENT_PENALTY = -2.0
CHEESE_REWARD = 1.0
WINNING_REWARD = 0

ENEMY_CHEESE_PENALTY = 0.0
ENEMY_WINNING_PENALTY = 0

TEST_EVERY = 1000
SAVE_EVERY = 1000


def base_from_div2_time(t):
    return 2 ** (-1.0 / t)


class DqnAgent:

    def __init__(self, maze: Maze, network_directory: str=None) -> None:

        #hyper params
        self.memory: deque = deque(maxlen=10000)
        self.gamma: float = base_from_div2_time(20)  # discount rate
        self.target_network_update_batchs = 100

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

        self.q_nework = self.build_model(Adam(), True)
        self.target_network = self.build_model()
        self.update_target_network()

        if network_directory is not None:
            self.load(network_directory)

    def prepare(self, state: State):
        pass

    def state_vector_representation(self, state: State):

        player_position = state.player_info.current_position

        player_x = player_position[0]
        player_y = player_position[1]

        opponent_position = state.opponent_info.current_position

        opponent_x = opponent_position[0]
        opponent_y = opponent_position[1]

        far_view_tensor: np.ndarray = np.zeros([self.far_view_size, self.far_view_size, 2])

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

        near_view_tensor: np.ndarray = np.zeros([self.near_view_size, self.near_view_size, 3])

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


        collision_vector: np.ndarray = np.zeros(4)

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


    def get_wall_mud_score(self, graph: MazeGraph, node: position) -> float:

        result = 1.0

        neighbors = graph.get(node, None)

        #Out of map
        if neighbors is None:
            return 0.0

        for i in range(4):

            distance = neighbors.get(add(node, direction_map[i]), None)

            if distance is None:
                result += -0.25 #Wall score
            elif distance > 1:
                result += -0.025 #Mud score

        return result

    def state_representation_to_string(self, state_obj: State) -> str:

        state = self.state_vector_representation(state_obj)

        near_view_tensor = state[0]
        far_view_tensor = state[1]
        collision_vector = state[2]
        last_positions_tensor = state[3]
        score_vector = state[4]

        result = "\nOpponent:\n "

        for y in range(near_view_tensor.shape[1] - 1, -1, -1):

            for x in range(near_view_tensor.shape[0]):
                result += "%.0f " % (near_view_tensor[x, y, 0])

            result += '\n '

        result += "\nCheese:\n "

        for y in range(near_view_tensor.shape[1] - 1, -1, -1):

            for x in range(near_view_tensor.shape[0]):
                result += "%.0f " % (near_view_tensor[x, y, 1])

            result += '\n '

        result += "\nOpponent:\n "

        for y in range(far_view_tensor.shape[1] - 1, -1, -1):

            for x in range(far_view_tensor.shape[0]):
                result += "%.0f " % (far_view_tensor[x, y, 0])

            result += '\n '

        result += "\nCheese:\n "

        for y in range(far_view_tensor.shape[1] - 1, -1, -1):

            for x in range(far_view_tensor.shape[0]):
                result += "%.0f " % (far_view_tensor[x, y, 1])

            result += '\n '

        result += "\nWall Mud Score:\n "

        for y in range(far_view_tensor.shape[1] - 1, -1, -1):

            for x in range(far_view_tensor.shape[0]):
                result += "%.2f\t" % far_view_tensor[x, y, 2]

            result += '\n '

        result += "\nCollision:\n"

        for i in range(4):
            result += action_map[i] + "\t"

        result += "\n"

        for i in range(4):
            result += "%.2f\t" % (collision_vector[i])

        result += "\nLast Positions:\n"

        result += "t:\t"

        for i in range(self.positions_saved):
            result += "%d\t" % i

        result += "\nx:\t"

        for i in range(self.positions_saved):
            result += "%d\t" % last_positions_tensor[i, 0]

        result += "\ny:\t"

        for i in range(self.positions_saved):
            result += "%d\t" % last_positions_tensor[i, 1]

        result += "\n"

        result +="\n Score Player: %.1f Opponent: %.1f\n" % (score_vector[0], score_vector[1])

        return result

    def action_vector_representation(self, action: str) -> np.ndarray:
        result = np.zeros([4])
        result[action_map[action]] = 1
        return result


    def build_model(self, optimizer=None, log: bool=False) -> Model:

        #Atari
        # model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
        #                         input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        # model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(self.num_actions))

        #RL
        # super(ConvGridNet, self).__init__()
        # self.conv1 = nn.Conv2d(6, 32, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.conv3_drop = nn.Dropout2d()
        # self.fc_player = nn.Linear(58, 20)
        # self.fc1 = nn.Linear(3860, 512)
        #
        #
        # c1 = F.relu(F.max_pool2d(self.conv1(input_grid), 2))
        # c2 = F.relu(F.max_pool2d(self.conv2(c1), 2))
        # x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(c2)), 2))
        # x = x.view(input_grid.size(0), -1)
        # y = self.fc_player(input_player.view(input_grid.size(0), -1))
        # y = F.relu(y)
        # combined = torch.cat((x, y), 1)
        # return F.relu(self.fc1(combined))

        # input1 = Input(shape=(2 * mazeHeight - 1, 2 * mazeWidth - 1, 10))
        # input2 = Input(shape=(3, 3, 10))
        # conv1 = Conv2D(32, (2, 2), padding="same")(input1)
        # relu1 = Activation("relu")(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(relu1)
        # conv2 = Conv2D(64, (2, 2), padding="same")(pool1)
        # relu2 = Activation("relu")(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(relu2)
        # conv3 = Conv2D(128, (2, 2), padding="same")(pool2)
        # relu3 = Activation("relu")(conv3)
        # pool3 = GlobalAveragePooling2D()(relu3)
        # flattened_2 = Flatten()(input2)
        # concat = Concatenate()([pool3, flattened_2])
        # dense2 = Dense(4)(concat)
        # softmax = Activation("softmax")(dense2)

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

        if optimizer is not None:
            model.compile(loss='mse', optimizer=optimizer)

        if(log):
            print(model.summary())

        return model

    def update_target_network(self) -> None:
        weights = self.q_nework.get_weights()
        self.target_network.set_weights(weights)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append(
            (self.state_vector_representation(state),
             action_map[action],
             reward,
             self.state_vector_representation(next_state),
             done)
        )

    def act(self, state, exploration_rate=0.0, print_qs=False):

        if np.random.rand() < exploration_rate:
            return action_map[random.randrange(self.action_size)]

        vector_state = self.stack_state([self.state_vector_representation(state)])

        act_values = self.target_network.predict(vector_state)

        if print_qs:

            result = ""

            for i in range(4):
                result += action_map[i] + "\t"

            result += "\n"

            for i in range(4):
                result += "%.2f\t" % (act_values[0, i])

            print(result)

        return action_map[np.argmax(act_values[0])]

    def stack_state(self, state_batch):
        result = []

        for i in range(len(state_batch[0])):
            result.append(
                np.stack([state[i] for state in state_batch], axis=0)
            )

        return result

    def replay(self, batch_size: int) -> float:

        minibatch = random.sample(self.memory, batch_size)

        state_batch = self.stack_state([mem[0] for mem in minibatch])
        action_batch = np.stack([mem[1] for mem in minibatch], axis=0)
        reward_batch = np.stack([mem[2] for mem in minibatch], axis=0)
        next_state_batch = self.stack_state([mem[3] for mem in minibatch])

        target = reward_batch + self.gamma * np.amax(self.target_network.predict(next_state_batch), axis=1)

        target_vector = self.q_nework.predict(state_batch)

        for i in range(batch_size):
            target_vector[i, action_batch[i]] = target[i]

        history = self.q_nework.fit(x=state_batch, y=target_vector, epochs=1, verbose=0)

        loss = history.history['loss'][-1]

        if self.batchs_trained % self.target_network_update_batchs == 0:
            self.update_target_network()

        self.batchs_trained += 1

        return loss

    def load(self, name: str) -> None:
        self.q_nework.load_weights("{}.q_network.h5".format(name))
        self.target_network.load_weights("{}.target_network.h5".format(name))

    def save(self, name: str) -> None:
        self.q_nework.save_weights("{}.q_network.h5".format(name))
        self.target_network.save_weights("{}.target_network.h5".format(name))


def compute_reward(state1: State, state2: State, action):

    result = (state2.player_info.score - state1.player_info.score) * CHEESE_REWARD

    result += (state2.opponent_info.score - state1.opponent_info.score) * ENEMY_CHEESE_PENALTY

    nodes = state1.maze.graph[state1.player_info.current_position]

    if add(state1.player_info.current_position, direction_map[action]) not in nodes:
        result += NO_MOVEMENT_PENALTY

    if state1.last_positions[0] == state1.last_positions[2] and \
            state1.last_positions[1] == state1.last_positions[3]:
        result += NO_MOVEMENT_PENALTY

    if state2.finished:
        result += WINNING_REWARD if state2.player_info.is_winner else 0
        result -= ENEMY_CHEESE_PENALTY if state2.opponent_info.is_winner else 0

    result += cheese_distance_score(state2) - cheese_distance_score(state1)

    return result


def cheese_distance_score(state: State):

    result = 0

    for cheese in state.cheese_list:
        delta_x = cheese[0] - state.player_info.current_position[0]
        delta_y = cheese[1] - state.player_info.current_position[1]

        distance = abs(delta_x) + abs(delta_y)

        if distance != 0:
            result += 1.0/distance

    if len(state.cheese_list) != 0:
        result /= len(state.cheese_list)

    return result * 1.0


def average(collection):

    n = len(collection)

    if n == 0:
        return 0.0

    sum = 0.0

    for x in collection:
        sum += x

    return sum / n


def run_match(
        env: PyratGame,
        player1: DqnAgent,
        player2: GreedyAgent,
        exploration_rate: float=0.0,
        train_player1: bool=False,
        time_limit: int=200,
        log: bool=False
) -> Observation:

    observation = env.initialize(cheese_count=40)

    state1 = State(observation, 1)
    state2 = State(observation, 2)

    player1.prepare(state1)
    player2.prepare(state2)

    if log:
        print(observation)

    for i in range(time_limit):

        action1 = player1.act(state1, exploration_rate=exploration_rate, print_qs=log)
        action2 = player2.act(state2, exploration_rate=exploration_rate)

        if log:
            sleep(0.5)

        observation = env.step(action1, action2)

        done = observation.finished

        next_state1 = State(observation, 1, state1)
        next_state2 = State(observation, 2, state2)

        reward1 = compute_reward(state1, next_state1, action1)
        reward2 = compute_reward(state2, next_state2, action2)

        if train_player1:
            player1.remember(state1, action1, reward1, next_state1, done)
            player1.remember(state2, action2, reward2, next_state2, done)

        if log:
            print(observation)
            print("Reward 1: %f  \t  Reward 2: %f" % (reward1, reward2))

        state1 = next_state1
        state2 = next_state2

        # if np.random.rand() < 1.0/10000:
        #     print(observation)
        #     print(player1.state_representation_to_string(next_state1))

        if done:
            break

    return observation


def eta_to_text(eta: float) -> str:

    if eta < 60:
        eta_text = "{:.2f} s".format(eta)
    else:
        eta /= 60

        if eta < 60:
            eta_text = "{:.2f} m".format(eta)
        else:
            eta /= 60

            if eta < 24:
                eta_text = "{:.2f} h".format(eta)
            else:
                eta /= 24

                eta_text = "{:.2f} d".format(eta)


    return eta_text


if __name__ == "__main__":

    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = base_from_div2_time(2000)

    player1 = DqnAgent(default_maze)
    player2 = GreedyAgent()

    env = PyratGame(maze=default_maze)

    env.initialize(cheese_count=40)

    done = False
    batch_size = 128

    loss_history = deque(maxlen=20)

    start_time = time.time()

    eta_text = ""

    for e in range(EPISODES):

        if e % 9 == 0:

            now = time.time()

            eta_text = eta_to_text((EPISODES - (e + 1)) * (now - start_time) / 10.0)

            start_time = now


        last_observation = run_match(
            env=env,
            player1=player1,
            player2=player2,
            exploration_rate=epsilon,
            train_player1=True,
            time_limit=TIME_LIMIT,
            log=False
        )

        if len(player1.memory) > batch_size:
            loss_history.append(player1.replay(batch_size))


        print("episode: {}/{},\t e: {:.2},\t Player1: {},\t Player2: {},\t time steps: {},\t loss avg: {} \t ETA: {}"
            .format(
            e,
            EPISODES,
            epsilon,
            last_observation.player1_info.score,
            last_observation.player2_info.score,
            last_observation.time_steps,
            average(loss_history),
            eta_text
        )
        )

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if e % TEST_EVERY == 0 and e >= 1000:

            last_observation = run_match(
                env=env,
                player1=player1,
                player2=player2,
                exploration_rate=0.0,
                train_player1=False,
                time_limit=200,
                log=True
            )

            print("Test: Player1: {},\t Player2: {},\t time steps: {}"
                .format(
                last_observation.player1_info.score,
                last_observation.player2_info.score,
                last_observation.time_steps
            )
            )

        if e % SAVE_EVERY == 0  and e > 0:

            directory = "./dqn/episode-{0:d}/".format(e)

            if not os.path.exists(directory):
                os.makedirs(directory)

            player1.save("{0}dqn_agent-{1:d}".format(directory, e))

