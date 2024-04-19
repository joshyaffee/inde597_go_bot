from environment1 import GoEnv

import os
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras
import tqdm
import time
import random

class ConvAgentA:
    def __init__(self, env = None, policy_path = None, name = None):

        self.env = env
        self.board_size = env.size
        # enumerate all actions
        self.action_space = env.get_actions()
        if policy_path is None:
            self.nn = self.build_network()
            self.prev_episodes = 0
        else:
            self.load(policy_path)
            self.prev_episodes = int(policy_path.split("_")[-1].split(".")[0])
        self.name = time.time() if name is None else name

    def save(self, path):
        Model.save(self.nn, path)

    def load(self, path):
        self.nn = keras.models.load_model(path)

    def _predict(self, states):
        # reshape states to fit the input shape of the network
        states = np.array(states).reshape(-1, self.board_size, self.board_size, 1)
        return self.nn.predict(states, verbose=0)
    
    def _fit(self, states, targets):
        # reshape states to fit the input shape of the network
        states = np.array(states).reshape(-1, self.board_size, self.board_size, 1)
        self.nn.fit(states, targets, verbose=0)
    
    def train(self, num_episodes=1001, batch_size=32, gamma=0.9, epsilon=0.9, epsilon_decay = 0.95, epsilon_min = 0.1):
        replay_memory = []
        for episode in tqdm.tqdm(range(self.prev_episodes, self.prev_episodes + num_episodes)):
            _board, _turn = env.reset()
            board = _board.copy()
            turn = _turn
            done = False
            total_reward = 0

            while not done:
                action = self.act(board, turn, epsilon=epsilon)
                (next_board, next_turn), reward, done, extra = env.step(action) # does this mutate the board???
                # add small reward for captures
                reward += 0.01 * extra['num_captured']

                replay_memory.append((board, turn, self.action_space.index(action), reward, next_board, next_turn, done))
                if len(replay_memory) > 100000:
                    replay_memory.pop(0)
                total_reward += reward
                board = next_board.copy()
                turn = next_turn

                if len(replay_memory) >= batch_size:
                    minibatch = random.sample(replay_memory, batch_size)
                    boards, turns, actions, rewards, next_boards, next_turns, dones = map(np.array, zip(*minibatch))
                    # multiply all values in the i'th board by the i'th turn
                    perspective = []
                    for i in range(len(boards)):
                        perspective.append(boards[i] * turns[i])
                    boards = np.array(perspective)

                    next_perspective = []
                    for i in range(len(next_boards)):
                        next_perspective.append(next_boards[i] * next_turns[i])
                    next_boards = np.array(next_perspective)
                    
                    q_values_next = np.max(self._predict(next_boards * -1), axis=1)
                    targets = rewards - gamma * q_values_next * (1 - dones)
                    current_q_values = self._predict(boards)
                    for i, action in enumerate(actions):
                        current_q_values[i][action] = targets[i]
                    self._fit(boards, current_q_values)

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # print(f"\nEpisode {episode + 1}, Total reward: {total_reward}, Epsilon: {epsilon:.4f}")
            if episode % 10 == 0:  # Save every 10 episodes
                self.env.board.print_history()
                self.save(f'nn_policies\\size5\\{self.name}_{episode}.keras')
    
    def act(self, board, turn, epsilon=0.1):
        legal_moves = self.env.get_legal_moves()
        if np.random.rand() <= epsilon:
            return random.choice(legal_moves)
        else:
            pred = self._predict(board * turn) # I think this is right.
            # get top move that is legal
            for i in np.argsort(pred)[::-1][0]:
                if self.action_space[i] in legal_moves:
                    return self.action_space[i]
        raise Exception("No legal moves found. That's wrong.")
        
    def build_network(self):
        input_shape = (self.board_size, self.board_size, 1)
        # Define input layers for the board matrix and turn indicator
        board_input = Input(shape= input_shape, name='board_input')
        # turn_input = Input(shape=(1,), name='turn_input')

        # Convolutional layers
        conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(board_input)
        # add dropout layer
        dropout_layer1 = tf.keras.layers.Dropout(0.15)(conv_layer1)
        conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(dropout_layer1)
        # add dropout layer
        dropout_layer2 = tf.keras.layers.Dropout(0.15)(conv_layer2)
        max_pooling_layer = MaxPooling2D(pool_size=(2, 2))(dropout_layer2)
        flat_layer = Flatten()(max_pooling_layer)
        flat_board = Flatten()(board_input)

        # Concatenate flattened board and turn indicator
        concatenated_layers = Concatenate()([flat_layer, flat_board])

        # Fully connected layers
        hidden_layer1 = Dense(256, activation='relu')(concatenated_layers)


        hidden_layer2 = Dense(256, activation='relu')(hidden_layer1)

        # Output layer. one output for each action
        output_layer = Dense(len(self.action_space), activation='sigmoid')(hidden_layer2)

        # Define model
        model = Model(inputs=[board_input], outputs=output_layer)

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        return model
    
    def play_human(self, color = 1):
        self.env.reset()
        done = False
        self.env.board.print_board()
        while not done:
            if self.env.board.turn == color:
                action = self.act(self.env.board.matrix_board, self.env.board.turn, epsilon=0)
                next_state, reward, done, _ = self.env.step(action)
                print(action)
                self.env.board.print_board()
            else:
                action = input("Enter move: ")
                if action.upper() == "RESIGN":
                    print("You resigned.")
                    return None
                try:
                    next_state, reward, done, _ = self.env.step(action)
                except:
                    print("Illegal move")
                    continue
                self.env.board.print_board()
        # the turn will switch to the next player after the game ends
        # the reward corresponds to the player who just passed
        # So, reward * self.env.turn will be 1 if white won, -1 if black won, and 0 if it
        # was a draw.
        # so if color * reward * self.env.turn is 1, the human won. If it is -1, the human
        # lost. If it is 0, it was a draw.
        x = color * reward * self.env.board.turn
        if x == 1:
            print("You won!")
        elif x == -1:
            print("You lost!")
        else:
            print("It was a draw.")
        return None

# test nn agent
env = GoEnv(size = 5)
agent = ConvAgentA(env, name = "conv5b", policy_path=r"nn_policies\\size5\\conv5b_810.keras")
try:
    agent.train(1, batch_size=32, gamma=0.995)
except(KeyboardInterrupt):
    print("Training interrupted.")
# test human play
agent.play_human(color = 1)
print("\n"*5)
agent.play_human(color = -1)