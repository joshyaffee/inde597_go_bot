import environment1 as GoEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras
import tqdm
import time
import random

class ConvAgentA:
    def __init__(self, env: GoEnv.GoEnv = None, policy_path = None, name = None):

        self.env = env
        self.board_size = env.size
        if policy_path is None:
            self.nn = self.build_network()
        else:
            self.load(policy_path)
        self.name = time.time() if name is None else name

    def save(self, path):
        Model.save(self.nn, path)

    def load(self, path):
        self.nn = keras.models.load_model(path)

    def sim_all_moves(self, rep, turn, capture_prop=0.01):
        """
        Simulate all possible moves including pass and return {move: q_predict} dictionary.
        """
        board = GoEnv.GoBoard(size = self.board_size)
        board.reset(board = rep, turn = turn)

        legal_moves = board.get_legal_moves()
        q = {lm: 0 for lm in legal_moves}
        r = {lm: 0 for lm in legal_moves}

        batch_states = []
        batch_moves = []

        for move in legal_moves:
            # simulate move
            board, ((matrix, turn), reward, done, extra) = self.env.look_ahead(move, board, representation=1)

            # add reward for captures - this is a heuristic
            reward += capture_prop * extra['num_captured']
            r[move] = reward

            # if game is draw, set slightly negative reward - heuristic to discourage peaceful games
            if done and reward == 0:
                r[move] = -0.1

            batch_states.append(-1 * matrix)
            batch_moves.append(turn)
            
            self.env.pop(board)

        if batch_states:
            batch_states = np.array(batch_states)
            batch_moves = np.array(batch_moves)
            q_values = self._predict(batch_states, batch_moves)
            for move, q_value in zip(legal_moves, q_values):
                q[move] = r[move] - q_value

        return q

    def _predict(self, states, turns):
        return self.nn.predict([states, turns], verbose=0)
    
    def _fit(self, states, turns, targets):
        self.nn.fit([states, turns], targets, verbose=0)
    
    def train(self, N, batch_size=32, gamma=0.9, epsilon=0.1, greedy="softmax", capture_prop=0.01):
        self.replay_buffer = []
        try:
            for episode in tqdm.tqdm(range(N), desc=f'Working on {N} episodes of training'):
                # Initialize the state
                state = self.env.reset()
                done = False
                while not done:
                    # Choose action
                    action = self.act(epsilon=epsilon, greedy="softmax")
                    # Take action
                    state, reward, done, extra = self.env.step(action)
                    # print board
                    if episode % 10 == 0:
                        self.env.board.print_board()

                    # Add reward for captures - this is a heuristic
                    reward += capture_prop * extra['num_captured']

                    # if game is draw, set slightly negative reward - heuristic to
                    # discourage peaceful games
                    if done and reward == 0:
                        reward = -0.1

                    # If the game is over, set the target to the reward
                    if done:
                        max_q = 0
                    else:
                        # get target from greedy policy
                        q = self.sim_all_moves(rep = self.env.board.matrix_board, turn = self.env.board.turn)
                        max_q = max(q.values())
                    target = reward + gamma * max_q

                    # Add this state to the replay buffer
                    self.replay_buffer.append((state, target))

                    # Train the network using experience replay
                    if len(self.replay_buffer) >= batch_size:
                        minibatch = random.sample(self.replay_buffer, batch_size)
                        images = np.array([sample[0][0] for sample in minibatch])  # Extracting images
                        others = np.array([sample[0][1] for sample in minibatch])  # Extracting others
                        # Extract targets making sure dimensions are correct
                        targets =[]
                        for sample in minibatch:
                            if type(sample[1]) == np.ndarray:
                                targets.append(sample[1])
                            else:
                                targets.append(np.array([sample[1]]))
                        targets = np.array(targets).reshape(-1, 1)
                        self._fit(images, others, targets)
        except KeyboardInterrupt:
            print("Training interrupted")
            if episode < 10:
                print("Training did not complete 10 episodes. Nothing will be saved.")
                return None
        self.save(f"nn_policies/size{self.board_size}/{self.name}.keras")
        return self.nn
    
    def act(self, epsilon=0.1, greedy="softmax"):
        if random.random() < epsilon:
            legal_moves = self.env.get_legal_moves()
            return random.choice(legal_moves)
        else:
            q = self.sim_all_moves(rep = self.env.board.matrix_board, turn = self.env.board.turn)
            if greedy == "softmax":
                # softmax
                q_values = np.array(list(q.values()))
                q_values = np.exp(q_values / epsilon)
                q_values = q_values / np.sum(q_values)
                move = random.choices(list(q.keys()), weights=q_values)
                if type(move) == list:
                    move = move[0]
            elif greedy == "greedy":
                # epsilon greedy
                move = max(q, key=q.get)
            return move

    def build_network(self):
        input_shape = (self.board_size, self.board_size, 1)
        # Define input layers for the board matrix and turn indicator
        board_input = Input(shape= input_shape, name='board_input')
        turn_input = Input(shape=(1,), name='turn_input')

        # Convolutional layers
        conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(board_input)
        # add dropout layer
        dropout_layer1 = tf.keras.layers.Dropout(0.15)(conv_layer1)
        conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(dropout_layer1)
        # add dropout layer
        dropout_layer2 = tf.keras.layers.Dropout(0.15)(conv_layer2)
        max_pooling_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer2)
        flat_layer = Flatten()(max_pooling_layer)
        flat_board = Flatten()(board_input)

        # Concatenate flattened board and turn indicator
        concatenated_layers = Concatenate()([flat_layer, flat_board, turn_input])

        # Fully connected layers
        hidden_layer1 = Dense(256, activation='relu')(concatenated_layers)
        hidden_layer2 = Dense(256, activation='relu')(hidden_layer1)

        # Output layer
        output_layer = Dense(1, activation='tanh')(hidden_layer2)

        # Define model
        model = Model(inputs=[board_input, turn_input], outputs=output_layer)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def play_human(self, color = 1):
        self.env.reset()
        done = False
        self.env.board.print_board()
        while not done:
            if self.env.board.turn == color:
                action = self.act(epsilon=0, greedy="greedy")
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
env = GoEnv.GoEnv(size = 5)
agent = ConvAgentA(env, name = "conv3", policy_path=r"nn_policies\size5\conv3.keras")
agent.train(200, batch_size=32, gamma=0.9, epsilon=0.1, greedy="softmax", capture_prop=0.007)

# test human play
agent.play_human(color = 1)
print("\n"*5)
agent.play_human(color = -1)