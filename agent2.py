"""
Agent super class for Go game
subclasses will implement the policy function
"""

import environment1 as env1
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate, LSTM, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import random
import tqdm
import time

class Agent:
    def __init__(self, color, policy = None, env: env1.GoEnv = None):
        self.current_color = color
        self.env = env 
        self.board_size = env.size
        if policy is None:
            policy = self._init_policy()
        self.policy = policy
        if abs(color) != 1:
            raise ValueError("color must be 1 or -1")
        
    
    def act(self):
        # get legal moves
        legal_moves = self.board.get_legal_moves(self.color)

        # use policy to select a move
        move = self.policy(self.board, self.color)

        # check that the move is legal
        if move not in legal_moves:
            # warn the user
            print("Illegal move chosen: " + str(move))
            return "resign"
        
        # make the move
        return move
    
    def _init_policy(self):
        pass # to be implemented by subclasses
    
    def _get_legal_moves(self, board):
        return board.get_legal_moves(self.color)
    
class RandomAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        return np.random.choice(legal_moves)
    
class CaptureFirstAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        for move in legal_moves:
            x, y = self.board._coord_to_pos(move)
            if self.board.check_captures(x, y, self.color, check_only=True):
                return move
            
class RandomDiagonalAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        diagonal_moves = []
        for move in legal_moves:
            x, y = self.board._coord_to_pos(move)
            if x % 2 == y % 2:
                diagonal_moves.append(move)
        if diagonal_moves:
            return np.random.choice(diagonal_moves)
        return np.random.choice(legal_moves)
    
class NeuralAgent:

    def __init__(self, env: env1.GoEnv = None, policy_path = None, name = None):

        self.env = env
        self.board_size = env.size
        if policy_path is None:
            self.nn = self._init_nn()
        else:
            self.load(policy_path)
        self.name = time.time() if name is None else name

    def sim_all_moves(self, rep, turn, capture_prop=0.01):
        """
        Simulate all possible moves including pass and return {move: q_predict} dictionary.
        """
        board = env1.GoBoard(size = self.board_size)
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

            # # if matrix only has one zero left after playing, set its q value to -1
            # if np.sum(matrix == 0) == 1:
            #     q[move] = -1
            #     self.env.pop(board)
            #     continue

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

    def _predict(self, images, others, verbose=0):
        # get lowest hashed symmetry
        syms = [self.collapse_symmetries(image) for image in images]
        syms = np.array(syms)
        return self.nn.predict({'image_input': syms, 'other_input': others.reshape(-1, 1)}, verbose=verbose)[:, 0]
    
    def _fit(self, images, others, targets, verbose=0):
        # Initialize arrays to store the images and others
        num_samples = len(images)
        sym_images = np.empty((num_samples, 5, 5))
        sym_others = np.empty((num_samples, 1))

        # Preprocess images and others
        for i in range(num_samples):
            sym_images[i] = self.collapse_symmetries(images[i])
            sym_others[i] = others[i]

        # Fit the network
        self.nn.fit({'image_input': sym_images, 'other_input': sym_others}, targets, verbose=verbose)

    def pass_adjust(self, q):
        """
        Adjust the q value for pass depending on the number of legal moves.
        """
        if len(q) == 1:
            # if there is only one legal move, it is pass. Q value does not matter.
            return q
        
        q["pass"] += self.board_size/(2 * len(q) - 2) - 0.5
        return q
        

    def act(self, state, epsilon = 0.1, greedy = "softmax"):
        """
        Choose an action based on the current state.
            - epsilon is the probability of choosing a random action.
            - greedy is the method of choosing an action.
                - "softmax": Softmax of all Q values if not random action.
                - "greedy": Choose the action with the highest Q value.
        """
        q = self.sim_all_moves(state[0], state[1])

        # adjust q value for pass depending on the number of legal moves
        # q = self.pass_adjust(q) # purpose: to encourage the agent to finish games when the board fills up.

        if random.random() < epsilon:
            # random action
            return random.choice(list(q.keys()))

        if greedy == "softmax":
            # softmax
            stretch = 3
            exp_q = {k: np.exp(stretch * v) for k, v in q.items()}
            exp_sum = sum(exp_q.values())
            q = {k: v / exp_sum for k, v in exp_q.items()}
            return random.choices(list(q.keys()), weights = list(q.values()))[0]
        
        elif greedy == "greedy":
            # greedy
            return max(q, key = q.get)
        
        else:
            raise ValueError("greedy must be 'softmax' or 'greedy'")

    def _init_nn(self):
        # Define input shapes
        image_input_shape = (self.board_size, self.board_size, 1)  # Shape of the image input
        other_input_shape = (1,)  # Shape of the other input

        # Define input layers
        image_input = Input(shape=image_input_shape, name='image_input')
        other_input = Input(shape=other_input_shape, name='other_input')

        # Convolutional layers for image processing
        conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(image_input)
        conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer1)

        # Flatten convolutional output
        conv_flat = Flatten()(conv_layer2)

        # Concatenate convolutional output with other input
        concatenated = concatenate([conv_flat, other_input])

        # Dense layers for combined features
        dense1 = Dense(128, activation='relu')(concatenated)
        dense2 = Dense(16, activation='relu')(dense1)
        output = Dense(1, activation='sigmoid')(dense2)

        # Define model
        model = Model(inputs=[image_input, other_input], outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Print model summary
        model.summary() 

        return model

    def train(self, N, batch_size=32, gamma=0.9, epsilon=0.1, greedy="softmax", capture_prop=0.01):
        self.replay_buffer = []
        try:
            for episode in tqdm.tqdm(range(N), desc=f'Working on {N} episodes of training'):
                # Initialize the state
                state = self.env.reset()
                done = False
                while not done:
                    # Choose action
                    action = self.act(state, epsilon=epsilon, greedy="softmax")
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
                        q = self.sim_all_moves(-1 * state[0], -1 * state[1])
                        max_q = max(q.values())
                    target = np.array([reward + gamma * max_q])

                    # Add this state to the replay buffer
                    self.replay_buffer.append((state, target))

                    # flip the state for the next player
                    state = (-1 * state[0], -1 * state[1])

                    # Train the network using experience replay
                    if len(self.replay_buffer) >= batch_size:
                        minibatch = random.sample(self.replay_buffer, batch_size)
                        images = np.array([sample[0][0] for sample in minibatch])  # Extracting images
                        others = np.array([sample[0][1] for sample in minibatch])  # Extracting others
                        targets = np.array([sample[1] for sample in minibatch])
                        self._fit(images, others, targets)
        except KeyboardInterrupt:
            print("Training interrupted")
            if episode < 10:
                print("Training did not complete 10 episodes. Nothing will be saved.")
                return None
        self.save(f"nn_policies/size{self.board_size}/{self.name}.keras")
        return self.nn    

    def save(self, path):
        Model.save(self.nn, path)

    def load(self, path):
        self.nn = keras.models.load_model(path)
    
    def play_human(self, color = 1):
        state = self.env.reset()
        done = False
        self.env.board.print_board()
        while not done:
            if self.env.board.turn == color:
                action = self.act(state, epsilon=0, greedy="greedy")
                state, reward, done, _ = self.env.step(action)
                print(action)
                self.env.board.print_board()
            else:
                action = input("Enter move: ")
                if action.upper() == "RESIGN":
                    print("You resigned.")
                    return None
                try:
                    state, reward, done, _ = self.env.step(action)
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

    # later enhancement: collapse symmetries
    def collapse_symmetries(self, matrix):
        """
        Return the lowest hashed representation of all symmetries of the state.
        """
        # All symmetries of dihedral group D4
        e = matrix
        r1 = np.rot90(matrix)
        r2 = np.rot90(matrix, 2)
        r3 = np.rot90(matrix, 3)
        s = np.flip(matrix, 0)
        sr1 = np.rot90(s)
        sr2 = np.rot90(s, 2)
        sr3 = np.rot90(s, 3)

        # sort by hash
        symmetries = [e, r1, r2, r3, s, sr1, sr2, sr3]
        symmetries = sorted(symmetries, key = lambda x: hash(str(x)))
        # TODO: it might make more sense to sort by some other metric for deep learning to
        # maintain a manifold!
        return symmetries[0]
                
# test nn agent
env = env1.GoEnv(size = 5)
agent = NeuralAgent(env, policy_path= r"nn_policies\size5\conv2.keras", name = "conv2")
agent.train(150, batch_size=32, gamma=0.99, epsilon=0.1, greedy="greedy")
agent.play_human(color = 1)
print("\n"*5)
agent.play_human(color = -1)