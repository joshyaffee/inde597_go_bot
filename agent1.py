"""
Agent super class for Go game
subclasses will implement the policy function
"""

import environment1 as env1
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
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

    def __init__(self, env: env1.GoEnv = None, policy_path = None):

        self.env = env
        self.board_size = env.size
        self.nn = self._init_nn()
        if policy_path is None:
            pass
        else:
            self.load(policy_path)

    def act(self, epsilon = 0.1, greedy = "softmax"):
        """
        Choose an action based on the current state.
            - epsilon is the probability of choosing a random action.
            - greedy is the method of choosing an action.
                - "softmax": Softmax of all positive Q values if not random action.
                - "greedy": Choose the action with the highest Q value.
            
            
        """
        legal_moves = self.env.get_legal_moves()
        q = {lm: 0 for lm in legal_moves}
        board = self.env.board.__deepcopy__()

        legal_moves.remove("pass")
        if np.random.rand() < epsilon:
            if len(legal_moves) == 0:
                return "pass"
            return random.choice(legal_moves)
        else:        
            for move in legal_moves:
                if len(legal_moves) + 1 != len(board.get_legal_moves()):
                    print("error detected")
                    pass
                # simulate move
                board, ((matrix, turn), reward, done, _) = self.env.look_ahead(move, board, representation = 1)

                # if matrix only has one zero left after playing, set its q value to -1
                if np.sum(matrix == 0) == 1:
                    q[move] = -1
                    self.env.pop(board)
                    continue

                # get Q value
                # q[move] = -1 * self.nn.predict([np.array([matrix]), turn])
                # multiply by -1 since we consider the opponent's turn
                q[move] = -1 * self.nn.predict({'image_input': matrix.reshape(1, 5, 5, 1), 'other_input': np.array([[turn]])}, verbose = 0)[0,0]
                self.env.pop(board)

        if greedy == "softmax":
            # softmax
            max_q = np.max(list(q.values()))
            exp_q = {k: np.exp(v - max_q) for k, v in q.items() if v > 0}  # Subtracting max_q for numerical stability
            exp_sum = sum(exp_q.values())
            q = {k: v / exp_sum for k, v in exp_q.items()}
            if len(q) == 0:
                return "pass"
            
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
        output = Dense(1, activation='sigmoid')(dense1)

        # Define model
        model = Model(inputs=[image_input, other_input], outputs=output)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        # Print model summary
        model.summary() 

        return model

    def train(self, N, batch_size=32, gamma=0.9, epsilon=0.1, greedy="softmax"):
        self.replay_buffer = []
        try:
            for episode in tqdm.tqdm(range(N), desc=f'Working on {N} episodes of training'):
                # Initialize the state
                state = self.env.reset()
                done = False
                while not done:
                    # Choose action
                    action = self.act(epsilon=epsilon, greedy=greedy)
                    # Take action
                    next_state, reward, done, _ = self.env.step(action)
                    matrix, turn = next_state
                    # Compute target
                    target = reward - gamma * self.nn.predict({'image_input': matrix.reshape(1, 5, 5, 1), 'other_input': np.array([[turn]])}, verbose = 0)[0][0]
                    
                    # Add this state to the replay buffer
                    self.replay_buffer.append((state, target))
                    # Train the network using experience replay
                    if len(self.replay_buffer) >= batch_size:
                        minibatch = random.sample(self.replay_buffer, batch_size)
                        for state, target in minibatch:
                            self.nn.fit({'image_input': state[0].reshape(1, 5, 5, 1), 'other_input': np.array([[state[1]]])}, target.reshape(1,-1), verbose=0)
                    # Update state
                    state = next_state
        except KeyboardInterrupt:
            print("Training interrupted")
            if episode < 10:
                print("Training did not complete 10 episodes. Nothing will be saved.")
                return None
        self.save(f"nn_policies/size{self.board_size}/{time.time()}.h5")
        return self.nn   

    def save(self, path):
        self.nn.save_weights(path) 

    def load(self, path):
        self.nn.load_weights(path)
    
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
        x = color * reward * self.env.board.turnC
        if x == 1:
            print("You won!")
        elif x == -1:
            print("You lost!")
        else:
            print("It was a draw.")
        return None

    # later enhancement: collapse symmetries
    # def collapse_symmetries(self, state):
    #     """
    #     Return the lowest hashed representation of all symmetries of the state.
    #     """
    #     matrix, turn = state
        
    #     # All symmetries of dihedral group D4
    #     e = matrix
    #     r1 = np.rot90(matrix)
    #     r2 = np.rot90(matrix, 2)
    #     r3 = np.rot90(matrix, 3)
    #     s = np.flip(matrix, 0)
    #     sr1 = np.rot90(s)
    #     sr2 = np.rot90(s, 2)
    #     sr3 = np.rot90(s, 3)

    #     # sort by hash
    #     symmetries = [e, r1, r2, r3, s, sr1, sr2, sr3]
    #     symmetries = sorted(symmetries, key = lambda x: hash(str(x)))
    #     return symmetries[0], turn
                
# test nn agent
env = env1.GoEnv(size = 5)
agent = NeuralAgent(env, policy_path = r"nn_policies/size5/1710135423.8806815.h5")
# agent.train(25)
agent.play_human(color = 1)
print("\n"*5)
agent.play_human(color = -1)