import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import time
from environment2 import GoEnv
import tqdm

def index_to_action(index, board_size):
    if index == board_size * board_size:
        return 'pass'
    else:
        x = index // board_size
        y = index % board_size
        return chr(y + ord("A")) + str(x + 1)
    
def move_to_index(move, board_size):
    if move == 'pass':
        return board_size * board_size
    else:
        col = ord(move[0].upper()) - ord('A')  # Convert letter to column index (A=0, B=1, ...)
        row = int(move[1:]) - 1  # Convert numeric string to row index, adjusting for zero-indexing
        return row * board_size + col

def build_actor_model(board_size):
    input_shape = (board_size, board_size, 1)  # Only one channel for the board input
    inputs = Input(shape=input_shape, name='board_input')
    
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Assuming board_size * board_size possible actions (positions on the board)
    outputs = Dense(board_size * board_size + 1, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def build_critic_model(board_size):
    input_shape = (board_size, board_size, 1)
    inputs = Input(shape=input_shape, name='board_input')
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    
    x = Dense(256, activation='relu')(x)
    
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

class ActorCriticModel:
    def __init__(self, actor_model, critic_model, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.1):
        self.actor_model = actor_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.critic_model = critic_model

    def act(self, state_tensor, legal_moves):
        action_probs = self.actor_model(state_tensor).numpy()[0]
        mask = np.zeros_like(action_probs)
        board_size = int(np.sqrt(len(action_probs) - 1))  # -1 to exclude the pass action
        indices = [move_to_index(move, board_size) for move in legal_moves]  # Assume move_to_index converts move to index
        mask[indices] = 1

        # Apply mask to probabilities and renormalize
        masked_probs = action_probs * mask
        if np.sum(masked_probs) == 0:
            masked_probs += mask  # to avoid division by zero if no legal move has a probability
        normalized_probs = masked_probs / np.sum(masked_probs)

        if np.random.rand() < self.epsilon:
            idx = np.random.choice(len(normalized_probs), p=normalized_probs)
        else:
            idx = np.argmax(normalized_probs)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return index_to_action(idx, board_size)  # Assuming index_to_action translates index back to move format


    def compute_value(self, state_tensor):
        return self.critic_model(state_tensor)

    def update(self, state_tensor, next_state_tensor, reward, done, gamma, optimizer):
        with tf.GradientTape(persistent=True) as tape:
            value = self.compute_value(state_tensor)[0, 0]
            next_value = self.compute_value(next_state_tensor)[0, 0]
            target_value = reward - gamma * next_value * (1 - int(done))
            
            critic_loss = tf.square(target_value - value)

            action_probs = self.actor_model(state_tensor)
            action_taken = np.argmax(action_probs[0].numpy())
            log_prob = tf.math.log(action_probs[0, action_taken])
            actor_loss = -log_prob * (target_value - value)

        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))
        optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables)) # errors
        # use fit instead of apply_gradients
        # self.actor_model.fit(state_tensor, target_value, verbose=0)
        del tape
        return critic_loss

class ActorCriticAgent:

    def __init__(self, env = None, actor_path = None, critic_path = None, name = None):

        self.env = env
        self.board_size = env.size
        # enumerate all actions
        self.action_space = env.get_actions()
        if actor_path is None:
            self.actor_model = build_actor_model(self.board_size)
            self.prev_episodes = 0
        else:
            self.actor_model = keras.models.load_model(actor_path)
            self.prev_episodes = int(actor_path.split("_")[-1].split(".")[0])
        if critic_path is None:
            self.critic_model = build_critic_model(self.board_size)
        else:
            self.critic_model = keras.models.load_model(critic_path)
        self.model = ActorCriticModel(self.actor_model, self.critic_model)
        self.name = time.time() if name is None else name

    def train_symmetric_actor_critic(self, env, optimizer, num_episodes, gamma=0.99):
        optimizer.build(self.actor_model.trainable_variables + self.critic_model.trainable_variables)
        with tqdm.tqdm(total=num_episodes, desc="Training Progress", unit="episode") as pbar:
            for episode in range(self.prev_episodes, self.prev_episodes + num_episodes):
                state = env.reset()
                state = np.array(state, dtype=np.int8)  # Cast to np.int8 for efficient memory usage
                total_rewards = [0, 0]
                cumulative_critic_loss = 0
                move_count = 0

                while True:
                    # Player 1's turn
                    state_tensor = tf.convert_to_tensor([state], dtype=tf.int8)
                    legal_moves = env.get_legal_moves()
                    # if more moves than 3 * number of intersections, pass
                    if move_count > 3 * self.board_size * self.board_size:
                        action1 = 'pass'
                    else:
                        action1 = self.model.act(state_tensor, legal_moves)
                    next_state, reward1, done, extra = env.step(action1)
                    reward1 += 0.01 * extra['num_captured']  # Add small reward for captures
                    next_state = np.array(next_state, dtype=np.int8)
                    next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.int8)
                    critic_loss = self.model.update(state_tensor, next_state_tensor, reward1, done, gamma, optimizer)
                    cumulative_critic_loss += critic_loss
                    move_count += 1
                    total_rewards[0] += reward1

                    if done:
                        break

                    # Transform state for Player 2
                    state_transformed = -next_state
                    state_tensor_transformed = tf.convert_to_tensor([state_transformed], dtype=tf.int8)
                    legal_moves = env.get_legal_moves()
                    # if more moves than 3 * number of intersections, pass
                    if move_count > 3 * self.board_size * self.board_size:
                        action1 = 'pass'
                    else:
                        action2 = self.model.act(state_tensor_transformed, legal_moves)
                    next_state, reward2, done, _ = env.step(action2)
                    reward2 += 0.01 * extra['num_captured']  # Add small reward for captures
                    next_state_transformed = -np.array(next_state, dtype=np.int8)
                    next_state_transformed_tensor = tf.convert_to_tensor([next_state_transformed], dtype=tf.int8)
                    critic_loss = self.model.update(state_tensor, next_state_tensor, reward2, done, gamma, optimizer)
                    cumulative_critic_loss += critic_loss
                    move_count += 1
                    total_rewards[1] += reward2

                    if done:
                        break

                    state = next_state

                # Calculate the average critic loss
                average_critic_loss = cumulative_critic_loss / move_count if move_count else 0

                # Update tqdm with current episode info
                pbar.set_description(f"Episode {episode}: Avg Critic Loss {average_critic_loss:.4f}, Moves {move_count}")
                pbar.update(1)

                if episode % 10 == 0:  # Save every 10 episodes
                    self.env.board.print_history()
                    self.save(f'nn_policies\\size5\\{self.name}_actor_{episode}.keras', f'nn_policies\\size5\\{self.name}_critic_{episode}.keras')

    def save(self, actor_path, critic_path):
        Model.save(self.actor_model, actor_path)
        Model.save(self.critic_model, critic_path)


    def load(self, actor_path, critic_path):
        self.actor_model = keras.models.load_model(actor_path)
        self.critic_model = keras.models.load_model(critic_path)
        self.model = ActorCriticModel(self.actor_model, self.critic_model)

    def act(self, state, turn):
        # To be called when playing against another agent/human
        state_tensor = tf.convert_to_tensor([state * turn], dtype=tf.int8)
        legal_moves = env.get_legal_moves()
        return self.model.act(state_tensor, legal_moves)
    
    def play_human(self, color = 1):
        self.env.reset()
        done = False
        self.env.board.print_board()
        while not done:
            if self.env.board.turn == color:
                action = self.act(self.env.board.matrix_board, self.env.board.turn)
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

env = GoEnv(5)
agent = ActorCriticAgent(env, name = 'ac5x5', critic_path = r"nn_policies\size5\ac5x5_critic_1000.keras", actor_path = r"nn_policies\size5\ac5x5_actor_160.keras")
optimizer = keras.optimizers.Adam(learning_rate=0.01)
try:
    agent.train_symmetric_actor_critic(env, optimizer, 2001, gamma=0.99)
except(KeyboardInterrupt):
    print("Training interrupted.")

# test human play
agent.play_human(color = 1)
agent.play_human(color = -1)