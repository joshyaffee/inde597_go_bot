import environment1 as GoEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras
import tqdm

class DistributionalDQN:
    def __init__(self, state_shape, num_actions, num_atoms, learning_rate, gamma):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Initialize networks
        self.online_net = self.build_network()
        self.target_net = self.build_network()
        self.target_net.set_weights(self.online_net.get_weights())
        
        # Initialize distributional parameters
        self.v_min = -10
        self.v_max = 10
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = np.linspace(self.v_min, self.v_max, self.num_atoms)
        
    def build_network(self):
        # Define input layers for the board matrix and turn indicator
        board_input = Input(shape=self.state_shape, name='board_input')
        turn_input = Input(shape=(1,), name='turn_input')

        # Convolutional layers
        conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(board_input)
        max_pooling_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
        flat_layer = Flatten()(max_pooling_layer)
        flat_board = Flatten()(board_input)

        # Concatenate flattened board and turn indicator
        concatenated_layers = Concatenate()([flat_layer, flat_board, turn_input])

        # Fully connected layers
        hidden_layer1 = Dense(256, activation='relu')(concatenated_layers)
        hidden_layer2 = Dense(256, activation='relu')(hidden_layer1)

        # Output layer
        output_layer = Dense(self.num_actions * self.num_atoms, activation='softmax')(hidden_layer2)

        # Reshape output to match the distributional Q-value shape
        output_layer_reshaped = tf.keras.layers.Reshape((self.num_actions, self.num_atoms))(output_layer)

        # Define model
        model = Model(inputs=[board_input, turn_input], outputs=output_layer_reshaped)
        
        return model
        
    def train(self, batch_size):
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.env.sample_batch(batch_size)
        
        # Predict q-values for current and next states
        online_q_values = self.online_net.predict(states)
        next_online_q_values = self.online_net.predict(next_states)
        next_target_q_values = self.target_net.predict(next_states)
        
        # Select best actions for next states using online network
        next_actions = np.argmax(next_online_q_values, axis=1)
        
        # Compute TD target
        target_probabilities = self.compute_target_probabilities(next_target_q_values, next_actions, rewards, dones)
        
        # Compute loss and update online network
        loss = self.online_net.train_on_batch(states, actions, target_probabilities)
        
        return loss
        
    def compute_target_probabilities(self, next_q_values, next_actions, rewards, dones):
        target_probabilities = np.zeros((len(rewards), self.num_atoms))
        
        for i in range(len(rewards)):
            if dones[i]:
                target_distribution = np.zeros(self.num_atoms)
                value = min(self.v_max, max(self.v_min, rewards[i]))
                idx = (value - self.v_min) / self.delta_z
                target_distribution[int(idx)] += 1
                target_probabilities[i] = target_distribution
            else:
                next_action = next_actions[i]
                target_distribution = np.zeros(self.num_atoms)
                for j in range(self.num_atoms):
                    tz = rewards[i] + self.gamma * self.support[j] * (1 - dones[i])
                    idx = (tz - self.v_min) / self.delta_z
                    l = int(np.floor(idx))
                    u = int(np.ceil(idx))
                    if u >= 0 and l >= 0:
                        target_distribution[l] += -next_q_values[i, next_action, j] * (u - idx)  # Negative for zero-sum game
                    if u < self.num_atoms and l >= 0:
                        target_distribution[u] += -next_q_values[i, next_action, j] * (idx - l)  # Negative for zero-sum game
                target_probabilities[i] = target_distribution
        
        return target_probabilities
        
    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.online_net.predict(state[np.newaxis])
            return np.argmax(q_values)
        
    def step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done
    
    def update_target_network(self):
        self.target_net.set_weights(self.online_net.get_weights())
        
    def reset(self):
        self.env.reset()
        
class GoAgent:
    def __init__(self, env, policy_path=None, name=None, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, replay_buffer_size=10000, batch_size=32, num_atoms=51, learning_rate=0.001, gamma=0.99):
        self.env = env
        state_shape = (env.size, env.size, 1)  # Assuming a square board with a single channel for grayscale
        num_actions = env.size * env.size + 1  # Number of possible actions including passing
        self.dqn_agent = DistributionalDQN(state_shape, num_actions, num_atoms, learning_rate, gamma)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.replay_buffer = []
        self.policy_path = policy_path
        if policy_path is not None:
            self.load_policy(policy_path)
        self.name = str(name)
        self.callbacks = []
        
    def update_replay_buffer(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def sample_batch(self):
        batch = np.random.choice(self.replay_buffer, size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def train(self, num_episodes):
        # Define callbacks
        # checkpoint_callback = ModelCheckpoint(filepath= + '/' + self.name + '_checkpoint.h5', 
        #                                       monitor='loss', 
        #                                       save_best_only=True, 
        #                                       save_weights_only=True, 
        #                                       mode='min', 
        #                                       verbose=1)
        # tensorboard_callback = TensorBoard(log_dir=self.policy_path + '/' + self.name + '_logs', 
        #                                    histogram_freq=1)
        # self.callbacks = [tensorboard_callback]
        # self.callbacks = [checkpoint_callback, tensorboard_callback]
        
        for episode in tqdm.tqdm(range(num_episodes)):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.dqn_agent.epsilon_greedy_policy(state, self.epsilon)
                next_state, reward, done = self.env.step(action)
                self.update_replay_buffer(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.sample_batch()
                    loss = self.dqn_agent.train(states, actions, rewards, next_states, dones)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
            
        # Save policy after training is complete
        self.save_policy()
            
    def save_policy(self, path):
        Model.save(self.nn, path)
        
    def load_policy(self, path = None):
        if path == None:
            path = f"nn_policies/size{self.board_size}/{self.name}.keras"
        self.nn = keras.models.load_model(path)

    def play_human(self, color=1):
        state = self.env.reset()
        done = False
        self.env.render()  # Assuming the render method prints the board
        while not done:
            if self.env.board.turn == color:
                action = self.dqn_agent.epsilon_greedy_policy(state, epsilon=0)  # Use greedy policy
                next_state, reward, done = self.dqn_agent.step(state, epsilon=0)  # Use greedy policy
                self.update_replay_buffer(state, action, reward, next_state, done)
                self.env.render()  # Assuming the render method prints the board
            else:
                action = input("Enter move: ")
                if action.upper() == "RESIGN":
                    print("You resigned.")
                    return None
                try:
                    state, reward, done = self.dqn_agent.step(state, epsilon=0)  # Use greedy policy
                except:
                    print("Illegal move")
                    continue
                self.update_replay_buffer(state, action, reward, next_state, done)
                self.env.render()  # Assuming the render method prints the board

        x = color * reward * self.env.board.turn
        if x == 1:
            print("You won!")
        elif x == -1:
            print("You lost!")
        else:
            print("It was a draw.")
        return None

# create agent, rain for 10 episodes, and save the policy
agent = GoAgent(env=GoEnv.GoEnv(), name = "distdqn1")
agent.train(10)
agent.play_human()
agent.play_human(color=-1)
