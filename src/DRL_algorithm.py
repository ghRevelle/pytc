
"""
# Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
model.evaluate(x_test, y_test)
"""
"""
import gymnasium as gym
import numpy as np
import gymnasium as gym  # Changed from 'gym' to 'gymnasium'
from collections import deque
import random

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())  # Shows how many CUDA devices (GPUs) are available


# 1. Define the neural network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 3. DQN Agent Class
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-3, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore: Random action
        else:
            state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            q_values = model(state_tensor)
            action = tf.argmax(q_values[0]).numpy()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state)
        
        # Store experience in memory
        memory.append((state, action, reward, next_state, done))
        if len(memory) > max_memory_length:
            memory.pop(0)
        
        state = next_state
        episode_reward += reward
        
        # Sample from memory and train model
        if len(memory) >= batch_size:
            states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = sample_memory(batch_size)
            
            next_q_values = target_model.predict(next_states_mb)
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q = rewards_mb + (1 - dones_mb) * gamma * max_next_q_values
            
            masks = tf.one_hot(actions_mb, num_actions)
            
            with tf.GradientTape() as tape:
                q_values = model(states_mb)
                q_action = tf.reduce_sum(q_values * masks, axis=1)
                loss = loss_function(target_q, q_action)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Update target network every 10 episodes
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
    
    print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
 """