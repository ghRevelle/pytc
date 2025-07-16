
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
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 1. Create Environment
env = gym.make("CartPole-v1")

num_actions = env.action_space.n
state_shape = env.observation_space.shape

# 2. Build Q-Network Model
def create_q_model():
    inputs = layers.Input(shape=state_shape)
    layer1 = layers.Dense(24, activation='relu')(inputs)
    layer2 = layers.Dense(24, activation='relu')(layer1)
    outputs = layers.Dense(num_actions)(layer2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_q_model()
target_model = create_q_model()
target_model.set_weights(model.get_weights())

# 3. Define Optimizer and Loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.Huber()

# 4. Hyperparameters
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
memory = []

# 5. Experience Replay Memory Buffer
max_memory_length = 100000

# 6. Helper function to sample from memory
def sample_memory(batch_size):
    indices = np.random.choice(len(memory), batch_size)
    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in indices])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# 7. Training Loop
episodes = 500
for episode in range(episodes):
    state, info = env.reset()
    state = np.array(state)
    episode_reward = 0
    
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
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
 