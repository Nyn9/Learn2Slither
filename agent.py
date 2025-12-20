import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, lr=0.01, gamma=0.9, epsilon=1.0):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.nb_actions = 4
        self.state_size = state_size
        self.memory = []
        self.model = self.init_neural_network()

    def init_neural_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.nb_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.nb_actions)
        value = self.model.predict(state, verbose=0)
        return np.argmax(value[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward,
                next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        q_values = self.model.predict(states, verbose=0)
        q_next_values = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + \
                                    self.gamma * np.amax(q_next_values[i])

        self.model.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
