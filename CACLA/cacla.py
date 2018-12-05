import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 1000


class CACLAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for CACLA learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        # TODO: Test this at some point....
        # model.compile(loss='categorical_crossentropy',  # From https://keras.io/#getting-started-30-seconds-to-keras
        #               optimizer='sgd',
        #               metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return 2*np.random.random_sample((1, 4)) - 1
        act_values = self.model.predict(state)
        return act_values  # returns actions

    def replay(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        if done:
            target[0] = reward
        else:
            t = self.model.predict(next_state)[0]
            target[0] = reward + self.gamma * t
            # target = 1  # TODO: super-duper target formula
        self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # def load(self, name):
    #     self.model.load_weights(name)
    #
    # def save(self, name):
    #     self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = CACLAAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            # print("time: {}".format(time))
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action[0])
            # reward = reward if not done else -10  # TODO: WHY did he do this??
            next_state = np.reshape(next_state, [1, state_size])

            # agent.remember(state, action, reward, next_state, done)
            agent.replay(state, action, reward, next_state, done)

            state = next_state
            if done:
                print("episode: {}/{}, e: {:.2}"
                      .format(e, EPISODES, agent.epsilon))
            # if len(agent.memory) > batch_size:
            #     agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")