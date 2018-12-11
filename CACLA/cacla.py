import random
import gym
import numpy as np
from scipy import integrate
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys


EPISODES = 10000
FPS = 50
dt = 1/FPS


class CACLACritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # Discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):  # Neural Net for CACLA learning Model - critic
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_size, activation='relu'))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Returns the Value for best policy
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def learn(self, state, reward, next_state):
        target = reward + self.gamma*self.model.predict(next_state)
        self.model.fit(state, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class CACLAActor:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.sigma = 1.0
        self.sigma_min = 0.01
        # self.sigma_decay = 0.9995
        self.sigma_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):  # Neural Net for CACLA learning Model - actor
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_size, activation='relu'))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        act_values = self.model.predict(state)
        actions = []  # Actions to perform using a Gaussian exploration method
        for val in act_values[0]:
            actions.append(np.random.normal(val, self.sigma))
        return np.array(actions)

    def learn(self, state, action, value_state, value_nextState):
        if value_nextState > value_state:
            target = action - self.model.predict(state)
            self.model.fit(state, target, epochs=1, verbose=0)
            # if self.sigma > self.sigma_min:
            #     self.sigma *= self.sigma_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = CACLAActor(state_size, action_size)
    critic = CACLACritic(state_size, action_size)

    actor_filename = "BipedalWalker-actor.h5"
    critic_filename = "BipedalWalker-critic.h5"

    X_vel_offset = 30.0

    # so we can train it with different params
    if len(sys.argv) > 1:
        actor.sigma_decay = float(sys.argv[1])
        X_vel_offset = float(sys.argv[2])
        actor_filename = sys.argv[3]
        critic_filename = sys.argv[4]
    print("actor_filename: {}, critic_filename: {}, sigmadecay: {}, xvel: {}".format(actor_filename, critic_filename, actor.sigma_decay, X_vel_offset))

    # actor.load(actor_filename)
    # critic.load(critic_filename)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # Initialize the X position after reset. We will use this to calculate reward
        posX = 0.0
        velX_array = []
        done = False

        while not done:
            # if e > 2000:
            env.render()
            action = actor.act(state)
            next_state, reward, done, _ = env.step(action)

            velX_array.append(state[0][2])

            posX = integrate.trapz(velX_array, dx=dt)
            posRewardFactor = 10.0

            reward += posRewardFactor*posX

            # TODO: WHAT IF WE INTEGRATE THE VEL TO GET POS AND THEN REWARD BASED ON HOW FAR WE MOVED
            # x_vel = state[0][2]
            # r_X_vel = X_vel_offset
            # # r_X_vel = X_vel_offset + actor.sigma*100.0
            # if x_vel > 0:
            #     reward += r_X_vel*x_vel
            # # print('vel x = {}, reward = {}'.format(x_vel, reward))

            next_state = np.reshape(next_state, [1, state_size])
            actor.learn(state, action, critic.model.predict(state), critic.model.predict(next_state))
            critic.learn(state, reward, next_state)

            state = next_state
            if done:
                print("episode: {}/{}, reward: {}, sigma: {:.2}" .format(e, EPISODES, reward, actor.sigma))
                print(" posX: {}, rewardposX: {}" .format(posX, posRewardFactor*posX))
                if actor.sigma > actor.sigma_min:
                   actor.sigma *= actor.sigma_decay
        if e > 2000:
            actor.save(actor_filename)
            critic.save(critic_filename)

