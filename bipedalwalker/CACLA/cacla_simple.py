import random
import gym
import numpy as np
from scipy import integrate
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import progressbar
import logging
from bipedalwalker.utils import utils
import matplotlib.pyplot as plt

EPISODES = 10000


class CACLAmodel:
    def __init__(self, state_size, action_size, action_min, action_max):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space_low = action_min
        self.action_space_high = action_max
        self.stateRMS = utils.obsRunningMeanStd(state_size)
        # HyperParams
        self.gamma = 0.99  # Discount rate
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.learning_rate_decay = 0.0
        self.sigma = 1.0
        self.sigma_min = 0.1
        self.sigma_decay = 0.999
        self.batch_size = 32
        # Build NN
        self.critic_model = self._build_critic_model()
        self.actor_model = self._build_actor_model()

    def _build_critic_model(self):  # Neural Net for CACLA learning Model - critic
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Returns the Value for best policy
        model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return model

    def _build_actor_model(self):  # Neural Net for CACLA learning Model - actor
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.actor_lr))
        return model

    def exploration(self, state, action_space):  # SPG-OffGE - offline Gaussian exploration
        act_values = self.actor_model.predict(state)
        actions = act_values + np.random.normal(0, self.sigma, size=action_space)
        actions = actions.clip(self.action_space_low, self.action_space_high)
        return actions[0]


    def train_critic(self, state, reward, next_state):
        state = np.reshape(state, (1, self.state_size))
        next_state = np.reshape(next_state, (1, self.state_size))
        target = reward + self.gamma*self.critic_model.predict(next_state)
        self.critic_model.fit(state, target, epochs=1, verbose=0)

    def train_actor(self, state, action, tempDiffErr):
        state = np.reshape(state, (1, self.state_size))
        action = np.reshape(action, (1, self.action_size))
        target = action    
        if tempDiffErr > 0:
            target = action
            self.actor_model.fit(state, target, epochs=1, verbose=0)
            # if self.sigma > self.sigma_min:
            #     self.sigma *= self.sigma_decay

    def getTDE(self, state, reward, next_state):
        state = np.reshape(state, (1, self.state_size))
        next_state = np.reshape(next_state, (1, self.state_size))
        target = reward + self.gamma*self.critic_model.predict(next_state)
        return target - self.critic_model.predict(state)

    def load(self, critic_filename, actor_filename):
        self.critic_model.load_weights(critic_filename)
        self.actor_model.load_weights(actor_filename)

    def save(self, critic_filename, actor_filename):
        self.critic_model.save_weights(critic_filename)
        self.actor_model.save_weights(actor_filename)


if __name__ == "__main__":

    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    model = CACLAmodel(state_size, action_size, env.action_space.low, env.action_space.high)

    actor_filename = "BipedalWalker-actorV2.h5"
    critic_filename = "BipedalWalker-criticV2.h5"

    # model.load(critic_filename, actor_filename)

    all_the_tempDiffs = []

    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        # Initialize the X position after reset. We will use this to calculate reward
        posX = 0.0
        done = False
        
        total_epoch_reward = 0

        while not done:
            if e > 100:
                env.render()

            # Updating running mean and deviation for the state vector.
            model.stateRMS.update(state, state_size)

            # Performing an action.
            norm_state = utils.normalize(state, model.stateRMS)
            actions = model.exploration(np.reshape(norm_state, (1, state_size)), action_size)
            next_state, reward, done, _ = env.step(actions)

            # Measurements of progress.
            x_vel = state[2]
            posX = env.env.hull.position.x

            norm_next_state = utils.normalize(next_state, model.stateRMS)

            tempDiffErr = model.getTDE(norm_state, reward, norm_next_state)
            model.train_actor(norm_state, actions, tempDiffErr)
            model.train_critic(norm_state, reward, norm_next_state)
            state = next_state

            all_the_tempDiffs.append(tempDiffErr[0][0])
            total_epoch_reward += reward

            if done:
                logging.debug("episode: {}/{}, epoch_reward: {}, sigma: {:.2}, posX: {}, velX: {}"
                              .format(e, EPISODES, total_epoch_reward, model.sigma, posX, x_vel))
                if reward > 200:
                    logging.warning("Hallelujah! reward: {}".format(reward))
                if model.sigma > model.sigma_min:
                    model.sigma *= model.sigma_decay

        model.save(critic_filename, actor_filename)
        if (e % 500 == 0):
            plt.plot(all_the_tempDiffs)
            plt.ylabel('Tempdiffs over time')
            plt.show()
