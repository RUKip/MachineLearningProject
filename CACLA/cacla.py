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
import utils

EPISODES = 10000
MAX_TIMESTEPS = 1600  # It is 2000 for hardcore..


class CACLAmodel:
    def __init__(self, state_size, action_size, action_min, action_max):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = utils.Memory()
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
        self.n_iter = 100
        # Build NN
        self.critic_model = self._build_critic_model()
        self.actor_model = self._build_actor_model()

    def get_hyper_params(self):
        params = (self.gamma,
                  self.critic_lr,
                  self.actor_lr,
                  self.learning_rate_decay,
                  self.sigma,
                  self.sigma_min,
                  self.sigma_decay,
                  self.batch_size,
                  self.n_iter)
        return params

    def _build_critic_model(self):  # Neural Net for CACLA learning Model - critic
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_size, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Returns the Value for best policy
        model.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return model

    def _build_actor_model(self):  # Neural Net for CACLA learning Model - actor
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_size, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(lr=self.actor_lr))
        return model

    def exploration(self, state, action_space):  # SPG-OffGE - offline Gaussian exploration
        act_values = self.actor_model.predict(state)
        actions = act_values + np.random.normal(0, self.sigma, size=action_space)
        actions = actions.clip(self.action_space_low, self.action_space_high)
        return actions[0]


    def train_critic(self):
        for _ in range(self.n_iter):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            for i in range(self.batch_size):
                st = np.reshape(utils.normalize(state[i], self.stateRMS), (1, self.state_size))
                st1 = np.reshape(utils.normalize(next_state[i], self.stateRMS), (1, self.state_size))
                rt = reward[i]
                target = rt + self.gamma * self.critic_model.predict(st1)
                self.critic_model.fit(st, target, epochs=1, verbose=0)

    def train_actor(self):
        for _ in range(self.n_iter):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            for i in range(self.batch_size):
                st = np.reshape(utils.normalize(state[i], self.stateRMS), (1, self.state_size))
                st1 = np.reshape(utils.normalize(next_state[i], self.stateRMS), (1, self.state_size))
                at = np.reshape(action[i], (1, self.action_size))
                rt = reward[i]
                tempDiffErr = self.getTDE(st, st1, rt)
                if tempDiffErr > 0:
                    target = at
                    self.actor_model.fit(st, target, epochs=1, verbose=0)

    def getTDE(self, state, next_state, reward):
        target = reward + self.gamma * self.critic_model.predict(next_state)
        return target - self.critic_model.predict(state)

    def load(self, critic_filename, actor_filename):
        self.critic_model.load_weights(critic_filename)
        self.actor_model.load_weights(actor_filename)

    def save(self, critic_filename, actor_filename):
        self.critic_model.save_weights(critic_filename)
        self.actor_model.save_weights(actor_filename)


if __name__ == "__main__":

    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(message)s')
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    model = CACLAmodel(state_size, action_size, env.action_space.low, env.action_space.high)
    critic_filename = "BipedalWalker-critic.h5"
    actor_filename = "BipedalWalker-actor.h5"

    logging.info("Starting exercise: actor_filename: {}, critic_filename: {}\n".format(actor_filename, critic_filename))
    paramList = ["gamma",
                 "critic_lr",
                 "actor_lr",
                 "learning_rate_decay",
                 "sigma",
                 "sigma_min",
                 "sigma_decay",
                 "batch_size",
                 "n_iter"]
    msg = utils.print_hyperparam(paramList, model.get_hyper_params())
    logging.info(msg)

    # model.load(critic_filename, actor_filename)
    # model.sigma = 0.0

    avg_reward = 0
    ep_reward = 0

    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        # Initialize the X position after reset. We will use this to calculate reward
        model.replay_buffer.clear()
        finishLine = False
        for t in range(MAX_TIMESTEPS):
            env.render()

            # Updating running mean and deviation for the state vector.
            model.stateRMS.update(state, state_size)

            # Performing an action.
            norm_state = utils.normalize(state, model.stateRMS)
            actions = model.exploration(np.reshape(norm_state, (1, state_size)), action_size)
            next_state, reward, done, _ = env.step(actions)

            # Measurements of progress.
            posX = env.env.hull.position.x
            x_vel = state[2]
            finishLine = ((reward > 200.0) or (posX > 100))
            if finishLine:
                logging.warning("Hallelujah! reward: {}, posX: {}".format(reward, posX))

            # Save values to buffer
            model.replay_buffer.append(state, actions, reward, next_state, done)
            state = next_state

            avg_reward += reward
            ep_reward += reward

            if done:
                # Train model.
                model.train_actor()
                model.train_critic()

                # Output variables to log.
                paramList = ["latest_r",
                             "ep_reward",
                             "avg_reward",
                             "sigma",
                             "posX",
                             "t"]
                params = (e+1, EPISODES, reward, ep_reward, avg_reward, model.sigma, posX, t)
                msg = utils.print_episode(paramList, params)
                logging.debug(msg)

                # Sigma decay.
                if model.sigma > model.sigma_min:
                    model.sigma *= model.sigma_decay
                else:
                    model.sigma = model.sigma_min

                break

        ep_reward = 0
        model.save(critic_filename, actor_filename)
