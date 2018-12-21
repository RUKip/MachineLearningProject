import random
import gym
import numpy as np
from scipy import integrate
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys
import progressbar
import logging
from CACLA.utils import ReplayBuffer

EPISODES = 5000
MAX_TIMESTEPS = 2000


class CACLA:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # Discount rate
        self.learning_rate = 0.001
        # self.learning_rate = 0.5
        self.critic_model = self._build_critic_model()
        self.actor_model = self._build_actor_model()
        self.sigma = 0.5
        self.sigma_min = 0.1
        self.sigma_decay = 0.99

    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Returns the Value for best policy
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, action_space, min, max):
        act_values = self.actor_model.predict(state)
        actions = act_values + np.random.normal(0, self.sigma, size=action_space)
        actions = actions.clip(min, max)

        # actions = []  # Actions to perform using a Gaussian exploration method
        # for val in act_values[0]:
        #     actions.append(np.random.normal(val, self.sigma))

        return actions[0]

    def getTDE(self, state, reward, next_state):
        target = reward + self.gamma*self.critic_model.predict(next_state)
        return target - self.critic_model.predict(state)

    def learn(self, replay_buff, batch_size, n_iter):
        for i in range(n_iter):
            # ACTOR training
            buffer = replay_buff.buffer
            np.random.shuffle(buffer)
            state_arr = []
            target_arr = []
            for state, action, reward, tempDiffErr, next_state, done in buffer:
                if len(target_arr) >= 32:
                    break
                elif tempDiffErr > 0:
                    state_arr.append(state[0])
                    target = action
                    target_arr.append(target[0])
            target_arr = np.array(target_arr)
            state_arr = np.array(state_arr)
            if len(target_arr) == 32:
                self.actor_model.fit(state_arr, target_arr, batch_size=batch_size, epochs=1, verbose=0)

            # CRITIC training
            batch = replay_buff.sample(batch_size)
            state, action, reward, tempDiffErr, next_state, done = replay_buff.sample(batch_size)
            next_state = np.reshape(next_state, [batch_size, self.state_size])
            target_critic = np.reshape(reward, [len(reward), 1]) + self.gamma*self.critic_model.predict(next_state)
            state = np.reshape(state, [batch_size, self.state_size])

            self.critic_model.fit(state, target_critic, batch_size=batch_size, epochs=1, verbose=0)

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
    CACLA_model = CACLA(state_size, action_size)

    actor_filename = "BipedalWalker-actor.h5"
    critic_filename = "BipedalWalker-critic.h5"

    logging.info("Starting exercise: actor_filename: {}, critic_filename: {}, sigmadecay: {}"
                 .format(actor_filename, critic_filename, CACLA_model.sigma_decay))

    # CACLA_model.load(critic_filename, actor_filename)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    replay_buffer = ReplayBuffer()
    batch_size = 32
    n_iter = 100

    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        finishLine = False

        for t in range(MAX_TIMESTEPS):
            # env.render()

            action = CACLA_model.act(state, env.action_space.shape[0], env.action_space.low, env.action_space.high)
            next_state, reward, done, _ = env.step(action)

            posX = env.env.hull.position.x
            x_vel = state[0][2]
            finishLine = ((reward > 200.0) or (posX > 100))

            if finishLine:
                logging.warning("Hallelujah! reward: {}, posX: {}".format(reward, posX))

            next_state = np.reshape(next_state, [1, state_size])
            action = np.reshape(action, [1, action_size])

            tempDiffErr = CACLA_model.getTDE(state, reward, next_state)
            replay_buffer.add((state, action, reward, tempDiffErr, next_state, done))
            state = next_state

            avg_reward += reward
            ep_reward += reward

            if done or (t == (MAX_TIMESTEPS-1)):
                CACLA_model.learn(replay_buffer, batch_size, n_iter)
                logging.debug("episode: {}/{}, reward: {}, ep_reward: {}, avg_reward: {}, sigma: {:.2}, l_rate: {:.2}, "
                              "posX: {:.2}, velX: {:.2}".format(e, EPISODES, reward, ep_reward, avg_reward,
                                                                CACLA_model.sigma, CACLA_model.learning_rate,
                                                                posX, x_vel))

                if CACLA_model.sigma > CACLA_model.sigma_min:
                    CACLA_model.sigma *= CACLA_model.sigma_decay
                else:
                    CACLA_model.sigma = CACLA_model.sigma_min
                if CACLA_model.learning_rate > 0.001:
                    CACLA_model.learning_rate *= 0.99
                else:
                    CACLA_model.learning_rate = 0.001

                break

        ep_reward = 0
        CACLA_model.save(critic_filename, actor_filename)
