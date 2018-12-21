import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import progressbar
import logging
import utils


EPISODES = 10000
MAX_TIMESTEPS = 2000


class spgModel:
    def __init__(self, state_size, action_size, action_min, action_max):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = utils.Memory()
        self.action_space_low = action_min
        self.action_space_high = action_max
        # HyperParams
        self.gamma = 0.99  # Discount rate
        self.learning_rate = 0.001
        self.sigma = 1.0
        self.sigma_min = 0.1
        self.sigma_decay = 0.99
        self.batch_size = 32
        self.n_iter = 100
        self.n_sampled_actions = 3
        # Build NN
        self.critic_model = self._build_critic_model()
        self.actor_model = self._build_actor_model()

    def get_hyper_params(self):
        params = (self.gamma, self.learning_rate, self.sigma, self.sigma_min,
                 self.sigma_decay, self.batch_size, self.n_iter, self.n_sampled_actions)
        return params

    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size+self.action_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def exploration(self, state, action_space):  # SPG-OffGE - offline Gaussian exploration
        act_values = self.actor_model.predict(state)
        actions = act_values + np.random.normal(0, self.sigma, size=action_space)
        actions = actions.clip(self.action_space_low, self.action_space_high)
        return actions[0]

    def train_critic(self):
        for _ in range(self.n_iter):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            st_at = np.hstack((next_state, self.actor_model.predict(next_state)))  # state-action pair
            target_critic = np.reshape(reward, [len(reward), 1]) \
                            + self.gamma*self.critic_model.predict(st_at)

            self.critic_model.fit(st_at, target_critic, epochs=1, verbose=0)

    def train_actor(self):
        for _ in range(self.n_iter):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            target_arr = []
            state_arr = []
            for i in range(self.batch_size):
                s, a = np.reshape(state[i], (1, self.state_size)), np.reshape(action[i], (1, self.action_size))
                pi_s = self.actor_model.predict(s)
                best = pi_s
                if self.get_Q(s, a) > self.get_Q(s, pi_s):
                    best = a
                for _ in range(self.n_sampled_actions):
                    sampled = self.apply_gaussian(best)
                    if self.get_Q(s, sampled) > self.get_Q(s, best):
                        best = sampled
                if self.get_Q(s, best) > self.get_Q(s, pi_s):
                    # TODO: if doesn't work, first uncomment this and update 1 by 1
                    # target = best
                    # self.actor_model.fit(s, target, batch_size=self.batch_size, epochs=1, verbose=0)
                    state_arr.append(s)
                    target_arr.append(best)
            len_arrs = len(state_arr)
            self.actor_model.fit(np.reshape(state_arr, (len_arrs, self.state_size)),
                                 np.reshape(target_arr, (len_arrs, self.action_size)), epochs=1, verbose=0)

    def apply_gaussian(self, best_action):
        best_action = best_action + np.random.normal(0, self.sigma, size=self.action_size)
        return best_action.clip(self.action_space_low, self.action_space_high)

    def get_Q(self, state, actions):
        return self.critic_model.predict(np.hstack((state, actions)))

    def save(self, critic_filename, actor_filename):
        self.critic_model.save_weights(critic_filename)
        self.actor_model.save_weights(actor_filename)

    def load(self, critic_filename, actor_filename):
        self.critic_model.load_weights(critic_filename)
        self.actor_model.load_weights(actor_filename)


if __name__ == "__main__":

    # logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(message)s')
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    model = spgModel(state_size, action_size, env.action_space.low, env.action_space.high)

    critic_filename = "BipedalWalker-critic.h5"
    actor_filename = "BipedalWalker-actor.h5"

    logging.info("Starting exercise: actor_filename: {}, critic_filename: {}\n".format(actor_filename, critic_filename))
    paramList = ["gamma", "learning_rate", "sigma", "sigma_min", "sigma_decay", "batch_size",
                 "n_iter", "n_SampledActions"]
    msg = utils.print_hyperparam(paramList, model.get_hyper_params())
    logging.info(msg)

    # model.load(critic_filename, actor_filename)

    avg_reward = 0
    ep_reward = 0

    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        finishLine = False
        for t in range(MAX_TIMESTEPS):
            # env.render()
            actions = model.exploration(np.reshape(state, (1, state_size)), action_size)
            next_state, reward, done, _ = env.step(actions)
            posX = env.env.hull.position.x
            x_vel = state[2]
            finishLine = ((reward > 200.0) or (posX > 100))
            if finishLine:
                logging.warning("Hallelujah! reward: {}, posX: {}".format(reward, posX))

            model.replay_buffer.append(state, actions, reward, next_state, done)
            state = next_state
            avg_reward += reward
            ep_reward += reward

            if done or (t == (MAX_TIMESTEPS-1)):
                model.train_actor()
                model.train_critic()
                paramList = ["latest_r", "ep_reward", "avg_reward", "sigma",
                             "l_rate", "posX", "t"]
                params = (e+1, EPISODES, reward, ep_reward, avg_reward, model.sigma, model.learning_rate, posX, t)
                msg = utils.print_episode(paramList, params)
                logging.debug(msg)

                if model.sigma > model.sigma_min:
                    model.sigma *= model.sigma_decay
                else:
                    model.sigma = model.sigma_min
                if model.learning_rate > 0.001:
                    model.learning_rate *= 0.99
                else:
                    model.learning_rate = 0.001

                break

        ep_reward = 0
        model.save(critic_filename, actor_filename)
