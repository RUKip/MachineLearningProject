import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import progressbar
import logging
from bipedalwalker.utils import utils
from bipedalwalker.CACLA import cacla_final


EPISODES = 100
MAX_TIMESTEPS = 2000

if __name__ == "__main__":

    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(message)s')
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    model = cacla_final.CACLAmodel(state_size, action_size, env.action_space.low, env.action_space.high)
    critic_filename = "BipedalWalker-critic.h5"
    actor_filename = "BipedalWalker-actor.h5"

    logging.info("Starting exercise: actor_filename: {}, critic_filename: {}\n".format(actor_filename, critic_filename))

    model.load(critic_filename, actor_filename)

    avg_reward = 0
    ep_reward = 0

    # Train newNormalizer a bit before start normalizing
    state = env.reset()
    model.newNormalizer.setMinMax(state)
    for i in range(50):
        state = env.reset()
        done = False
        while not done:
            model.newNormalizer.update(state)
            next_state, reward, done, _ = env.step(env.action_space.sample())
            state = next_state


    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            env.render()

            # Updating running mean and deviation for the state vector.
            model.newNormalizer.update(state)

            # Performing an action.
            norm_state = model.newNormalizer.normalize(state)
            actions = model.actor_model.predict(np.reshape(norm_state, (1, state_size)))
            next_state, reward, done, _ = env.step(actions[0])

            state = next_state

            posX = env.env.hull.position.x
            avg_reward += reward
            ep_reward += reward

            if done:
                # Output variables to log.
                paramList = ["ep_reward",
                             "avg_reward",
                             "posX",
                             "t"]
                params = (e+1, EPISODES, ep_reward, avg_reward, posX, t)
                msg = utils.print_episode(paramList, params)
                logging.debug(msg)
                break

        ep_reward = 0
