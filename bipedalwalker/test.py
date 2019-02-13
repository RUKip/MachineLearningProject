import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
import progressbar
import logging
from bipedalwalker.utils import utils
from bipedalwalker.CACLA import cacla_final
from bipedalwalker.SPG import spg_final


EPISODES = 100
MAX_TIMESTEPS = 2000


def testCACLA(state_size, action_size):
    model = cacla_final.CACLAmodel(state_size, action_size, env.action_space.low, env.action_space.high)
    critic_filename = "CACLA/BipedalWalker-critic.h5"
    actor_filename = "CACLA/BipedalWalker-actor.h5"
    model.load(critic_filename, actor_filename)
    logging.info("Starting testing CACLA: actor_filename: {}, critic_filename: {}\n".format(actor_filename, critic_filename))
    ep_reward_str = "CACLA-ep_reward_arr.csv"
    posX_str = "CACLA-posX_arr.csv"
    t_str = "CACLA-t_arr.csv"
    test(model, ep_reward_str, posX_str, t_str)


def testSPG(state_size, action_size):
    model = spg_final.spgModel(state_size, action_size, env.action_space.low, env.action_space.high)
    critic_filename = "SPG/BipedalWalker-critic.h5"
    actor_filename = "SPG/BipedalWalker-actor.h5"
    model.load(critic_filename, actor_filename)
    logging.info("Starting testing SPG: actor_filename: {}, critic_filename: {}\n".format(actor_filename, critic_filename))
    ep_reward_str = "SPG-ep_reward_arr.csv"
    posX_str = "SPG-posX_arr.csv"
    t_str = "SPG-t_arr.csv"
    test(model, ep_reward_str, posX_str, t_str)


def test(model, ep_reward_str, posX_str, t_str):
    avg_reward = 0
    ep_reward = 0
    ep_reward_arr = []
    posX_arr = []
    t_arr = []
    state = env.reset()
    minmaxValues = "./savedData/minmaxVal.csv"
    model.newNormalizer.setMinMax(state, minmaxValues)

    for e in progressbar.progressbar(range(EPISODES)):
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            env.render()  # Uncomment to see the robot

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
                params = (e + 1, EPISODES, ep_reward, avg_reward, posX, t)
                msg = utils.print_episode(paramList, params)
                logging.debug(msg)
                break

        ep_reward_arr.append(ep_reward)
        posX_arr.append(posX)
        t_arr.append(t)
        ep_reward = 0
    utils.saveToCSV(ep_reward_arr, ep_reward_str)
    utils.saveToCSV(posX_arr, posX_str)
    utils.saveToCSV(t_arr, t_str)
    print("Values saved to CSV")

if __name__ == "__main__":

    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(message)s')
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    testCACLA(state_size, action_size)
    testSPG(state_size, action_size)
