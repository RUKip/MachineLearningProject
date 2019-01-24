import numpy as np
from collections import deque
import math
import pandas as pd
import os


class Memory:
    def __init__(self, max_len=2000):
        self.states = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.next_states = deque(maxlen=max_len)
        self.done = deque(maxlen=max_len)

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)

    def sample(self, batch_size=32):
        indexes = np.random.randint(0, len(self.states), size=batch_size)
        state_batch = np.asarray(self.states)[indexes]
        action_batch = np.asarray(self.actions)[indexes]
        reward_batch = np.asarray(self.rewards)[indexes]
        nex_state_batch = np.asarray(self.next_states)[indexes]
        done_batch = np.asarray(self.done)[indexes]
        return state_batch, action_batch, reward_batch, nex_state_batch, done_batch

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.done.clear()


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    #
    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def __init__(self, count=0, mean=0.0, M2=0.0):
        self.count = count
        self.mean = mean
        self.M2 = M2

    def __len__(self):
        return self.count

    def update(self, newValue):
        self.count += 1
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        self.M2 += delta * delta2

    # retrieve the mean, variance and sample variance from an aggregate
    def getMeanVar(self):
        (self.mean, variance) = (self.mean, self.M2/self.count)
        # if self.count < 2: # TODO: esto es necesario?? asi me ahorro limitarlo en el codigo..
        #     return float('nan')
        # else:
        return self.mean, variance

class obsRunningMeanStd:
    def __init__(self, size=0):
        self.obsRMS = []
        self.state_size = size
        for i in range(self.state_size):
            rms = RunningMeanStd()
            self.obsRMS.append(rms)

    def __len__(self):
        return len(self.obsRMS)

    def getSize(self):
        return len(self.obsRMS)

    def update(self, state_arr, state_size):
        for i in range(state_size):
            self.obsRMS[i].update(state_arr[i])

    def getMeanStd(self):
        mean_var = []
        for i in range(self.state_size):
            mean_var.append(self.obsRMS[i].getMeanVar())
        return mean_var

    def print(self):
        print("RunningMeanStd:")
        print(np.asarray(self.getMeanStd()))


def normalize(X, obsRMS):
    if obsRMS is None:
        return X
    mean_var = obsRMS.getMeanStd()
    normVal = []
    for i, x in enumerate(X):
        if mean_var[i][1] == 0:  # if variance equals 0
            norm_val = x
        else:
            norm_val = (x - mean_var[i][0]) / mean_var[i][1]
        normVal.append(norm_val)
    return normVal


def normalize_batch(X, obsRMS):
    if obsRMS is None:
        return X
    mean_var = np.asarray(obsRMS.getMeanStd())
    normVal = np.zeros((len(X), len(mean_var)))
    for i, x in enumerate(X):
        for j, val in enumerate(x):
            if mean_var[j][1] == 0:
                norm_val = val
            else:
                norm_val = (val - mean_var[j][0]) / mean_var[j][1]
            normVal[i][j] = norm_val
    return normVal


def denormalize(X, obsRMS):
    if obsRMS is None:
        return X
    mean_var = obsRMS.getMeanStd()
    denomVal = []
    for i, x in enumerate(X):
        denorm_val = x * mean_var[i][1] + mean_var[i][0]
        denomVal.append(denorm_val)
    return denomVal


class newNormalizer:
    def __init__(self, size=0):
        self.maxValues = np.zeros((1, size))
        self.minValues = np.zeros((1, size))

    def setMinMax(self, state, filename):
        self.maxValues = np.copy(state)
        self.minValues = np.copy(state)
        if os.path.isfile(filename):
            values = pd.read_csv(filename)
            self.minValues = values.values[0]
            self.maxValues = values.values[1]

    def saveMinMax(self, filename):
        values = pd.DataFrame(np.vstack((self.minValues, self.maxValues)))
        values.to_csv(filename, index=False)

    def update(self, values):
        compMax = np.greater(values, self.maxValues)
        idxMax = np.where(compMax==True)
        np.put(self.maxValues, idxMax, values[idxMax])
        compMin = np.less(values, self.minValues)
        idxMin = np.where(compMin==True)
        np.put(self.minValues, idxMin, values[idxMin])

    def normalize(self, values):
        normVal = 2*(values - self.minValues) / (self.maxValues - self.minValues) - 1
        return normVal

    def normalize_batch(self, X):
        normVal = np.zeros(np.shape(X))
        for i, values in enumerate(X):
            aux = 2*(values - self.minValues) / (self.maxValues - self.minValues) - 1
            normVal[i] = aux
        return normVal


def print_hyperparam(paramList, hyperparams):
    message = "    HYPERPARAMETERS    \n" \
              "-----------------------\n"
    for i, param in enumerate(paramList):
        message += param+": {}\n".format(hyperparams[i])
    message += "-----------------------\n"
    return message


def print_episode(paramList, params):
    message = "    Episode: {}/{}\n" \
              "-----------------------\n".format(params[0], params[1])
    for i, param in enumerate(paramList, start=2):
        message += param+": {}\n".format(params[i])
    message += "-----------------------\n"
    return message

