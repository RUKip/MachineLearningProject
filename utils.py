import numpy as np


class Buffer:
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, v):
        self.buffer.append(v)

    def get_batch(self, idxs):
        return np.asarray(self.buffer)[idxs]


class Memory:
    def __init__(self):
        self.states = Buffer()
        self.actions = Buffer()
        self.rewards = Buffer()
        self.next_states = Buffer()
        self.done = Buffer()

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.states.buffer), size=batch_size)
        state_batch = self.states.get_batch(indexes)
        action_batch = self.actions.get_batch(indexes)
        reward_batch = self.rewards.get_batch(indexes)
        nex_state_batch = self.next_states.get_batch(indexes)
        done_batch = self.done.get_batch(indexes)
        return state_batch, action_batch, reward_batch, nex_state_batch, done_batch

    def clear(self):
        self.states.buffer.clear()
        self.actions.buffer.clear()
        self.rewards.buffer.clear()
        self.next_states.buffer.clear()
        self.done.buffer.clear()


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

    def update(self, newValue):
        self.count += 1
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        self.M2 += delta * delta2

    # retrieve the mean, variance and sample variance from an aggregate
    def getMeanVar(self):
        (self.mean, variance) = (self.mean, self.M2/self.count)
        if self.count < 2:
            return float('nan')
        else:
            return self.mean, variance

class obsRunningMeanStd:
    def __init__(self):
        self.obsRMS = []
        self.state_size = 0

    def setSize(self, size=0):
        for i in range(size):
            rms = RunningMeanStd()
            self.obsRMS.append(rms)
            self.state_size += 1

    def getSize(self):
        return len(self.obsRMS)

    def update(self, state_arr, state_size):
        for i in range(state_size):
            self.obsRMS[i].update(state_arr[i])

    def getMeanStd(self):
        meanVar = []
        for i in range(self.state_size):
            meanVar.append(self.obsRMS[i].getMeanVar())
        return meanVar



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

