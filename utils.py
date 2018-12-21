import numpy as np


class Buffer:
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return np.size(self.buffer)

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
