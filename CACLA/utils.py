import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=2000)

    def add(self, transition):
        # Transition is tuple of (state, action, reward, tempDiff, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, tempDiff, next_state, done = [], [], [], [], [], []

        for i in indexes:
            s, a, r, t, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            tempDiff.append(np.array(t, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(tempDiff), np.array(next_state), np.array(done)

