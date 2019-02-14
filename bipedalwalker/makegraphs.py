import numpy as np
from bipedalwalker.utils import utils
import matplotlib.pyplot as plt


# CACLA training
ep_reward_str = "../gcloud-data/CACLA/ep_reward_arr.csv"
posX_str = "../gcloud-data/CACLA/posX_arr.csv"

# # SPG training
# ep_reward_str = "SPG/ep_reward_arr.csv"
# posX_str = "SPG/posX_arr.csv"
# t_str = "SPG/t_arr.csv"

# # CACLA testing
# ep_reward_str = "CACLA-ep_reward_arr.csv"
# posX_str = "CACLA-posX_arr.csv"
# t_str = "CACLA-t_arr.csv"
#
# # SPG testing
# ep_reward_str = "SPG-ep_reward_arr.csv"
# posX_str = "SPG-posX_arr.csv"
# t_str = "SPG-t_arr.csv"

ep_reward_arr = np.asarray(utils.loadFromCSV(ep_reward_str))
posX_arr = np.asarray(utils.loadFromCSV(posX_str))
# t_arr = np.asarray(utils.loadFromCSV(t_str))
avg_reward_arr = np.cumsum(ep_reward_arr)

plt.plot(ep_reward_arr)
plt.title("Episode reward")
plt.show()

plt.plot(avg_reward_arr)
plt.title("Average reward")
plt.show()

plt.plot(posX_arr)
plt.title("Episode last posX")
plt.show()

# plt.plot(t_arr)
# plt.title("N timesteps before finish")
# plt.show()