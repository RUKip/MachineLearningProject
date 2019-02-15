import numpy as np
from bipedalwalker.utils import utils
import matplotlib.pyplot as plt


# # CACLA training
# ep_reward_str = "../gcloud-data/CACLA/ep_reward_arr.csv"
# posX_str = "../gcloud-data/CACLA/posX_arr.csv"

# SPG training
ep_reward_str = "../gcloud-data/SPG/ep_reward_arr.csv"
avg_reward_str = "../gcloud-data/SPG/avg_reward_arr.csv"
posX_str = "../gcloud-data/SPG/posX_arr.csv"

# # CACLA testing
# ep_reward_str = "CACLA-ep_reward_arr.csv"
# avg_reward_str = "CACLA-avg_reward_arr.csv"
# posX_str = "CACLA-posX_arr.csv"
# t_str = "CACLA-t_arr.csv"

# # SPG testing
# ep_reward_str = "SPG-ep_reward_arr.csv"
# avg_reward_str = "SPG-avg_reward_arr.csv"
# posX_str = "SPG-posX_arr.csv"
# t_str = "SPG-t_arr.csv"

ep_reward_arr = np.asarray(utils.loadFromCSV(ep_reward_str))
posX_arr = np.asarray(utils.loadFromCSV(posX_str))
# t_arr = np.asarray(utils.loadFromCSV(t_str))
avg_reward_arr = np.cumsum(ep_reward_arr)
utils.saveToCSV(avg_reward_arr, avg_reward_str)

# # plt.plot(ep_reward_arr)
# plt.bar(np.arange(np.size(ep_reward_arr)), np.ravel(ep_reward_arr), width=0.2)
# plt.title("CACLA Training Episode Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()
#
# # plt.plot(avg_reward_arr)
# plt.bar(np.arange(np.size(avg_reward_arr)), np.ravel(avg_reward_arr), width=0.3)
# plt.title("CACLA Training Average Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()
#
# # plt.plot(posX_arr)
# plt.bar(np.arange(np.size(posX_arr)), np.ravel(posX_arr), width=0.4)
# plt.title("CACLA Training Episode Last PosX")
# plt.xlabel("Episode")
# plt.ylabel("Position")
# plt.show()

# # plt.plot(t_arr)
# plt.bar(np.arange(np.size(t_arr)), np.ravel(t_arr))
# plt.title("N timesteps before finish")
# plt.show()