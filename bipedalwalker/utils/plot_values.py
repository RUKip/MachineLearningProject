from bipedalwalker.utils import utils
import sys
import matplotlib.pyplot as plt
import numpy as np

if(len(sys.argv) < 2):
	print("Missing location file to read as argument")
	exit();

reader = utils.ReadInRewards()
reader.read(sys.argv[1])

print(reader.episode_rewards)

plt.plot(reader.episode_rewards)
plt.xlabel('episode rewards')
plt.show()

plt.plot(reader.average_rewards)
plt.xlabel('average rewards')
plt.show()
