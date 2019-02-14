# Comparing PPO and CACLA machine learning algorithm

## Installation
Create virtualenv in base repo directory:
```
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
source myenv/bin/activate
export PYTHONPATH="$PYTHONPATH:/<path_to_main_folder>"
```

## Training 
###CACLA
```
cd bipedalwalker/CACLA/
python cacla_final.py
```
###SPG
```
cd bipedalwalker/SPG/
python spg_final.py
```
## Testing
Copy trained weights and minmaxvalues, then run test file.
###CACLA
```
cp gcloud-data/CACLA/BipedalWalker-* bipedalwalker/CACLA/
cp gcloud-data/CACLA/minmaxVal.csv bipedalwalker/savedData/
python test.py
```
###SPG
```
cp gcloud-data/SPG/BipedalWalker-* bipedalwalker/SPG/
cp gcloud-data/SPG/minmaxVal.csv bipedalwalker/savedData/
python test.py
```
Note: Min-max values used for normalization are updated dynamically when training an algorithm. Currently, min-max values computed in both algorithms are slightly different because they were trained using different machines. To see the optimal test performance, load the specific min-max values.    

## Useful Links

```
BipedalWalker => https://github.com/openai/gym/wiki/BipedalWalker-v2
https://www.cs.ubc.ca/~gberseth/blog/demystifying-the-many-deep-reinforcement-learning-algorithms.html
SPG => https://arxiv.org/abs/1809.05763
CACLA => https://hadovanhasselt.files.wordpress.com/2015/12/reinforcement_learning_in_continuous_action_spaces.pdf
CACLA => http://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/CACLA_Discrete_Problems.pdf
KERAS => https://keras.io/#getting-started-30-seconds-to-keras
CARTPOLE => https://keon.io/deep-q-learning/#Replay
```
