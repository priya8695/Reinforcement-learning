# RL Assignment 2

This programm solves the Cartpole problem, using the OpenAI environment.


## Installation

Required Python-Packages: 
```
gym
keras
tensorflow
gym[classic_control]
```
    
## Documentation

To reproduce the results, you just have to execute the main.py Python file insite the folder with following arguments:

``` --exp_type ```  defines the number of experiment (1-6), select value from 1-6 as per the following experiment:

Experiment 1: Just training with DQN with optimal parameters

Experiment 2: comparison of DQN with DQN-ER, DQN-TN, DQN-ER-TN

Experiment 3: experiment with different policy

Experiment 4: experiment with different learning rates

Experiment 5: experiment with gamma

Experiment 6: experiment with different architecture



``` --ER=True ```        Update the Q-value for the chosen action with replay buffer (Default: False)

``` --TN =True```        Model with target network(Default: False)

``` --episodes ```  amount of episodes to use (Default 500)
