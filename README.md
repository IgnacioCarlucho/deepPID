# Deep PID 

This is the implementation of the inverted deep PID algorithm found in the article: 

**"An adaptive deep reinforcement learning approach for
MIMO PID control of mobile robots"**    

submitted to ***ISA Transactions***, currently *under review*.  


### This repo contains: 

- deep PID controller using tensorflow
- Vanilla DDPG
- DDPG with inverted gradients
- An implementation of the TD3 algorithm


### Requirements: 

- tensorflow > v1.0
- numpy
- scipy
- python 2.7
- [pioneer simulator](https://github.com/IgnacioCarlucho/amr-ros-config)
- ROS kinetic 
- Gazebo > 7 

## How to run

Have in mind that the Pioneer simulator speeds up the simulation time, so you will need a machine that is able to run it. 
To run the Deep PID with the pioneer with the inverted gradient:

```
python main.py --alg iddpg --gpu gpu --epochs 1000
```

The alg argument selects with which algoritm to train the agent, iddpg or it3.   
Results on the real robot can be obtained such as: 

![alt text](https://github.com/IgnacioCarlucho/deepPID/blob/master/figs/DPID_spchangevelocity.png)


The algorithm periodically saves the actions and velocities obtained every x number of epochs, this number can be changed by using the argument *save_mod*. The neural networks are saved automatically after the max number of epochs is achieved. 

## DDPG demos

All the implemented algoritms can be runned as demos with different gym environments. All the implementations are within the classes files. So for instance running: 
```
python ddpg.py 
```
will run the ddpg algorithm using the inverted gradients for the gym's pendulum example. While td3.py runs the same for the td3 algoritm.

## Authors

Ignacio Carlucho, Mariano De Paula, Gerardo G. Acosta


