# Deep PID 

Implementation of the Deep PID controller. With inverted gradients.
Both the gym environtm
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

## How to run

Have in mind that each of the algorithms use a random seed that allows for reproducibility of the algorithms. If you want to change this you need to change the seeds (in future implementations this will be passed as a parameter).  
To run the Deep pid with the pioneer with the inverted gradient

```
python main.py 
```


with arguments you can run it on the real robot: 

```
python main.py --simulation False
```
## DDPG demos

All the implemented algoritms can be runned as demos with different gym environments. All the implementations are within the classes files. So for instance running: 
```
python ddpg.py 
```
will run the ddpg algorithm using the inverted gradients for the gym's pendulum example. While td3.py runs the same for the td3 algoritm.