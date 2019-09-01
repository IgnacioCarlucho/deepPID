from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise   
from ddpg import DDPG
from robots import pioneer_pi
# if in windows
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import str2bool
import argparse

ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE =  0.0001
# Soft target update param
TAU = 0.001



parser = argparse.ArgumentParser('deepid')
parser.add_argument('--gpu', type=str, choices=['gpu', 'cpu'], default='cpu')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--sim', type=int, default=100)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=True, help="Activate train mode.")
parser.add_argument("--img", type=str2bool, nargs='?', const=True, default=True, help="Activate save image.")
parser.add_argument("--reset", type=str2bool, nargs='?', const=True, default=True, help="resets after each episode.")
parser.add_argument("--simulation", type=str2bool, nargs='?', const=True, default=True, help="Is this run on gazebo or in the real vehicel.")
parser.add_argument("--load", type=str2bool, nargs='?', const=True, default=True, help="Loads policy.")
parser.add_argument('--epsilon_decay', type=float, default=0.0002)
parser.add_argument('--psi', type=float, default=1.)
parser.add_argument('--pid', type=str, choices=['pid', 'pi'], default='pi')
parser.add_argument('--max_action', type=float, default=50.)
parser.add_argument('--min_action', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=51234)
parser.add_argument('--save_mod', type=int, default=500)
args = parser.parse_args()    




DEVICE = args.gpu
max_action = args.max_action
min_action = args.min_action
epochs = args.epochs
epsilon = args.epsilon
min_epsilon = 0.1
decay_rate =  args.epsilon_decay # 0.9/ num of episodes until min_epsilon (0.1)
BUFFER_SIZE = 100000
RANDOM_SEED = args.seed
MINIBATCH_SIZE = 64
PSI = args.psi
save_mod = args.save_mod



with tf.Session() as sess:
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    
    state_dim = 20 + 2 +4
    action_dim = 4
    robot = pioneer_pi("pioneer", n_actions=action_dim,save_image=False, dt=0.1, Teval = 1, simulation=args.simulation,reset=args.reset, ep_length = args.sim)  
    low = DDPG(sess, state_dim, action_dim, max_action, min_action, ACTOR_LEARNING_RATE,CRITIC_LEARNING_RATE, TAU, RANDOM_SEED, device=DEVICE)
    sess.run(tf.global_variables_initializer())
    if args.load:
        low.load()
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    ruido = OUNoise(action_dim, mu = 0.0)
    total_ep_reward = np.zeros(epochs)
    for i in range(epochs):
        
        # define goal 
        # option 1: sample from especific velocities 
        sample_vx = np.array([0.,.1,.2,.3,.4,.5,.15,.25,.35,.45,0.,-.1,-.2,-.3,-.4,-.5,-.15,-.25,-.35,-.45])
        sample_wz = np.array([0.,.1,.2,.3,.15,.25,0.,-.1,-.2,-.3,-.15,-.25,])
        velocity_req = np.array([np.random.choice(sample_vx), np.random.choice(sample_wz)]) 
        # option 2: all random (problem is you get velocities like 0.08 which make no sense)
        # velocity_req = np.array([np.random.rand(1)[0] -.5, 0.4*np.random.rand(1)[0] -.2]) 
        # Option 3: Define a specific velocity
        # velocity_req = np.array([0.21,0.11]) 

        # Initial state
        action = np.zeros(action_dim)
        position, velocities, u = robot.reset() # velocities = np.zeros((10,6))
        robot.get_set_point(velocity_req)
        velocity_error = np.subtract(velocities[9], velocity_req)
        state = np.reshape(np.vstack((velocity_error,velocities,np.reshape(action,(2,2)))),(state_dim,))


        done = False
        epsilon -= decay_rate
        epsilon = np.maximum(min_epsilon,epsilon)
        episode_r = 0.
        step = 0
        action_buffer = np.zeros(((args.sim+1),action_dim))
        velocity_buffer = np.zeros(((args.sim+1)*10,2))
        r_vector = np.zeros(args.sim+1)

        while (not done):
            
            action = low.predict_action(np.reshape(state,(1,state_dim)))[0]
            action = np.clip(action,min_action,max_action)
            action = action + max(4.*epsilon,0)*ruido.noise()
            action = np.clip(action,min_action,max_action)
           
            
            new_position, new_velocities, u = robot.run(action)   
            # this should all go to the robot.run 
            velocity_error = np.subtract(new_velocities[9], velocity_req)
            next_state = np.reshape(np.vstack((new_velocities,velocity_error,np.reshape(action,(2,2)))),(state_dim,))
            reward, done = robot.get_reward_v2(new_velocities,velocity_req,step)

            action_buffer[step] = action

            
            j = step*10
            velocity_buffer[j] = new_velocities[0]
            velocity_buffer[j+1] = new_velocities[1]
            velocity_buffer[j+2] = new_velocities[2]
            velocity_buffer[j+3] = new_velocities[3]
            velocity_buffer[j+4] = new_velocities[4]
            velocity_buffer[j+5] = new_velocities[5]
            velocity_buffer[j+6] = new_velocities[6]
            velocity_buffer[j+7] = new_velocities[7]
            velocity_buffer[j+8] = new_velocities[8]
            velocity_buffer[j+9] = new_velocities[9]
            r_vector[step] = reward
            

            replay_buffer.add(np.reshape(state, (state_dim,)), np.reshape(action, (action_dim,)), reward,
                                  done, np.reshape(next_state, (state_dim,)))
            state = next_state
            step += 1
            episode_r = episode_r + reward
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                #low.train(s_batch, a_batch, r_batch, t_batch, s2_batch,MINIBATCH_SIZE)
                low.test_gradient(s_batch, a_batch, r_batch, t_batch, s2_batch,MINIBATCH_SIZE)
        print(i, step, 'last r', round(reward,3), 'epsilon', round(epsilon,3),'episode reward','**',round(episode_r,3),'**' )                
        print('req', velocity_req, 'last v', np.round(new_velocities[9],3))
        ruido.reset()
        robot.reset()
        total_ep_reward[i] = episode_r
        # np.save('velocity-buff.np',velocity_buffer)
        if i%save_mod==0:
            np.save('figs/velocity_buffer' + str(i), velocity_buffer)
            np.save('figs/action_buffer' + str(i), action_buffer)
            
            plt.plot(action_buffer)
            plt.legend(['kp1','ki1','kp2','ki2'])
            plt.savefig('figs/action' + str(i) + '.png')
            # plt.savefig('action.png')
            # plt.show()
            plt.clf()
            
            plt.plot(velocity_buffer)
            plt.legend(['vx = ' + str(np.round(robot.set_point[0],2)) ,'wz = '+ str(np.round(robot.set_point[1],2))])
            plt.savefig('figs/velocity' + str(i) + '.png')
            # plt.savefig('velocity.png')
            # plt.show()
            plt.clf()
            
            np.save('velocity-buff',velocity_buffer)
            np.save('action_buf',action_buffer)
            np.save('r_vector',r_vector)

    print('total_av_reward' , np.mean(total_ep_reward))

    low.save()