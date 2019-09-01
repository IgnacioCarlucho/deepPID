import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import tf.transformations
from plotter import plotter
from pid import pi_controller_pioneer, pid_controller_pioneer
from std_srvs.srv import Empty
#TODO nothing is stored here yet. So no plots will be abailable to the user. 


ALPHA = 0.1


class pioneer_pi():

    def __init__(self, name = "pioneer", n_actions=4, controller_type='pi',save_image=False,  dt = 0.1, Teval = 1., simulation = True,reset=False, ep_length=100):


        self.dt = dt
        self.Teval = Teval
        self.execution = np.divide(self.Teval,self.dt).astype(int)
        self.pos = np.zeros(2)
        self.vel_v = np.zeros(3)
        self.vel_w = np.zeros(3)
        self.vel = np.zeros(2)
        self.velocities = np.zeros((self.execution,2))
        self.reading = False
        self.euler = np.zeros(3)
        self.Quater = np.zeros(4)
        self.done = False
        self.error = np.zeros((3,2))
        self.u0 = np.zeros(2)
        self.u = np.zeros(2)
        self.n_actions = n_actions
        self.action = np.zeros(self.n_actions)
        self.set_point = np.zeros(2)
        self.ep_length = ep_length
        self.node = rospy.init_node('DQPID', anonymous=False)
        if simulation: 
            self.Publisher =  rospy.Publisher("/sim_p3at/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/sim_p3at/odom", Odometry, self.callback_pose, queue_size=1)
        else: 
            self.Publisher =  rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/RosAria/pose", Odometry, self.callback_pose, queue_size=1)    

        self.rate = rospy.Rate(10.) # 10hz
        self.msg = Twist()
        self.action_vx = np.zeros(2)
        self.action_wz = np.zeros(2)
        self.reward = -1.
        self.temporal_vx = np.zeros(self.execution)
        self.temporal_wz = np.zeros(self.execution)
        self.controller_type = controller_type

        if self.controller_type == 'pi':
            self.controller = pi_controller_pioneer(set_point=self.set_point, dt=self.dt)
        else: 
            self.controller = pid_controller_pioneer(set_point=self.set_point, dt=self.dt)
        # to plot
        # to plot
        self.save_image = save_image
        self.acc = np.zeros(1)
        self.thruster = np.zeros(1)
        if self.save_image: 
            self.plots_counter = 0
            self.plot = plotter()
            self.plot.update(self.vel,self.action,self.pos,self.u)
        self.time = 0. 
        self.reset_simulation= reset
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)           

    def reset(self):
        self.reading = False
        self.u = np.zeros(2)
        self.reward = 0.
        self.done = False
        self.velocities = np.zeros((10,2))
        for j in range(50):
            self.msg.linear.x = 0.
            self.msg.angular.z = 0.
            self.Publisher.publish(self.msg)
            self.rate.sleep()
        if self.reset_simulation:
            self.reset_world()
        #while not self.reading:
        #    self.rate.sleep(0.1)
        if self.save_image: 
            self.plot.reset()
            self.plot.update(self.vel,self.action,self.pos,self.thruster)
            self.plots_counter = self.plots_counter + 1

        return self.pos, self.velocities, self.u 

   

    def run(self, actions):

        self.controller.update_action(actions)
        # to plot
        self.action = actions
        
        for i in range(self.execution):
            
            self.u = self.controller.update(self.vel)
            self.u = np.clip(self.u, -0.8, 0.8)
            self.u0 = self.u
            # send msg
            self.msg.linear.x = self.u[0]
            self.msg.angular.z = self.u[1]
            self.Publisher.publish(self.msg)

                      
            self.time = self.time + self.dt
            self.velocities[i] = self.vel
            # to keep sampling rate
            self.rate.sleep()
            
            
            # plot
            if self.save_image: self.plot.update(self.vel,self.action,self.pos,self.thruster)

        return self.pos, self.velocities, self.u

    def get_set_point(self, set_point):
        self.set_point = set_point
        self.controller.set_point = set_point


    def get_gaussian_reward(self, state, set_point, step):

        beta = self.n_actions*0.025  
        self.reward = -1.5  + beta
        
        v0 = [_[0] for _ in self.velocities] 
        v1 = [_[1] for _ in self.velocities] 
        v0 = np.mean(v0) + np.std(v0)
        v1 = np.mean(v1) + np.std(v1)
        v_weight = [1.,1.5]
        velocity = np.array([v0,v1])
        e_weight = 0.9*np.array([0.02, 0.02]) # np.array([0.02,0.02,0.02,0.02,0.02,0.02])*0.25
        for x in range(len(set_point)):
            diff = (velocity[x] - set_point[x])**2
            exponent = (diff/(e_weight[x]**2))
            self.reward = self.reward + v_weight[x]*np.exp(-0.5*exponent)

          
        # self.reward = self.reward - beta*np.sum(np.abs(action))
       
        if np.abs(self.vel[0])>1.:
            self.done = True
        elif np.abs(self.vel[1])>1.:
            self.done = True
        elif np.abs(self.euler[0]) > 0.5:
            self.done = True
        elif np.abs(self.euler[1]) > 0.5:
            self.done = True
        else: 
            self.done = False

        if step >= self.ep_length:
            self.done = True


        return self.reward, self.done

    def get_reward_v2(self, state, velocity_req, step):
        velocity = self.velocities[9]
        self.reward = -1.
        # x, y, z, roll, pitch, yaw
        v_weight = [1.,1.]
        e_weight = 0.4*np.array([0.04, 0.04])
        for x in range(len(velocity_req)):
            diff = (velocity[x] - velocity_req[x])**2
            exponent = (diff/(e_weight[x]**2))
            self.reward = self.reward  + v_weight[x]*np.exp(-0.5*exponent)

        if np.abs(self.vel[0])>1.:
            self.reward = -10.
            self.done = True
        elif np.abs(self.vel[1])>1.:
            self.reward = -10.
            self.done = True
        elif np.abs(self.euler[0]) > 0.5:
            self.reward = -1.
            self.done = True
        elif np.abs(self.euler[1]) > 0.5:
            self.reward = -1.
            self.done = True
        else: 
            self.done = False

        if step >= self.ep_length:
            self.done = True

        return self.reward, self.done



    def plot_if(self):
        if self.save_image: 
            self.plot_imgs()
            
    def plot_imgs(self):
        self.plot.plot(self.plots_counter)

    def wrapToPi(self, angles):
        
        if angles > np.pi:
            angles = angles - 2*np.pi
        elif angles < -np.pi:
            angles = angles + 2*np.pi
        return angles 


    def callback_pose(self, msg_odometry):
        self.reading = True
        x = msg_odometry.pose.pose.position.x
        y = msg_odometry.pose.pose.position.y
        z = msg_odometry.pose.pose.position.z
        self.pos = np.array([x, y])

        vx = msg_odometry.twist.twist.linear.x
        vy = msg_odometry.twist.twist.linear.y
        vz = msg_odometry.twist.twist.linear.z
        self.vel_v = np.array([vx, vy, vz])
        wx = msg_odometry.twist.twist.angular.x
        wy = msg_odometry.twist.twist.angular.y
        wz = msg_odometry.twist.twist.angular.z
        self.vel_w = np.array([wx, wy, wz])

        Qx = msg_odometry.pose.pose.orientation.x
        Qy = msg_odometry.pose.pose.orientation.y
        Qz = msg_odometry.pose.pose.orientation.z
        Qw = msg_odometry.pose.pose.orientation.w
        #z y x representation
        #Quater=[Qz,Qy,Qx,Qw];
        #Quater = np.array([Qw,Qx,Qy,Qz]) # este es el que eestaba usando
        self.Quater = np.array([Qx,Qy,Qz, Qw])
        #z y x representation of quaternions
        euler_original = tf.transformations.euler_from_quaternion(self.Quater) #[rad]
        
        self.euler = [ self.wrapToPi(_) for _ in euler_original] 
        # print(np.round(self.euler,3))
        #self.velocity = np.array([vx, wz])
        a = 0.9*self.vel[0] + 0.1*vx
        b = 0.9*self.vel[1] + 0.1*wz
        # print('a',a)
        self.vel = np.array([a, b])
        # self.vel = np.array([vx, wz])


    def stop(self):
        self.msg.linear.x = 0.
        self.msg.angular.z = 0.
        self.Publisher.publish(self.msg)