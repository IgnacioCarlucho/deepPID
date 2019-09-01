import numpy as np
import pylab as py
from collections import deque
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def create_circle(x,y,r):

    ang = np.linspace(0., 2*np.pi, 1000)
    xp = r*np.cos(ang)
    yp = r*np.sin(ang)
    x_tot = x + xp
    y_tot = y + yp
    return x_tot, y_tot


class plotter(object):

    def __init__(self):

        self.actions = deque()
        self.velocity = deque()
        self.position = deque()
        self.accelerations = deque()
        self.i = []
        self.set_point = []

        self.done = deque()
        self.wp = deque()

    def update(self,velocities,action,accelerations,position):
        self.actions.append(action)
        self.velocity.append(velocities)
        self.accelerations.append(accelerations)
        self.position.append(position)

    def update_goal(self,done,wp):
        self.done.append(done)
        self.wp.append(wp)


    def plot(self, i):

        
        py.plot(self.position)
        py.xlabel('Time (time steps)')
        py.ylabel('Position')
        py.title('DDPG for AUV Control')
        py.legend(('x','y','z','roll', 'pitch', 'yaw' ))
        #py.axis([0, simulation_lenght, -0.5, 0.5])
        py.savefig('Positions' + str(i) + '.png')
        py.show()
        
        py.plot(self.velocity)
        py.xlabel('Time (time steps)')
        py.ylabel('Velocities')
        py.title('DDPG for AUV Control')
        py.legend(('1', '2','3', '4' ,'5','6'  ))
        #py.axis([0, simulation_lenght, -1., 1.])
        py.savefig('velocities' + str(i) + '.png')
        py.show()

        py.plot(self.actions)
        py.xlabel('Time (time steps)')
        py.ylabel('u')
        py.title('DDPG for AUV Control')
        py.legend(('1', '2','3', '4' ,'5','6'  ))
        #py.axis([0, simulation_lenght, -1., 1.])
        py.savefig('U' + str(i) + '.png')
        py.show()

        x0 = np.array([_[0] for _ in self.position])
        y0 = np.array([_[1] for _ in self.position])
        z0 = np.array([_[2] for _ in self.position])
        py.plot(x0,y0)
        py.xlabel('x')
        py.ylabel('y')
        py.title('DDPG for AUV Control')
        #py.legend(('x','y','z' ))
        #py.axis([0, simulation_lenght, -0.5, 0.5])
        py.savefig('pose' + str(i) + '.png')
        py.show()
        

        py.plot(x0,y0, 'b')
        py.hold()      # toggle hold
        py.hold(True)
        # plot origin
        py.plot(0.0,0.0, 'ob')
        #plot goal 
        scale = 0.
        xsp = self.set_point[0] - scale 
        ysp = self.set_point[1] - scale
        py.plot(xsp, ysp, 'or')
        # create and plot circle
        xc, yc = create_circle(xsp,ysp,1.)
        py.plot(xc, yc, 'r')


        py.xlabel('X [m]')
        py.ylabel('Y [m]')
        #py.title('Position Control Using Deep RL')
        #py.legend(('x','y','z' ))
        py.axis([-10., 10., -10., 10.])
        py.savefig('pose_for_abstract.png')
        py.show()
        self.save()

    def plot_map(self):
        for _ in range(len(self.done)):
            if self.done[_]:
                py.plot(self.wp[_][0],self.wp[_][1],'ob')
            else:
                py.plot(self.wp[_][0],self.wp[_][1],'xr')
        py.axis([-10., 10., -10., 10.])
        py.savefig('sparse_map.png')


    def reset(self, set_point):
        self.actions = deque()
        self.velocity = deque()
        self.position = deque()
        self.accelerations = deque()
        self.set_point = set_point

    def save(self):
        np.save('actions', self.actions)
        np.save('velocity', self.velocity)
        np.save('position', self.position)
        np.save('accelerations', self.accelerations)
        np.save('i', self.i)
        np.save('set_point', self.set_point)



    def load(self):
        self.actions = np.load('actions.npy')
        self.velocity = np.load('velocity.npy')
        self.position = np.load('position.npy')
        self.accelerations = np.load('accelerations.npy')
        self.i = np.load('i.npy')
        self.set_point = np.load('set_point.npy')
        return self.i
        
if __name__ == '__main__':
    plot = plotter()
    i = plot.load()
    plot.plot(i)


