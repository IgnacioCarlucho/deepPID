import numpy as np

class pid_controller():
    def __init__(self,set_point,n_actions=6, dt = 0.1):
        self.dt = dt
        self.actions=n_actions
        self.set_point = set_point
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))

    def update_action(self,actions):
          
        self.action_vx = actions[0:3] # PIDs[0:2]
        self.action_wz = actions[3:6] # PIDs[2:4]
          
        return 

    def update(self, velocity):
    	self.velocity = velocity
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]  
        self.error[0][0] = self.set_point[0] - self.velocity[0] # vx
        self.error[0][1] = self.set_point[5] - self.velocity[5] # wz

        self.u[0] = self.controller_pid(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
        self.u[1] = self.controller_pid(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
        self.u = np.clip(self.u, -0.8, 0.8)
        self.u0[0] = self.u[0]
        self.u0[1] = self.u[1]

        return self.u

    def controller_pid_2(self, et, et1, et2, action, u0):
        # this is the old version, it was not working properly 
        Kp = action[0]
        Ti = action[1]
        Td = action[2] 

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def controller_pid(self, et, et1, et2, action, u0):
        
        Kp = action[0]
        Ti = action[1]
        Td = action[2] 

        k1 = Kp*(1 + self.dt/Ti + Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt)
        k3 = Kp*(Td/self.dt)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def reset(self):
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))
        return



class pi_controller():

    def __init__(self,set_point,n_actions=6, dt = 0.1):
        self.dt = dt
        self.actions=n_actions
        self.set_point = set_point
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))

    def update_action(self,actions):
          
        self.action_vx = actions[0:2] # PIDs[0:2]
        self.action_wz = actions[2:4] # PIDs[2:4]
          
        return 

    def update(self, velocity):
    	self.velocity = velocity
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]  
        self.error[0][0] = self.set_point[0] - self.velocity[0] # vx
        self.error[0][1] = self.set_point[5] - self.velocity[5] # wz

        self.u[0] = self.controller_pi(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
        self.u[1] = self.controller_pi(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
        self.u = np.clip(self.u, -0.8, 0.8)
        self.u0[0] = self.u[0]
        self.u0[1] = self.u[1]

        return self.u

    def controller_pi(self, et, et1, et2, action, u0):
        
        Kp = action[0]
        Ti = action[1]
        Td = 0.

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2
       
        return u
   

    def reset(self):
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))
        return



class pi_controller_pioneer():

    def __init__(self,set_point,n_actions=6, dt = 0.1):
        self.dt = dt
        self.actions=n_actions
        self.set_point = set_point
        self.action_vx = np.zeros(2)
        self.action_wz = np.zeros(2)
        self.u = np.zeros(2)
        self.u0 = np.zeros(2)
        self.error = np.zeros((3,2))

    def update_action(self,actions):
          
        self.action_vx = actions[0:2] # PIDs[0:2]
        self.action_wz = actions[2:4] # PIDs[2:4]
          
        return 

    def update(self, velocity):
        self.velocity = velocity
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]  
        self.error[0][0] = self.set_point[0] - self.velocity[0] # vx
        self.error[0][1] = self.set_point[1] - self.velocity[1] # wz

        self.u[0] = self.controller_pi(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
        self.u[1] = self.controller_pi(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
        self.u = np.clip(self.u, -0.8, 0.8)
        self.u0[0] = self.u[0]
        self.u0[1] = self.u[1]

        return self.u

    def controller_pi(self, et, et1, et2, action, u0):
        # this is the old version
        Kp = action[0]
        Ti = action[1]
        Td = 0.

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2
       
        return u

    def controller_pi_2(self, et, et1, et2, action, u0):
        
        Kp = action[0]
        Ti = action[1]
        Td = 0.

        k1 = Kp*(1 + self.dt/Ti + Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt)
        k3 = Kp*(Td/self.dt)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u
   

    def reset(self):
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))
        return






class pid_controller_pioneer():
    def __init__(self,set_point,n_actions=6, dt = 0.1):
        self.dt = dt
        self.actions=n_actions
        self.set_point = set_point
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(2)
        self.u0 = np.zeros(2)
        self.error = np.zeros((3,2))

    def update_action(self,actions):
          
        self.action_vx = actions[0:3] # PIDs[0:2]
        self.action_wz = actions[3:6] # PIDs[2:4]
          
        return 

    def update(self, velocity):
        self.velocity = velocity
        self.error[2] = self.error[1]
        self.error[1] = self.error[0]  
        self.error[0][0] = self.set_point[0] - self.velocity[0] # vx
        self.error[0][1] = self.set_point[1] - self.velocity[1] # wz

        self.u[0] = self.controller_pid(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
        self.u[1] = self.controller_pid(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
        self.u = np.clip(self.u, -0.8, 0.8)
        self.u0[0] = self.u[0]
        self.u0[1] = self.u[1]

        return self.u

    def controller_pid_2(self, et, et1, et2, action, u0):
        # this is the old version, it was not working properly 
        Kp = action[0]
        Ti = action[1]
        Td = action[2] 

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def controller_pid(self, et, et1, et2, action, u0):
        
        Kp = action[0]
        Ti = action[1]
        Td = action[2] 

        k1 = Kp*(1 + self.dt/Ti + Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt)
        k3 = Kp*(Td/self.dt)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def reset(self):
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.u = np.zeros(6)
        self.u0 = np.zeros(6)
        self.error = np.zeros((3,2))
        return



       