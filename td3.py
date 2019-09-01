import tensorflow as tf
import numpy as np
import gym
from ou_noise import OUNoise

LAYER_1 = 400
LAYER_2 = 300
LAYER_3 = 300
keep_rate = 0.8
LAMBDA = 0.00001 # regularization term
GAMMA = 0.99
class TD3(object):


    def __init__(self, sess, state_dim, action_dim, max_action, min_action, actor_learning_rate, critic_learning_rate, tau, RANDOM_SEED, device = '/cpu:0'):

        self.sess = sess
        np.random.seed(RANDOM_SEED)
    	tf.set_random_seed(RANDOM_SEED)
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        self.device = device
        self.max_action = max_action
        self.min_action = min_action
        self.count = 0 # count for 
        # Placeholders
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='state')
        self.action = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='actions')
        scope = 'net'    
        self.v_1,self.v_2, self.a, self.scaled_a, self.saver = self._build_net(scope)
        self.a_params = tf.trainable_variables(scope=scope + '/actor')
        self.c_params_1 = tf.trainable_variables(scope=scope + '/critic_1')
        self.c_params_2 = tf.trainable_variables(scope=scope + '/critic_2')
        #self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        #self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        scope = 'target'    
        self.v_target_1, self.v_target_2, self.a_target, self.scaled_a_target, self.saver_target = self._build_net(scope)
        self.a_params_target = tf.trainable_variables(scope=scope + '/actor')
        self.c_params_target_1 = tf.trainable_variables(scope=scope + '/critic_1')
        self.c_params_target_2 = tf.trainable_variables(scope=scope + '/critic_2')
        #self.a_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        #self.c_params_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        
        with tf.variable_scope('learning_rate'): 
            # global step
            self.global_step = tf.Variable(0, trainable=False)
            self.actor_decay_learning_rate = tf.train.exponential_decay(self.actor_learning_rate, self.global_step, 100000, 0.96, staircase=True)
            self.critic_decay_learning_rate = tf.train.exponential_decay(self.critic_learning_rate, self.global_step, 100000, 0.96, staircase=True)
        
        with tf.device(self.device):
            # Op for periodically updating target network with online network
            # weights with regularization
            self.generate_param_updater()
           
            self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
            # Define loss and optimization Op
            self.v_target_min = tf.minimum(self.v_target_1,self.v_target_2)
            
            self.squared_1 = tf.square(tf.subtract(self.predicted_q_value,self.v_1))
            self.l2_loss_1 = tf.losses.get_regularization_loss(scope="net/critic_1")
            self.loss_1 = tf.reduce_mean(self.squared_1) + self.l2_loss_1 
            self.squared_2 = tf.square(tf.subtract(self.predicted_q_value,self.v_2))
            self.l2_loss_2 = tf.losses.get_regularization_loss(scope="net/critic_2")
            self.loss_2 = tf.reduce_mean(self.squared_2) + self.l2_loss_2
            self.loss = self.loss_1 + self.loss_2
            self.critic_optimize = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss, global_step=self.global_step) 
            

            self.action_grads = tf.gradients(self.v_1, self.action)[0]
            self.actor_gradients = tf.gradients(self.a, self.a_params, -self.action_grads)
            self.actor_optimize = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self.actor_gradients, self.a_params), global_step=self.global_step)
            
            # inverting gradients
            self.inverting_gradients_placeholder = tf.placeholder(tf.float32, shape=[None, self.a_dim], name='inverting_gradients')
            self._dq_da = tf.gradients(self.v_1, self.action)[0] # q, a 
            self._grad = tf.gradients(self.a, self.a_params, -self.inverting_gradients_placeholder)
            self._train_actor = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(zip(self._grad, self.a_params),global_step=self.global_step)
            
            


    def _build_net(self,scope):
       
        with tf.device(self.device):        
            with tf.variable_scope(scope + '/critic_1'):

                regularizer = tf.contrib.layers.l2_regularizer(scale=LAMBDA)
                l1 = tf.contrib.layers.fully_connected(self.inputs, LAYER_1, weights_regularizer=regularizer, activation_fn=tf.nn.leaky_relu)
                l2_a = tf.contrib.layers.fully_connected(self.action, LAYER_2, weights_regularizer=regularizer, activation_fn=None)
                l2_s = tf.contrib.layers.fully_connected(l1, LAYER_2, weights_regularizer=regularizer,activation_fn=None)
                l2 = tf.nn.leaky_relu(l2_s + l2_a)
                v_1 = tf.contrib.layers.fully_connected(l2, 1, weights_regularizer=regularizer, activation_fn=None)

            with tf.variable_scope(scope + '/critic_2'):
                l1 = tf.contrib.layers.fully_connected(self.inputs, LAYER_1, weights_regularizer=regularizer, activation_fn=tf.nn.leaky_relu)
                l2_a = tf.contrib.layers.fully_connected(self.action, LAYER_2, weights_regularizer=regularizer, activation_fn=None)
                l2_s = tf.contrib.layers.fully_connected(l1, LAYER_2, weights_regularizer=regularizer,activation_fn=None)
                l2 = tf.nn.leaky_relu(l2_s + l2_a)
                v_2 = tf.contrib.layers.fully_connected(l2, 1, weights_regularizer=regularizer, activation_fn=None)
				
            with tf.variable_scope(scope + '/actor'):
                l1 = tf.contrib.layers.fully_connected(self.inputs, LAYER_1,  activation_fn=tf.nn.leaky_relu) # tf.nn.leaky_relu tf.nn.relu
                l2 = tf.contrib.layers.fully_connected(l1, LAYER_2,  activation_fn=tf.nn.leaky_relu)
                w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                a = tf.contrib.layers.fully_connected(l2, self.a_dim, weights_initializer=w_init, activation_fn=None) # None  tf.nn.tanh
                scaled_a = a
                # scaled_a = tf.clip_by_value(a,self.min_action,self.max_action)#tf.multiply(a, self.action_bound)
                       
        saver = tf.train.Saver()
        return v_1, v_2, a, scaled_a, saver

    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch, MINIBATCH_SIZE):
        
        
        # get q target
        target_q = self.critic_predict_target(s2_batch, self.predict_action_target(s2_batch))
        # obtain y
        y_i = []
        for k in range(MINIBATCH_SIZE):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + GAMMA * target_q[k])
        # train critic
        LOSS = self.critic_train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
        # print(L2_LOSS)
        a_outs = self.predict_action(s_batch)
        self.actor_train(s_batch, a_outs)
        
        self.update_target_network()

        return

    def test_gradient(self, s_batch, a_batch, r_batch, t_batch, s2_batch, MINIBATCH_SIZE):
        
        
        # get q target
        target_q = self.critic_predict_target(s2_batch, self.predict_action_target(s2_batch))
        # obtain y
        y_i = []
        for k in range(MINIBATCH_SIZE):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + GAMMA * target_q[k])
        
        # train critic
        LOSS = self.critic_train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
        
        self.count +=1 
        
        if self.count == 2:
	        
	        self.count = 0
	        actions = self.predict_action(s_batch) 
	        noise = np.clip(0.2*np.random.randn(MINIBATCH_SIZE,self.a_dim) + 0.,-0.5,0.5)
	        actions = np.clip(actions + noise, self.min_action,self.max_action)
	        

	        # get dq/da array, action array
	        #print(upper, '***************')
	        dq_das = self.sess.run([self._dq_da], feed_dict={self.inputs: s_batch, self.action:actions})[0]
	        # inverting gradients, if dq_da >= 0, apply upper method, else lower method
	        inverting_gradients = []
	        #'''
	        # print('1 dq_das, actions',dq_das, actions)
	        '''
	        # print('dq_das, actions',dq_das, actions)
	        for dq_da, action in zip(dq_das, actions):
	            # print('dq_da, action',dq_da, action)
	            if dq_da >= 0.0:
	                inverting_gradients.append(dq_da * (self.max_action - action) / (self.max_action - self.min_action))
	            else:
	                inverting_gradients.append(dq_da * (action - self.min_action) / (self.max_action - self.min_action))
	        inverting_gradients = np.array(inverting_gradients).reshape(-1, 1)
			'''

	        for i in range(MINIBATCH_SIZE):
	            #print('2', i,dq_das[i])
	            for j in range(self.a_dim):
	                if dq_das[i][j] >= 0.0:
	                    dq_das[i][j] = dq_das[i][j] * (self.max_action - actions[i][j]) / (self.max_action - self.min_action)
	                else:
	                    dq_das[i][j] = dq_das[i][j] * (actions[i][j] - self.min_action) / (self.max_action - self.min_action)
	        
	        # print(dq_das,inverting_gradients)
	        # exit()
	        inverting_gradients = dq_das 
	       
	        self.sess.run(self._train_actor, feed_dict={self.inputs: s_batch, self.inverting_gradients_placeholder: inverting_gradients})

        	self.update_target_network()

        return

    
    def _critic_train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.action_grads], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })


    def update_target_network(self):
        self.sess.run([self.a_updater,self.c_updater_1,self.c_updater_2])

    def generate_param_updater(self):
        self.a_updater = [self.a_params_target[i].assign(tf.multiply(self.a_params[i], self.tau) + tf.multiply(self.a_params_target[i], 1. - self.tau))
                for i in range(len(self.a_params))]
        self.c_updater_1 = [self.c_params_target_1[i].assign(tf.multiply(self.c_params_1[i], self.tau) + tf.multiply(self.c_params_target_1[i], 1. - self.tau))
                for i in range(len(self.c_params_1))]
        self.c_updater_2 = [self.c_params_target_2[i].assign(tf.multiply(self.c_params_2[i], self.tau) + tf.multiply(self.c_params_target_2[i], 1. - self.tau))
                for i in range(len(self.c_params_2))]

    def critic_train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.loss,self.critic_optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })


    def actor_train(self,inputs, action):
        return self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def save(self):
        self.saver.save(self.sess,"./model/model.ckpt")
        self.saver_target.save(self.sess,"./model/model_target.ckpt")
        print("Model saved in file: actor_model")

    
    def load(self):
        self.saver.restore(self.sess,"./model/model.ckpt")
        self.saver_target.restore(self.sess,"./model/model_target.ckpt")
        


    def critic_predict_target(self, state, action):
        return self.sess.run(self.v_target_min, feed_dict={
            self.inputs: state,
            self.action: action
        })
        
    def predict_action_target(self, state):
        return self.sess.run(self.scaled_a_target, feed_dict={
            self.inputs: state
        })

    def predict_action(self, state):
        return self.sess.run(self.scaled_a, feed_dict={
            self.inputs: state
        })




if __name__ == '__main__':
    from replay_buffer import ReplayBuffer
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE =  0.0001
    # Soft target update param
    TAU = 0.005
    DEVICE = '/cpu:0'
    # ENV_NAME = 'MountainCarContinuous-v0'
    ENV_NAME = 'Pendulum-v0'
    # import gym_foo
    # ENV_NAME = 'nessie_end_to_end-v0'
    max_action = 2.
    min_action = -2.
    epochs = 500
    epsilon = 1.0
    min_epsilon = 0.1
    EXPLORE = 200
    BUFFER_SIZE = 100000
    RANDOM_SEED = 51234
    MINIBATCH_SIZE = 100# 32 # 5
    FILL = 1000
    with tf.Session() as sess:
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        low = TD3(sess, state_dim, action_dim, max_action, min_action, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, TAU, RANDOM_SEED, device=DEVICE)
        sess.run(tf.global_variables_initializer())
        # check that we are effectively updating the parameters
        #print(low.a_params_target[0].eval()[0][0],low.a_params[0].eval()[0][0])
        #low.update_target_network()
        #print(low.a_params_target[0].eval()[0][0],low.a_params[0].eval()[0][0])
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        ruido = OUNoise(action_dim, mu = 0.0)
        for i in range(epochs):
            state = env.reset()
            done = False
            epsilon -= (epsilon/EXPLORE)
            epsilon = np.maximum(min_epsilon,epsilon)
            episode_r = 0.
            step = 0
            while (not done):
                step += 1
                action = low.predict_action(np.reshape(state,(1,state_dim)))
                action1 = action
                action = np.clip(action,min_action,max_action)
                action = action + max(epsilon,0)*ruido.noise()
                action = np.clip(action,min_action,max_action)
                # print(action1, action)
                next_state, reward, done, info = env.step(action)
                reward = reward + 1. 
                # reward = np.clip(reward,-1.,1.)
                replay_buffer.add(np.reshape(state, (state_dim,)), np.reshape(action, (action_dim,)), reward,
                                      done, np.reshape(next_state, (state_dim,)))
                state = next_state
                episode_r = episode_r + reward
                if replay_buffer.size() > MINIBATCH_SIZE + FILL:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                    #low.train(s_batch, a_batch, r_batch, t_batch, s2_batch,MINIBATCH_SIZE)
                    low.test_gradient(s_batch, a_batch, r_batch, t_batch, s2_batch,MINIBATCH_SIZE)
                else: 
                    epsilon  = 1.
            print(i, step, 'last r', round(reward,3), 'episode reward',round(episode_r,3), 'epsilon', round(epsilon,3))                


        low.save()